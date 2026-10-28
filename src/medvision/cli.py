"""Interface en ligne de commande.

medvision download-model                  télécharge le modèle de détection des mains
medvision record lumiere -n 5 --preview   enregistre un geste (5 exemples)
medvision list                            liste les gestes et la qualité de leurs exemples
medvision delete lumiere                  supprime un geste
medvision run --preview --dry-run         teste la reconnaissance sans MQTT
medvision run -c config.yaml              mode normal : publie vers Home Assistant
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import numpy as np

from medvision import __version__
from medvision.config import Config
from medvision.store import GestureStore, validate_name

log = logging.getLogger("medvision")


# --------------------------------------------------------------------------- utilitaires
def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class _Preview:
    """Fenêtre d'aperçu facultative ; se désactive seule si aucun écran n'est disponible."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def show(self, frame, hands, status: str) -> bool:
        """Affiche l'image ; retourne False si l'utilisateur appuie sur 'q' ou Échap."""
        if not self.enabled:
            return True
        import cv2

        from medvision.vision import draw_preview

        try:
            cv2.imshow("MedVision (q pour quitter)", draw_preview(frame, hands, status))
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            log.warning("Aperçu indisponible (pas d'écran) : désactivé.")
            self.enabled = False
            return True
        return key not in (ord("q"), 27)

    def close(self) -> None:
        if self.enabled:
            import cv2

            cv2.destroyAllWindows()


# --------------------------------------------------------------------------- commandes
def cmd_download_model(cfg: Config, args) -> int:
    from medvision.vision import download_model

    print(f"Modèle disponible : {download_model(cfg.model_path)}")
    return 0


def cmd_list(cfg: Config, args) -> int:
    store = GestureStore(cfg.gestures_dir)
    names = store.list()
    if not names:
        print(f"Aucun geste dans {store.dir}/ — commencez par : medvision record <nom>")
        return 0
    lib = store.build_library(cfg.default_threshold)
    labels = store.labels()
    print(f"{'Geste':<20}{'Libellé':<28}{'Ex.':>4}  {'Seuil':>6}  Qualité")
    for n in names:
        count = len(lib.templates[n])
        print(f"{n:<20}{labels[n][:27]:<28}{count:>4}  {lib.thresholds[n]:>6.3f}  {lib.quality(n)}")
    if cfg.attention_gesture and cfg.attention_gesture not in names:
        print(f"\nAttention : le geste d'attention {cfg.attention_gesture!r} n'est pas enregistré.")
    return 0


def cmd_delete(cfg: Config, args) -> int:
    ok = GestureStore(cfg.gestures_dir).delete(args.name)
    print("Supprimé." if ok else "Geste introuvable.")
    return 0 if ok else 1


def cmd_record(cfg: Config, args) -> int:
    from medvision.vision import Camera, HandTracker

    name = validate_name(args.name)
    store = GestureStore(cfg.gestures_dir)
    cam, tracker, preview = Camera(cfg), HandTracker(cfg), _Preview(args.preview)
    samples: list[np.ndarray] = []
    target_frames = max(5, int(args.duration * cfg.fps))
    print(f"\nEnregistrement du geste « {name} » : {args.samples} exemples de {args.duration} s.")
    print("Placez-vous face à la caméra, la main bien visible.\n")
    try:
        while len(samples) < args.samples:
            print(f"Exemple {len(samples) + 1}/{args.samples} — prêt…", end="", flush=True)
            end = time.monotonic() + args.countdown
            while time.monotonic() < end:  # compte à rebours avec aperçu vivant
                frame = cam.read()
                if frame is not None:
                    left = int(end - time.monotonic()) + 1
                    if not preview.show(frame, tracker.detect(frame), f"Pret... {left}"):
                        return 130
            print(" GO !", flush=True)
            seq, seen = [], 0
            while seen < target_frames:
                frame = cam.read()
                if frame is None:
                    continue
                seen += 1
                hands = tracker.detect(frame)
                if hands:
                    seq.append(hands[0].features)
                if not preview.show(frame, hands, f"Enregistrement {seen}/{target_frames}"):
                    return 130
            if len(seq) < cfg.min_hand_ratio * target_frames:
                print(f"  Main visible sur {len(seq)}/{target_frames} images : on recommence.")
                continue
            samples.append(np.stack(seq))
            print(f"  OK ({len(seq)} images avec main)")
    except KeyboardInterrupt:
        print("\nInterrompu.")
        if not samples:
            return 130
    finally:
        cam.close()
        tracker.close()
        preview.close()

    path = store.save(name, samples, label=args.label or "", append=not args.replace)
    lib = store.build_library(cfg.default_threshold)
    print(f"\nEnregistré dans {path} ({len(lib.templates[name])} exemples au total).")
    print(f"Qualité : {lib.quality(name)} — seuil calibré : {lib.thresholds[name]:.3f}")
    print("Astuce : enregistrez aussi un geste « repos » (main immobile) pour limiter les erreurs.")
    return 0


def cmd_run(cfg: Config, args) -> int:
    from medvision.engine import GestureEngine, State
    from medvision.vision import Camera, HandTracker, MotionGate

    if args.dry_run:
        cfg.mqtt_enabled = False
    store = GestureStore(cfg.gestures_dir)
    lib = store.build_library(cfg.default_threshold)
    if not lib.names:
        log.error(
            "Aucun geste enregistré dans %s. Lancez d'abord : medvision record <nom>", store.dir
        )
        return 1
    log.info("Gestes chargés : %s", ", ".join(lib.names))

    reload_flag, stop_flag = threading.Event(), threading.Event()
    ha = None
    if cfg.mqtt_enabled:
        from medvision.mqtt import HomeAssistantMqtt

        ha = HomeAssistantMqtt(
            cfg, on_command=lambda c: reload_flag.set() if c == "reload" else None
        )
        ha.start([n for n in lib.names if n not in cfg.rest_set], store.labels())

    status = {"text": "Veille"}

    def on_gesture(ev):
        status["text"] = f"Geste : {ev.name} ({ev.confidence:.0%})"
        if ha:
            ha.publish_gesture(ev)
        else:
            print(f">>> GESTE : {ev.name}  (confiance {ev.confidence:.0%})", flush=True)

    def on_state(st):
        status["text"] = "En attente d'une commande..." if st == State.ARMED else "Veille"
        if ha:
            ha.publish_state(st)

    engine = GestureEngine(
        lib, cfg, on_gesture=on_gesture, on_state=on_state,
        on_presence=(ha.publish_presence if ha else None),
    )  # fmt: skip

    signal.signal(signal.SIGTERM, lambda *_: stop_flag.set())  # arrêt propre (add-on, Docker)
    cam, tracker, preview = Camera(cfg), HandTracker(cfg), _Preview(args.preview)
    gate = MotionGate(cfg.motion_threshold, cfg.active_timeout) if cfg.motion_gating else None
    stats = {"frames": 0, "analysed": 0, "since": time.monotonic()}
    try:
        while not stop_flag.is_set():
            if reload_flag.is_set():
                reload_flag.clear()
                engine.lib = lib = store.build_library(cfg.default_threshold)
                engine.reset()
                if ha:
                    ha.update_gestures(
                        [n for n in lib.names if n not in cfg.rest_set], store.labels()
                    )
                log.info("Gestes rechargés : %s", ", ".join(lib.names))

            frame = cam.read()
            if frame is None:
                continue
            stats["frames"] += 1
            hands = []
            if gate is None or gate.should_process(frame):
                stats["analysed"] += 1
                hands = tracker.detect(frame)
                if hands and gate:
                    gate.keep_alive()
            engine.push(hands[0].features if hands else None)
            if not preview.show(frame, hands, status["text"]):
                break

            if time.monotonic() - stats["since"] > 60:
                f, a = stats["frames"], stats["analysed"]
                log.info("Performance : %.1f img/s, %d %% analysées par le réseau de neurones",
                         f / 60, 100 * a / max(1, f))  # fmt: skip
                stats.update(frames=0, analysed=0, since=time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Arrêt.")
        cam.close()
        tracker.close()
        preview.close()
        if ha:
            ha.stop()
    return 0


# --------------------------------------------------------------------------- point d'entrée
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medvision", description="Commande domotique par gestes.")
    p.add_argument("-V", "--version", action="version", version=f"medvision {__version__}")
    p.add_argument("-c", "--config", help="fichier YAML (ou options.json d'un add-on)")
    p.add_argument("--camera", help="remplace camera_source (index webcam ou URL RTSP)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download-model", help="télécharger le modèle de détection des mains")
    sub.add_parser("list", help="lister les gestes enregistrés")

    d = sub.add_parser("delete", help="supprimer un geste")
    d.add_argument("name")

    r = sub.add_parser("record", help="enregistrer un nouveau geste")
    r.add_argument("name", help="identifiant : minuscules, chiffres, - ou _")
    r.add_argument("-n", "--samples", type=int, default=5, help="nombre d'exemples (défaut 5)")
    r.add_argument("-d", "--duration", type=float, default=1.5, help="durée d'un exemple (s)")
    r.add_argument("--countdown", type=float, default=3.0, help="compte à rebours (s)")
    r.add_argument("--label", help="libellé lisible, ex. « Allumer le salon »")
    r.add_argument("--replace", action="store_true", help="remplacer les exemples existants")
    r.add_argument("--preview", action="store_true", help="afficher la caméra")

    run = sub.add_parser("run", help="lancer la reconnaissance")
    run.add_argument("--preview", action="store_true", help="afficher la caméra")
    run.add_argument("--dry-run", action="store_true", help="sans MQTT, affiche les gestes")
    return p


COMMANDS = {
    "download-model": cmd_download_model,
    "list": cmd_list,
    "delete": cmd_delete,
    "record": cmd_record,
    "run": cmd_run,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        cfg = Config.load(args.config)
    except (OSError, ValueError) as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        sys.exit(2)
    if args.camera is not None:
        cfg.camera_source = args.camera
    _setup_logging(cfg.log_level)
    try:
        sys.exit(COMMANDS[args.command](cfg, args))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

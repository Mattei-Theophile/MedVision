"""Interface en ligne de commande.

medvision sensors                         liste les capteurs pris en charge
medvision download-model                  télécharge le modèle de détection des mains (caméra)
medvision record lumiere -n 5 --preview   enregistre un geste (5 exemples) avec le capteur actif
medvision list                            liste les gestes et la qualité de leurs exemples
medvision delete lumiere                  supprime un geste
medvision run --preview --dry-run         teste la reconnaissance sans MQTT
medvision run --web                       … avec l'interface web sur http://127.0.0.1:8099
medvision --sensor ld2450 run             idem avec le radar mmWave
medvision -c config.yaml run              mode normal : publie vers Home Assistant
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

from medvision import __version__
from medvision.config import Config
from medvision.features import FORMAT_LABELS, HAND_FORMAT
from medvision.store import GestureStore

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

    def show(self, image) -> bool:
        """Affiche l'image ; retourne False si l'utilisateur appuie sur 'q' ou Échap."""
        if not self.enabled or image is None:
            return True
        import cv2

        try:
            cv2.imshow("MedVision (q pour quitter)", image)
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


def _active_format(cfg: Config) -> str:
    from medvision.sensors import sensor_class

    return sensor_class(cfg.sensor_type).feature_format or HAND_FORMAT


# --------------------------------------------------------------------------- commandes
def cmd_sensors(cfg: Config, args) -> int:
    from medvision.sensors import available_sensors

    for kind, desc in available_sensors().items():
        mark = "*" if kind == cfg.sensor_type else " "
        print(f"{mark} {kind:<12}{desc}")
    print("\n* = capteur actif (option sensor_type ou --sensor)")
    return 0


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
    active = _active_format(cfg)
    lib = store.build_library(cfg.default_threshold, active)
    print(f"{'Geste':<20}{'Libellé':<28}{'Capteur':<22}{'Ex.':>4}  {'Seuil':>6}  Qualité")
    for n in names:
        data = store.load_one(n)
        fmt, label = data["format"], data.get("label", n)
        sensor = FORMAT_LABELS.get(fmt, fmt)[:21]
        if fmt == active:
            thr, quality = f"{lib.thresholds[n]:>6.3f}", lib.quality(n)
        else:
            thr, quality = f"{'-':>6}", "(autre capteur)"
        print(f"{n:<20}{label[:27]:<28}{sensor:<22}{len(data['samples']):>4}  {thr}  {quality}")
    if cfg.attention_gesture and cfg.attention_gesture not in lib.names:
        print(
            f"\nAttention : le geste d'attention {cfg.attention_gesture!r} n'est pas enregistré"
            " pour ce capteur."
        )
    return 0


def cmd_delete(cfg: Config, args) -> int:
    ok = GestureStore(cfg.gestures_dir).delete(args.name)
    print("Supprimé." if ok else "Geste introuvable.")
    return 0 if ok else 1


def cmd_record(cfg: Config, args) -> int:
    from medvision.recorder import Phase, Recorder, RecordingRequest
    from medvision.sensors import create_sensor

    req = RecordingRequest(
        name=args.name,
        label=args.label or "",
        samples=args.samples,
        duration=args.duration,
        countdown=args.countdown,
        replace=args.replace,
    )
    store = GestureStore(cfg.gestures_dir)
    sensor, preview = create_sensor(cfg), _Preview(args.preview)
    print(f"\nEnregistrement du geste « {req.name} » avec le capteur {sensor.kind} : "
          f"{req.samples} exemples de {req.duration} s.")  # fmt: skip
    print("Placez-vous face au capteur.\n")
    rec, last_msg = None, None
    try:
        sensor.open()
        rec = Recorder(req, min_ratio=cfg.min_hand_ratio, start=time.monotonic())
        while not rec.finished:
            obs = sensor.read()
            if obs is None:
                continue
            rec.push(obs, time.monotonic())
            if rec.message != last_msg:
                print(rec.message, flush=True)
                last_msg = rec.message
            if not preview.show(sensor.render(obs, rec.status_line())):
                rec.cancel()
    except KeyboardInterrupt:
        print("\nInterrompu.")
        if rec:
            rec.cancel()
    finally:
        sensor.close()
        preview.close()

    if rec is None or not rec.captured:
        return 130 if rec and rec.phase == Phase.CANCELLED else 1
    if rec.phase != Phase.DONE:
        print(f"{len(rec.captured)} exemple(s) conservé(s) malgré l'interruption.")
    path = store.save(
        req.name,
        rec.captured,
        label=req.label,
        append=not req.replace,
        feature_format=sensor.feature_format,
    )
    lib = store.build_library(cfg.default_threshold, sensor.feature_format)
    print(f"\nEnregistré dans {path} ({len(lib.templates[req.name])} exemples au total).")
    print(f"Qualité : {lib.quality(req.name)} — seuil calibré : {lib.thresholds[req.name]:.3f}")
    print("Astuce : enregistrez aussi un geste « repos » (position neutre) contre les erreurs.")
    return 0


def cmd_run(cfg: Config, args) -> int:
    from medvision.runtime import Runtime
    from medvision.sensors import create_sensor

    if getattr(args, "dry_run", False):
        cfg.mqtt_enabled = False
    if getattr(args, "web", False):
        cfg.web_enabled = True

    sensor = create_sensor(cfg)
    runtime_ref: dict = {}
    ha = None
    if cfg.mqtt_enabled:
        from medvision.mqtt import HomeAssistantMqtt

        def on_command(cmd: str) -> None:
            if cmd == "reload" and "rt" in runtime_ref:
                runtime_ref["rt"].request_reload()

        ha = HomeAssistantMqtt(cfg, on_command=on_command, sensor=sensor)

    rt = Runtime(cfg, sensor=sensor, publisher=ha, echo=ha is None)
    runtime_ref["rt"] = rt
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())  # arrêt propre (add-on, Docker)
    preview, web = _Preview(getattr(args, "preview", False)), None
    try:
        rt.open()
        if not rt.lib.names:
            if not cfg.web_enabled:
                log.error(
                    "Aucun geste enregistré pour le capteur %s dans %s. "
                    "Lancez d'abord : medvision record <nom>  (ou run --web)",
                    sensor.kind,
                    rt.store.dir,
                )
                return 1
            log.warning("Aucun geste pour l'instant : enregistrez-les depuis l'interface web.")
        if cfg.web_enabled:
            from medvision.web import WebServer

            web = WebServer(
                rt, cfg.web_host, cfg.web_port, getattr(args, "behind_ingress", False)
            ).start()
        rt.run(stop, preview)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Arrêt.")
        if web:
            web.stop()
        rt.close()
        preview.close()
    return 0


# --------------------------------------------------------------------------- point d'entrée
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medvision", description="Commande domotique par gestes.")
    p.add_argument("-V", "--version", action="version", version=f"medvision {__version__}")
    p.add_argument("-c", "--config", help="fichier YAML (ou options.json d'un add-on)")
    p.add_argument("--sensor", help="remplace sensor_type (camera, ld2450, simulation…)")
    p.add_argument("--camera", help="remplace camera_source (index webcam ou URL RTSP)")
    p.add_argument("--radar-port", help="remplace radar_port (/dev/ttyUSB0, socket://ip:port)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("sensors", help="lister les capteurs pris en charge")
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
    r.add_argument("--preview", action="store_true", help="afficher l'aperçu du capteur")

    run = sub.add_parser("run", help="lancer la reconnaissance")
    run.add_argument("--preview", action="store_true", help="afficher l'aperçu du capteur")
    run.add_argument("--dry-run", action="store_true", help="sans MQTT, affiche les gestes")
    run.add_argument("--web", action="store_true", help="interface web (web_host:web_port)")
    return p


COMMANDS = {
    "sensors": cmd_sensors,
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
        if args.sensor is not None:
            cfg.sensor_type = args.sensor
        if args.camera is not None:
            cfg.camera_source = args.camera
        if args.radar_port is not None:
            cfg.radar_port = args.radar_port
        cfg.validate()
    except (OSError, ValueError) as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        sys.exit(2)
    _setup_logging(cfg.log_level)
    try:
        sys.exit(COMMANDS[args.command](cfg, args))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

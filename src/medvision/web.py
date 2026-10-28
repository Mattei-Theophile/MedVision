"""Interface web locale : état en direct, enregistrement et gestion des gestes.

Serveur HTTP de la bibliothèque standard (aucune dépendance), dans un fil séparé. Dans
l'add-on, Home Assistant l'affiche dans sa barre latérale via *ingress* (authentification
assurée par Home Assistant). Hors add-on, il écoute par défaut sur 127.0.0.1 seulement.

Aucune image ne transite : la vue en direct ne reçoit que des points (main) ou des
positions de cibles (radar).

API (JSON) :
    GET    api/state              état, capteur, vue en direct, enregistrement en cours
    GET    api/gestures           gestes enregistrés
    POST   api/record             {name, label, samples, duration, countdown, replace}
    POST   api/record/cancel
    POST   api/reload
    DELETE api/gestures/<nom>
Les requêtes qui modifient quelque chose exigent `Content-Type: application/json` : un
site tiers ne peut pas les forger depuis le navigateur sans requête préalable CORS, que ce
serveur refuse.
"""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import unquote, urlsplit

log = logging.getLogger(__name__)

MAX_BODY = 16 * 1024


def _jsonable(obj):
    """Types numpy (adapters tiers) -> types JSON natifs."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} non sérialisable")


def _index_html() -> bytes:
    return resources.files("medvision").joinpath("static/index.html").read_bytes()


class _Handler(BaseHTTPRequestHandler):
    server_version = "MedVision"
    runtime = None  # remplacé par make_server()

    # ------------------------------------------------------------- utilitaires
    def log_message(self, fmt, *args):  # journal HTTP en DEBUG seulement
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=_jsonable).encode()
        self._send(status, body, "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _route(self) -> str:
        # Sous ingress, Home Assistant retire déjà le préfixe /api/hassio_ingress/<jeton>.
        return unquote(urlsplit(self.path).path).rstrip("/") or "/"

    def _body(self) -> dict | None:
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type application/json requis")
            return None
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "requête trop volumineuse")
            return None
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._error(HTTPStatus.BAD_REQUEST, "JSON invalide")
            return None
        if not isinstance(data, dict):
            self._error(HTTPStatus.BAD_REQUEST, "objet JSON attendu")
            return None
        return data

    # ------------------------------------------------------------- routes
    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_GET(self):  # noqa: N802
        route = self._route()
        if route in ("/", "/index.html"):
            self._send(200, _index_html(), "text/html; charset=utf-8")
        elif route == "/api/state":
            self._json(self.runtime.snapshot())
        elif route == "/api/gestures":
            self._json({"gestures": self.runtime.gestures()})
        else:
            self._error(HTTPStatus.NOT_FOUND, "introuvable")

    def do_POST(self):  # noqa: N802
        route = self._route()
        body = self._body()
        if body is None:
            return
        try:
            if route == "/api/record":
                allowed = {"name", "label", "samples", "duration", "countdown", "replace"}
                kwargs = {k: v for k, v in body.items() if k in allowed}
                if "replace" in kwargs:
                    kwargs["replace"] = bool(kwargs["replace"])
                self._json(self.runtime.start_recording(**kwargs), HTTPStatus.ACCEPTED)
            elif route == "/api/record/cancel":
                self.runtime.cancel_recording()
                self._json({"ok": True})
            elif route == "/api/reload":
                self.runtime.request_reload()
                self._json({"ok": True})
            else:
                self._error(HTTPStatus.NOT_FOUND, "introuvable")
        except (TypeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_DELETE(self):  # noqa: N802
        route = self._route()
        if not route.startswith("/api/gestures/"):
            return self._error(HTTPStatus.NOT_FOUND, "introuvable")
        if self._body() is None:
            return
        try:
            ok = self.runtime.delete_gesture(route.removeprefix("/api/gestures/"))
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        if ok:
            self._json({"ok": True})
        else:
            self._error(HTTPStatus.NOT_FOUND, "geste introuvable")

    def do_OPTIONS(self):  # noqa: N802 - pas de CORS : on refuse les requêtes préalables
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "CORS non autorisé")


class WebServer:
    def __init__(self, runtime, host: str, port: int, behind_ingress: bool = False):
        self.behind_ingress = behind_ingress
        handler = type("Handler", (_Handler,), {"runtime": runtime})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        return self.httpd.server_address[:2]

    def start(self) -> WebServer:
        self.thread.start()
        host, port = self.address
        log.info("Interface web : http://%s:%s/", host, port)
        if not self.behind_ingress and host not in ("127.0.0.1", "localhost", "::1"):
            log.warning("L'interface web est accessible sur le réseau, sans mot de passe.")
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

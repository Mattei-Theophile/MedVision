import pytest

from medvision.cli import build_parser, cmd_list, cmd_sensors
from medvision.config import Config
from medvision.store import GestureStore


def test_parser_commands():
    p = build_parser()
    args = p.parse_args(["record", "lumiere", "-n", "3", "--preview"])
    assert args.command == "record" and args.samples == 3 and args.preview
    assert p.parse_args(["run", "--dry-run"]).dry_run
    args = p.parse_args(["--sensor", "ld2450", "--radar-port", "socket://h:1", "run", "--web"])
    assert args.sensor == "ld2450" and args.radar_port == "socket://h:1" and args.web
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_list_command(tmp_path, gesture, capsys):
    cfg = Config(gestures_dir=str(tmp_path))
    assert cmd_list(cfg, None) == 0
    assert "Aucun geste" in capsys.readouterr().out
    GestureStore(tmp_path).save("lumiere", [gesture("poing", seed=s) for s in range(4)])
    cmd_list(cfg, None)
    out = capsys.readouterr().out
    assert "lumiere" in out and "bon" in out


def test_sensors_command(capsys):
    assert cmd_sensors(Config(sensor_type="ld2450"), None) == 0
    out = capsys.readouterr().out
    assert "* ld2450" in out and "camera" in out


def test_list_shows_other_sensor_gestures(tmp_path, gesture, capsys):
    import numpy as np

    from medvision.features import RADAR_FORMAT

    store = GestureStore(tmp_path)
    store.save("lumiere", [gesture("poing", seed=s) for s in range(3)])
    store.save("approche", [np.zeros((8, 3))] * 2, feature_format=RADAR_FORMAT)
    cmd_list(Config(gestures_dir=str(tmp_path)), None)
    out = capsys.readouterr().out
    assert "(autre capteur)" in out and "radar" in out

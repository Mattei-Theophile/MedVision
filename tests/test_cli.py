import pytest

from medvision.cli import build_parser, cmd_list
from medvision.config import Config
from medvision.store import GestureStore


def test_parser_commands():
    p = build_parser()
    args = p.parse_args(["record", "lumiere", "-n", "3", "--preview"])
    assert args.command == "record" and args.samples == 3 and args.preview
    assert p.parse_args(["run", "--dry-run"]).dry_run
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

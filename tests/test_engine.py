import pytest

from medvision.classifier import GestureLibrary
from medvision.config import Config
from medvision.engine import GestureEngine, State


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _setup(gesture, attention=""):
    lib = GestureLibrary()
    for kind in ("poing", "vague", "pouce"):
        lib.add(kind, [gesture(kind, seed=s) for s in range(5)])
    cfg = Config(window_frames=20, evaluate_every=1, stable_hits=1, attention_gesture=attention)
    clock = FakeClock()
    events, states, presence = [], [], []
    eng = GestureEngine(
        lib,
        cfg,
        on_gesture=events.append,
        on_state=states.append,
        on_presence=presence.append,
        clock=clock,
    )
    return eng, clock, events, states, presence


def _play(eng, clock, seq, dt=1 / 15):
    for frame in seq:
        clock.t += dt
        eng.push(frame)


def test_direct_mode_emits_once_with_cooldown(gesture):
    eng, clock, events, _, presence = _setup(gesture)
    _play(eng, clock, gesture("vague", seed=50))
    assert [e.name for e in events] == ["vague"]
    assert presence == [True]
    # Pendant le cooldown, rien n'est émis même si le geste continue.
    _play(eng, clock, gesture("vague", seed=51)[:10])
    assert len(events) == 1


def test_no_hand_means_no_event(gesture):
    eng, clock, events, _, presence = _setup(gesture)
    _play(eng, clock, gesture("vague", seed=1)[:5])
    _play(eng, clock, [None] * 30)
    assert events == []
    assert presence == [True, False]


def test_attention_gesture_required(gesture):
    eng, clock, events, states, _ = _setup(gesture, attention="pouce")
    # Commande sans attention : ignorée.
    _play(eng, clock, gesture("vague", seed=60))
    assert events == []
    clock.t += 2
    # Attention puis commande : acceptée.
    _play(eng, clock, gesture("pouce", seed=61))
    assert eng.state == State.ARMED
    clock.t += 2
    _play(eng, clock, gesture("vague", seed=62))
    assert [e.name for e in events] == ["vague"]
    assert states == [State.ARMED, State.IDLE]


def test_armed_timeout(gesture):
    eng, clock, events, states, _ = _setup(gesture, attention="pouce")
    _play(eng, clock, gesture("pouce", seed=70))
    assert eng.state == State.ARMED
    clock.t += 10
    _play(eng, clock, [None])
    assert eng.state == State.IDLE
    _play(eng, clock, gesture("vague", seed=71))
    assert events == []


@pytest.mark.parametrize("kind", ["vague", "poing", "pouce"])
@pytest.mark.parametrize("lead", [4, 5, 6, 7])
def test_realistic_stream_with_default_settings(gesture, kind, lead):
    """Geste noyé dans du repos, à différents décalages par rapport à la fenêtre."""
    lib = GestureLibrary()
    for name in ("poing", "vague", "pouce", "repos"):
        lib.add(name, [gesture(name, frames=22, seed=s) for s in range(5)])
    clock, events = FakeClock(), []
    eng = GestureEngine(lib, Config(), on_gesture=events.append, clock=clock)
    stream = (
        [None] * lead
        + list(gesture("repos", frames=13, seed=90))
        + list(gesture(kind, frames=22, seed=91))
        + list(gesture("repos", frames=30, seed=92))
    )
    _play(eng, clock, stream)
    # Exactement un événement, le bon ; « repos » n'est jamais émis.
    assert [e.name for e in events] == [kind]


def test_unknown_attention_gesture_falls_back_to_direct(gesture):
    eng, clock, events, _, _ = _setup(gesture, attention="inexistant")
    _play(eng, clock, gesture("poing", seed=80))
    assert [e.name for e in events] == ["poing"]

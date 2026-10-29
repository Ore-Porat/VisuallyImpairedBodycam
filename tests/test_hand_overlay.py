"""HandOverlayDetector: product boxes from one model, hand boxes from another."""
from slr.server.detector import Detection, HandOverlayDetector, ScriptedDetector


def test_hand_overlay_merges_only_hands():
    main = ScriptedDetector(frames=[[Detection.centered("oreo", 0.9, 0.5, 0.5, 0.3, 0.3),
                                     Detection.centered("hand", 0.9, 0.2, 0.2, 0.1, 0.1)]],
                            names=["oreo", "lays"])
    hands = ScriptedDetector(frames=[[Detection.centered("hand", 0.8, 0.7, 0.7, 0.1, 0.1),
                                      Detection.centered("soda", 0.9, 0.5, 0.5, 0.3, 0.3)]],
                             names=["soda", "hand"])
    det = HandOverlayDetector(main, hands)
    assert det.names == ["oreo", "lays", "hand"]
    got = det.detect(None)
    assert [(d.cls, d.conf) for d in got] == [("oreo", 0.9), ("hand", 0.8)]


def test_ignore_classes_drops_them():
    from slr.server.detector import IgnoreClassesDetector
    inner = ScriptedDetector(frames=[[Detection.centered("cheetos", 0.9, 0.5, 0.5, 0.3, 0.3),
                                      Detection.centered("oreo", 0.8, 0.2, 0.2, 0.1, 0.1)]],
                             names=["oreo", "cheetos"])
    det = IgnoreClassesDetector(inner, ["cheetos"])
    assert det.names == ["oreo"]
    assert [d.cls for d in det.detect(None)] == ["oreo"]

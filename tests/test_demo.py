from safesight.inference import DemoBackend
from safesight.logic import RuleEngine, Zone
from safesight.video import SyntheticVideo


def test_demo_demonstrates_three_rules_and_clear_without_hardware():
    video,backend,engine = SyntheticVideo(),DemoBackend(),RuleEngine()
    zone = Zone('danger','Danger',((.52,.35),(.96,.35),(.96,.94),(.52,.94)))
    events = []
    for n in range(0,900,15):  # The demo worker samples at 2 Hz, preview at 30.
        video.frame = n
        ok,image = video.read()
        assert ok
        _,emitted = engine.process('demo',backend.infer(image),[zone],960,540,observed_at=n/30)
        events.extend(emitted)
    kinds = [event.violation_type for event in events]
    assert kinds.count('restricted_zone') == 1
    assert kinds.count('no_helmet') == 1
    assert kinds.count('fall_detected') == 1
    assert engine.active_violations == []

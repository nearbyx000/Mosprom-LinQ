import unittest
from safesight.logic import Detection, Zone, RuleEngine, inside_polygon, iou


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.zone = Zone("z1", "danger", ((.5, .2), (.95, .2), (.95, .95), (.5, .95)))

    @staticmethod
    def person(x=60):
        return Detection("person", .9, (x, 40, x + 20, 90))

    def test_polygon_and_iou(self):
        self.assertTrue(inside_polygon(.7, .5, self.zone.points))
        self.assertFalse(inside_polygon(.2, .5, self.zone.points))
        self.assertEqual(iou((0, 0, 5, 5), (5, 5, 10, 10)), 0)

    def test_one_incident_one_event(self):
        e = RuleEngine(confirm_frames=3, clear_after=5)
        alerts = []
        for _ in range(30):
            _, new = e.process("camera", [self.person()], [self.zone], 100, 100)
            alerts.extend(new)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].violation_type, "restricted_zone")

    def test_two_people_two_alerts(self):
        e = RuleEngine(confirm_frames=2)
        # Create second person with non-overlapping boxes, both feet inside zone.
        people = [Detection("person", .9, (51, 40, 64, 90)), Detection("person", .9, (79, 40, 91, 90))]
        alerts = []
        for _ in range(4):
            _, new = e.process("camera", [Detection(p.label, p.confidence, p.box) for p in people], [self.zone], 100, 100)
            alerts.extend(new)
        self.assertEqual(len(alerts), 2)

    def test_leave_then_new_incident(self):
        e = RuleEngine(confirm_frames=2, clear_after=3)
        alerts = []
        for _ in range(4):
            _, new = e.process("camera", [self.person()], [self.zone], 100, 100)
            alerts.extend(new)
        for _ in range(4):
            e.process("camera", [self.person(10)], [self.zone], 100, 100)
        for _ in range(4):
            _, new = e.process("camera", [self.person()], [self.zone], 100, 100)
            alerts.extend(new)
        self.assertEqual(len(alerts), 2)

    def test_no_helmet_requires_explicit_negative_class(self):
        e = RuleEngine(confirm_frames=2)
        all_alerts = []
        for _ in range(3):
            _, new = e.process("camera", [Detection("helmet", .95, (40, 5, 60, 20))], [self.zone], 100, 100)
            all_alerts.extend(new)
        self.assertEqual(all_alerts, [])
        for _ in range(3):
            _, new = e.process("camera", [Detection("person", .95, (30, 0, 70, 90)),
                                         Detection("no_helmet", .95, (40, 5, 60, 20))], [], 100, 100)
            all_alerts.extend(new)
        self.assertEqual(len(all_alerts), 1)

    def test_filtered_by_confidence(self):
        e = RuleEngine(confirm_frames=1)
        _, alerts = e.process("c", [Detection("person", .20, (60, 40, 80, 90))], [self.zone], 100, 100)
        self.assertEqual(alerts, [])


if __name__ == "__main__":
    unittest.main()

# Extracted official requirements — LinQ case (slides 4, 13, 15, 18, 19)

## Mandatory (slide 13)

- Process camera video or a set of images / emulated still frames.
- Recognize safety incidents with a neural network.
- Deliver operational alerts in web UI or local GUI containing image/video, camera, time, type.
- Suppress repeat notifications for the **same** incident.
- Log/filter by violation type, camera, zone, time.
- Run inference on LinQ accelerators and measure processing speed.
- Architect new behavioral patterns as addable rules.

## Optional (slide 13)

- Per-camera rules/zones, including different conditions by time.
- Aggregated analytics / dashboard; user acknowledgements.
- Multi-stream operation.

## Scenarios offered (slide 4)

1. Running in the workshop / sudden movement in operating equipment areas.
2. Dangerous proximity to moving machinery/vehicles.
3. Phone usage while operating machinery / movement.
4. Smoking/open flames outside designated areas.
5. Crossing barriers, going under conveyors / lifting loads.
6. Work at height without safety harness.
7. Falling motion / lying motionless.
8. Handling loads unsafely (additional scenario).

**Note:** Helmet violation is not separately enumerated among these 8 examples. It can be additional, but not a reason to exclude explicit required features.

## Milestones

- First checkpoint: 23 Oct 18:00 Moscow time.
- Second: 24 Oct 11:00 Moscow time.
- Third: 24 Oct 18:00 Moscow time.
- Upload deadline: 25 Oct 10:00 Moscow time.
- Defenses start: 25 Oct 12:00 Moscow time.
- Two missed checkpoints lead to disqualification.

## MVP coverage today

| Requirement | Current status |
|---|---|
| Video/images input | DEMO synthetic + OpenCV file/RTSP reading |
| Safety incident via neural network | NOT YET; optional local Ultralytics adapter (user-provided model required) |
| LinQ TPU inference | NOT YET; actual SDK examples needed |
| Alert with evidence and metadata | Implemented in code, local test |
| Repeat suppression | Implemented + unit tested for stable tracks |
| SQLite log and filters | Implemented + unit tested |
| Web interface | Implemented |
| Benchmarks | Backend call / FPS; **no real TPU measurements** |
| Extendible rules | Dedicated RuleEngine + detection boundary |

## Accuracy and reliability limitations

- Demo mode is synthetic color-threshold logic, **not** a neural network.
- Helmet detection only works with models producing an **explicit** positive `no_helmet` detection class. Missing helmet bounding boxes alone are insufficient.
- Lightweight IoU tracking can switch IDs under overlap and occlusion; this means exactly-one-event deduplication is not guaranteed under all real-world scenes.
- Evidence snapshot shows processed frame at decision time, not a pre-event clip.
- Source-restart and server-restart can create a new incident ID; deduplication is in-memory.
- No operator acknowledgment; multi-camera TPU is not enabled.
- The MVP cannot be represented as ready for industrial safety-critical deployment.

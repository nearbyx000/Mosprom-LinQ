# Remote LinQ JPEG gateways

The Web application sends JPEG images and receives compact JSON. Neural inference
in `linq` mode occurs only in the organizer's remote TPU worker. These sources
must run near that worker, not in the browser or on an assumed local TPU.

| Gateway | Existing worker program | Input/output | Decoder |
|---|---|---|---|
| `yolov8n` | COCO YOLOv8n 640 | `[1,3,640,640]` / `[1,84,8400]` | person only |
| `safetyvision` | SafetyVision v2 640 | `[1,3,640,640]` / `[1,17,8400]` | 13 verified class indices; SV person excluded |

Install `pika`, `numpy`, `Pillow` and the organizer-provided
`tpu_pycloud_format` in the cloud Python environment. SafetyVision additionally
requires `opencv-python-headless` for preprocessing aligned with calibration.
An existing working primary gateway need not be restarted or redeployed.

From this repository, in the cloud environment:

```bash
python -m gateways.yolov8n.gateway --program /auto/cache/YOUR_LOGIN/yolov8n.tpu --timeout 12
python -m gateways.safetyvision.gateway --program /auto/cache/YOUR_LOGIN/safetyvision_v2_640.tpu --timeout 12
```

Each process prints a unique `GATEWAY READY` route. Set those distinct routes as
`LINQ_JPEG_ROUTE` and `LINQ_SV_ROUTE` in Web. Programs are required explicitly;
the gateways never upload, compile, overwrite, or replace `.tpu` files.

For a flat upload, copy the selected gateway's `gateway.py` and
`preprocessing.py`; SafetyVision also needs `decoder.py`. Then run
`python gateway.py --program ...`. Keep the two upload directories separate.

The primary preprocessing is the original Pillow path. SafetyVision uses
OpenCV INTER_LINEAR. Do not silently swap primary resizing or model shapes.
The historical `SafeSight_*` local delivery folders are no longer canonical
source and are ignored by Git; their working copies remain intact locally.

Export a new SafetyVision ONNX candidate only in a separate ML environment:

```bash
python -m gateways.safetyvision.export --weights models/v2/best.pt --output models/safetyvision_candidate_op11.onnx --skip-download
```

Existing output files are refused. Export also avoids overwriting an ONNX next
to the source weights. Compilation/evaluation commands are in
[`docs/SAFETYVISION_EVALUATION.md`](../docs/SAFETYVISION_EVALUATION.md).

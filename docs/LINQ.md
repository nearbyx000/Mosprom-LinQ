# LinQ TPU: integration gate

The organizers supplied three key facts (not an SDK implementation):

1. Native route: ONNX -> LinQ SDK quantization/compilation -> TPU Program.
2. Official Python TPU API is used to transmit input tensors and retrieve output tensors.
3. Remote notebook is at http://hackathon.lin-q.ru/ **after** VPN login; the provided notebook includes a ResNet50 example.

SDK download: https://cloud.lin-q.ru/s/bEqr922zr6NXHy4
Python API example: https://cloud.lin-q.ru/s/QcFLdkdLb71BOmm

## Integration procedure

- Connect VPN; avoid committing `.ovpn`, credentials, notebooks with tokens or secret keys.
- Open Jupyter. Run the **provided ResNet50 example** unchanged, with your assigned login.
- Acquire `tpu_framework.pdf` and `usage_example_python_api.zip` via the official organizer links.
- Select an ONNX detection model. Verify supported ops and input tensor shape against the SDK documentation before converting.
- Compile with LinQ SDK. Save the resulting TPU program locally or in the authorized Jupyter directory, per official instructions.
- Examine the official API example to determine the concrete client class, initialization, authentication, input tensor shape/type, and output tensor format.
- Implement `LinQBackend.infer()` in `safesight/inference.py`, returning `list[Detection]` of pixel-space `xyxy` coords, labels and confidences. Include any necessary preprocessing and postprocessing, based on *actual* sample output.
- Compare the TPU outputs to a local reference on 10+ identical frames and measure backend call latency (and separately pre/postprocessing and alert latency).
- After a verified run, update the UI verification flag with measured test evidence; do not set it merely because the program compiles.

`SAFESIGHT_MODE=linq` currently intentionally **fails at startup**. There is no CPU fallback. This is a guard against falsely claiming that real inference is performed on TPU.

## Checklist to ask LinQ experts

- SDK-supported ONNX ops, static input shapes, precision/quantization calibration corpus.
- Supported YOLO version/head and availability of already compiled COCO/person detector.
- Does example Python API run inside Jupyter only or from any VPN-connected workstation?
- Whether remote service accepts multiple concurrent requests and how TPU programs are assigned per team.
- How throughput / latency must be measured for final assessment.

# Third-party components and model artifacts

The project keeps its existing [GNU GPL v3 license](LICENSE). This file is a
component inventory, not a substitute for upstream licenses or a legal opinion.

| Component | Source / delivery | Status in this source tree |
|---|---|---|
| YOLOv8n / Ultralytics | [Ultralytics](https://github.com/ultralytics/ultralytics) | Optional local weights, not redistributed here |
| SafetyVision v2 | [Author model card](https://huggingface.co/ayushgupta7777/safetyvision-yolov8) (AGPL-3.0 stated there) | Optional local weights, not redistributed here |
| LinQ compiler, TPU runtime, `tpu_pycloud_format` | Organizer-provided SDK | Must be supplied by the authorized environment; vendor ZIP removed from current Git index, local copy preserved |
| Hard-hat candidate `keremberke/yolov8n-hard-hat-detection` | [Author repository](https://huggingface.co/keremberke/yolov8n-hard-hat-detection) | Exploratory local comparison only; no upstream weight license found in inspected metadata, not selected or redistributed |
| Construction-PPE | [Dataset author documentation](https://docs.ultralytics.com/datasets/detect/construction-ppe) | Evaluation images kept under ignored `data/`, not bundled |
| Fall Detection Dataset | [Dataset source](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset) | Evaluation images kept under ignored `data/`, not bundled |

Python package names and bounds are recorded in the requirements files. A
tested clean-environment snapshot accompanies the handover. Model metadata,
shape contracts and SHA256 are included in evaluation reports; a local `.tpu`
hash is not proof of the program currently loaded on a remote worker.

Before distributing weights, SDKs, datasets, or offering a commercial deployment,
verify the relevant upstream terms and the intended distribution model. The
SafeSight source license does not relabel those separate artifacts. No license
change or rights-clearance claim is made by this repository cleanup.

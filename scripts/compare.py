from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

for path in ["t.1.jpg", "t2.jpg", "t3.jpg", "t4.jpg"]:
    if not Path(path).exists():
        continue

    result = model.predict(
        path,
        imgsz=640,
        classes=[0],  # Person в COCO
        conf=0.25,
        verbose=False
    )[0]

    output = f"{Path(path).stem}_person.jpg"
    result.save(filename=output)

    print(f"{path}: {len(result.boxes)} people")
    for box in result.boxes:
        print(f"  confidence: {float(box.conf[0]):.3f}")
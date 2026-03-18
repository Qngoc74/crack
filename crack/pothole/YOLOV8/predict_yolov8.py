from ultralytics import YOLO

# Load weight đã train
model = YOLO("/Users/vyna/Desktop/YOLO/result_yolov8n/runs_pothole2/weights/best.pt")

# Predict 1 ảnh
results = model.predict(
    source="/Users/vyna/Desktop/YOLO/United_States_004780.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)

# Lấy kết quả của ảnh đầu tiên
r = results[0]

# Tên class
names = model.names

# Duyệt từng box
for i, box in enumerate(r.boxes):
    cls_id = int(box.cls[0].item())          # id class
    conf = float(box.conf[0].item())         # độ tin cậy
    xyxy = box.xyxy[0].tolist()              # [x1, y1, x2, y2]

    x1, y1, x2, y2 = xyxy
    class_name = names[cls_id]

    print(f"Object {i+1}:")
    print(f"  Class ID   : {cls_id}")
    print(f"  Class name : {class_name}")
    print(f"  Confidence : {conf:.4f}")
    print(f"  Box xyxy   : x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
    print("-" * 40)

print("Done")
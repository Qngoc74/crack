import torch
from ultralytics import YOLO

def main():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Current device:", torch.cuda.current_device())
        print("Device count:", torch.cuda.device_count())
    else:
        print("PyTorch chưa nhận GPU")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="C:/Users/ezycloudx-admin/Desktop/YOLO/Data/pothole_detection.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
    
        project="C:/Users/ezycloudx-admin/Desktop/YOLO/Data/result_yolov8n",
        name="runs_pothole"
    )

    if torch.cuda.is_available():
        print("Max GPU memory allocated (MB):", torch.cuda.max_memory_allocated(0) / 1024**2)
        print("Max GPU memory reserved  (MB):", torch.cuda.max_memory_reserved(0) / 1024**2)

if __name__ == "__main__":
    main()
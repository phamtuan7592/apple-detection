import cv2
from ultralytics import YOLO
import numpy as np
#chương 5: sử dụng yolo để nhận diện ảnh
class ObjectDetector:
    def __init__(self, model_path="best.pt", min_score_thresh=0.35):
        """
        Khởi tạo detector với YOLO11
        
        Args:
            model_path: Đường dẫn đến file model .pt
            min_score_thresh: Ngưỡng confidence
        """
        self.min_score_thresh = min_score_thresh
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Lấy class names từ model
        self.classes = self.model.names
        
    def detect(self, frame, conf=0.35, imgsz=640, classes=None):
        """
        Phát hiện đối tượng trong frame
        
        Args:
            frame: ảnh đầu vào
            conf: ngưỡng confidence
            imgsz: kích thước ảnh đầu vào
            classes: list các class ID cần lọc (VD: [0,2,3] hoặc None để lấy tất cả)
            
        Returns:
            boxes: list of [x, y, width, height]
            class_ids: list of class IDs
            scores: list of confidence scores
        """
        if conf is None:
            conf = self.min_score_thresh
            
        # Chạy YOLO detection với filter classes
        
        results = self.model(frame, conf=conf, imgsz=imgsz, classes=classes, verbose=False)
        
        boxes = []
        class_ids = []
        scores = []
        
        # Lấy kết quả từ frame đầu tiên
        if len(results) > 0:
            result = results[0]
            
            # Lấy bounding boxes
            if result.boxes is not None:
                for box in result.boxes:
                    # Lấy tọa độ (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Chuyển sang [x, y, w, h]
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    # Lấy class id và confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    scores.append(confidence)
        
        return boxes, class_ids, scores
    
    def draw_boxes(self, frame, boxes, class_ids, scores, line_thickness=2):
        """
        Vẽ bounding boxes lên frame
        """
        for box, class_id, score in zip(boxes, class_ids, scores):
            x, y, w, h = box
            
            # Màu sắc theo class
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 0),    # Dark Blue
                (0, 128, 0)     # Dark Green
            ]
            color = colors[class_id % len(colors)]
            
            # Vẽ khung
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, line_thickness)
            
            # Tạo label
            class_name = self.classes.get(class_id, f'Class_{class_id}')
            label = f'{class_name}: {score:.2f}'
            
            # Vẽ nền cho label
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_size[1] - baseline), 
                         (x + label_size[0], y), color, cv2.FILLED)
            
            # Vẽ text
            cv2.putText(frame, label, (x, y - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def close(self):
        """Đóng detector"""
        pass
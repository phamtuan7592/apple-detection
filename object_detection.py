import cv2
from ultralytics import YOLO
import numpy as np

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
        
    def detect(self, frame, conf=0.35, imgsz=640, classes=None, track=False, persist=False):
        """
        Phát hiện đối tượng trong frame (hỗ trợ tracking)
        
        Args:
            frame: ảnh đầu vào
            conf: ngưỡng confidence
            imgsz: kích thước ảnh đầu vào
            classes: list các class ID cần lọc (VD: [0,2,3] hoặc None để lấy tất cả)
            track: Bật tracking (trả về track_ids)
            persist: Giữ tracking qua các frame
            
        Returns:
            Nếu track=False: boxes, class_ids, scores
            Nếu track=True: boxes, class_ids, scores, track_ids
        """
        if conf is None:
            conf = self.min_score_thresh
            
        # Chạy YOLO detection với tracking nếu được yêu cầu
        if track:
            # Sử dụng tracker của YOLO
            results = self.model.track(
                frame, 
                conf=conf, 
                imgsz=imgsz, 
                classes=classes, 
                persist=persist,
                tracker="bytetrack.yaml",  # Hoặc "botsort.yaml"
                verbose=False
            )
        else:
            results = self.model(frame, conf=conf, imgsz=imgsz, classes=classes, verbose=False)
        
        boxes = []
        class_ids = []
        scores = []
        track_ids = []
        
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
                    
                    # Lấy track ID nếu có
                    if track and hasattr(box, 'id'):
                        track_id = int(box.id[0].tolist())
                        track_ids.append(track_id)
                    elif track:
                        # Nếu không có track_id, tạo temporary
                        track_ids.append(-1)
        
        if track:
            return boxes, class_ids, scores, track_ids
        else:
            return boxes, class_ids, scores
    
    def draw_boxes(self, frame, boxes, class_ids, scores, track_ids=None, line_thickness=2):
        """
        Vẽ bounding boxes lên frame (có thể hiển thị cả track_id)
        """
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
        
        for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
            x, y, w, h = box
            
            color = colors[class_id % len(colors)]
            
            # Vẽ khung
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, line_thickness)
            
            # Tạo label
            class_name = self.classes.get(class_id, f'Class_{class_id}')
            
            if track_ids is not None and i < len(track_ids):
                label = f'#{track_ids[i]} {class_name}: {score:.2f}'
            else:
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
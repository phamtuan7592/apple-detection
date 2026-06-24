import cv2
import numpy as np
from object_detection import ObjectDetector
import sys
import os
import json

# === CẤU HÌNH KÍCH THƯỚC HIỂN THỊ ===
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

print("Loading YOLO11 model...")
od = ObjectDetector(model_path="best.pt", min_score_thresh=0.35)
print("Model loaded successfully!")

# --- THIẾT LẬP BACKGROUND SUBTRACTOR (Chạy ngầm để lọc dữ liệu) ---
print("Khởi tạo thuật toán Background Subtractor MOG2...")
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

# === HỆ THỐNG TRACKING VÀ ĐẾM ===
next_object_id = 0
tracked_objects = {}  
counted_objects = set()
total_unique_count = 0

def get_object_center(box):
    """Lấy tâm của bounding box"""
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def calculate_distance(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_inside_roi(box, roi):
    """Kiểm tra tâm của bounding box có nằm trong ROI không"""
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0

def update_tracking_and_count(bboxes_display, roi_display, frame_count, max_distance=50):
    """Cập nhật tracking và đếm các đối tượng DUY NHẤT"""
    global next_object_id, total_unique_count, tracked_objects, counted_objects
    
    current_detections = []
    
    for box in bboxes_display:
        center = get_object_center(box)
        inside_roi = cv2.pointPolygonTest(roi_display, center, False) >= 0
        current_detections.append({
            'box': box,
            'center': center,
            'inside_roi': inside_roi
        })
    
    matched_current_ids = set()
    
    for detection in current_detections:
        center = detection['center']
        inside_roi = detection['inside_roi']
        
        matched_id = None
        min_dist = float('inf')
        
        for obj_id, obj_data in tracked_objects.items():
            dist = calculate_distance(center, obj_data['last_position'])
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                matched_id = obj_id
        
        if matched_id is not None:
            matched_current_ids.add(matched_id)
            tracked_objects[matched_id]['last_position'] = center
            
            if inside_roi and not tracked_objects[matched_id]['counted']:
                total_unique_count += 1
                tracked_objects[matched_id]['counted'] = True
                counted_objects.add(matched_id)
        else:
            tracked_objects[next_object_id] = {
                'last_position': center,
                'counted': False,
                'first_seen_frame': frame_count
            }
            
            if inside_roi:
                total_unique_count += 1
                tracked_objects[next_object_id]['counted'] = True
                counted_objects.add(next_object_id)
            
            matched_current_ids.add(next_object_id)
            next_object_id += 1
    
    objects_to_remove = []
    for obj_id in list(tracked_objects.keys()):
        if obj_id not in matched_current_ids and tracked_objects[obj_id]['counted']:
            objects_to_remove.append(obj_id)
    
    for obj_id in objects_to_remove:
        del tracked_objects[obj_id]
    
    return total_unique_count

# --- MENU CHỌN NGUỒN ---
print("\n" + "="*40)
print("HỆ THỐNG GIÁM SÁT THÔNG MINH")
print("="*40)
print("1. Sử dụng Webcam (Real-time)")
print("2. Sử dụng File Video")
print("3. Thoát")
print("-"*40)

choice = input("Nhập lựa chọn của bạn (1, 2 hoặc 3): ")

if choice == '1':
    cap = cv2.VideoCapture(0)
    source_label = "Webcam"
elif choice == '2':
    video_path = "demo1.mkv"
    cap = cv2.VideoCapture(video_path)
    source_label = f"Video: {video_path}"
else:
    sys.exit()

if not cap.isOpened():
    print(f"\n[LỖI] Không thể mở được {source_label}!")
    sys.exit()

# --- TẢI VÙNG ROI ---
if os.path.exists("roi.json"):
    with open("roi.json", "r") as f:
        data = json.load(f)
        ROI_original = np.array(data["ROI"], dtype=np.int32)
    print("\n[INFO] Đã tải tọa độ ROI từ roi.json")
else:
    ROI_original = np.array([[0, 278], [1277, 145], [1266, 718], [6, 713], [1, 280]], dtype=np.int32)
    print("\n[INFO] Không tìm thấy roi.json, dùng ROI mặc định")

# Lấy kích thước gốc và tính tỉ lệ scale
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Kích thước gốc video: {original_width}x{original_height}")

scale_x = DESIRED_WIDTH / original_width
scale_y = DESIRED_HEIGHT / original_height

# CHỈ KHỞI TẠO DUY NHẤT CỬA SỔ CHƯƠNG 5 TỔNG HỢP
cv2.namedWindow("Nhan dang anh & Tong hop", cv2.WINDOW_NORMAL)

print("\n" + "="*50)
print("HƯỚNG DẪN BÀN PHÍM:")
print("- ESC: Thoát chương trình")
print("- SPACE: Tạm dừng / Tiếp tục")
# print("- r: Reset tổng số lượng đếm về 0")
print("="*50)

frame_count = 0
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Hết video hoặc mất kết nối camera.")
            break
        
        frame_count += 1
        
        # 1. Chuẩn bị khung hình hiển thị và scale vùng ROI
        frame_display = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
        ROI_display = (ROI_original * [scale_x, scale_y]).astype(np.int32)
        
        # 2. XỬ LÝ MOG2 (Chạy ngầm tạo mặt nạ chuyển động)
        blurred_frame = cv2.GaussianBlur(frame_display, (5, 5), 0)
        fg_mask = backSub.apply(blurred_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  
        
        # 3. XỬ LÝ CANNY (Chạy ngầm phát hiện biên cạnh)
        edges = cv2.Canny(fg_mask, 50, 150)
        
        # 4. NHẬN DẠNG VỚI YOLO11
        bboxes, class_ids, scores = od.detect(frame, conf=0.35)
        
        validated_bboxes = []
        validated_ids = []
        validated_scores = []
        
        # 5. BỘ LỌC TỔNG HỢP (KẾT HỢP NGẦM): Loại bỏ đối tượng tĩnh / nhiễu
        for box, cid, score in zip(bboxes, class_ids, scores):
            x_disp = int(box[0] * scale_x)
            y_disp = int(box[1] * scale_y)
            w_disp = int(box[2] * scale_x)
            h_disp = int(box[3] * scale_y)
            
            x_disp = max(0, x_disp)
            y_disp = max(0, y_disp)
            w_disp = min(w_disp, DESIRED_WIDTH - x_disp)
            h_disp = min(h_disp, DESIRED_HEIGHT - y_disp)
            
            if w_disp <= 0 or h_disp <= 0:
                continue
                
            # Trích xuất vùng cục bộ để tính mật độ pixel
            roi_fg = fg_mask[y_disp:y_disp+h_disp, x_disp:x_disp+w_disp]
            roi_edges = edges[y_disp:y_disp+h_disp, x_disp:x_disp+w_disp]
            
            total_pixels = w_disp * h_disp
            motion_ratio = np.sum(roi_fg == 255) / total_pixels
            edge_ratio = np.sum(roi_edges == 255) / total_pixels
            
            # Điều kiện lọc chất lượng đầu vào hệ thống tracking
            if motion_ratio > 0.05 and edge_ratio > 0.01:
                validated_bboxes.append([x_disp, y_disp, w_disp, h_disp])
                validated_ids.append(cid)
                validated_scores.append(score)
        
        # 6. CẬP NHẬT TRACKING VÀ ĐẾM
        total_unique = update_tracking_and_count(validated_bboxes, ROI_display, frame_count)
        
        # Lọc riêng các đối tượng trong ROI để chuẩn bị vẽ box
        current_in_roi = 0
        filtered_boxes = []
        filtered_ids = []
        filtered_scores = []
        
        for box_disp, cid, score in zip(validated_bboxes, validated_ids, validated_scores):
            if is_inside_roi(box_disp, ROI_display):
                current_in_roi += 1
                filtered_boxes.append(box_disp)
                filtered_ids.append(cid)
                filtered_scores.append(score)
        
        # 7. ĐỒ HỌA VÀ HIỂN THỊ KẾT QUẢ TỔNG HỢP DUY NHẤT
        frame_result = frame_display.copy()
        frame_result = od.draw_boxes(frame_result, filtered_boxes, filtered_ids, filtered_scores, line_thickness=2)
        
        cv2.polylines(frame_result, [ROI_display], True, (0, 255, 255), 2)
        
        cv2.putText(frame_result, f'Objects in ROI now: {current_in_roi}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame_result, f'TOTAL COUNT: {total_unique}', (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Chỉ gọi duy nhất 1 cửa sổ hiển thị tổng hợp
        cv2.imshow("Nhan dang anh & Tong hop", frame_result)
    
    # --- XỬ LÝ SỰ KIỆN PHÍM BẤM ---
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # Spacebar
        paused = not paused
        print("[PAUSE]" if paused else "[RESUME]")
    # elif key == ord('r'):  # Reset
    #     total_unique_count = 0
    #     counted_objects.clear()
    #     tracked_objects.clear()
    #     next_object_id = 0
# --- KẾT THÚC ---
print("KẾT QUẢ GIÁM SÁT SAU CÙNG:\n")
print(f"Tổng số đối tượng DUY NHẤT đã đi qua ROI: {total_unique_count}")
print(f"Tổng số frame hình đã được xử lý     : {frame_count}")
print("="*50)

cap.release()
cv2.destroyAllWindows()
od.close()
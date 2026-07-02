import cv2
import numpy as np
from collections import deque
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

# === HỆ THỐNG ĐẾM VÀ PHÂN LOẠI ===
counted_ids = set()          
total_unique_count = 0

# [THÊM MỚI] Các biến lưu trữ cho hệ thống Sorter
detailed_stats = {}          # Lưu thống kê: {'Táo': {'S (Nho)': 0, 'M (Vua)': 0, 'L (Lon)': 0}}
trajectories = {}            # Lưu quỹ đạo: {track_id: deque(maxlen=30)}
fruit_sizes = {}             # Lưu size cố định của từng quả: {track_id: "L (Lon)"}

def is_inside_roi(box, roi):
    """Kiểm tra tâm của bounding box có nằm trong ROI không"""
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0

# [THÊM MỚI] Hàm đo kích thước
def determine_size(w, h):
    area = w * h
    # Bạn có thể tinh chỉnh các con số này cho phù hợp với camera thực tế
    if area > 15000: return "L (Lon)"
    elif area > 8000: return "M (Vua)"
    else: return "S (Nho)"

# --- MENU CHỌN NGUỒN ---
print("\n" + "="*40)
print("HỆ THỐNG GIÁM SÁT VÀ PHÂN LOẠI THÔNG MINH")
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
        
        # 4. NHẬN DẠNG VỚI YOLO (Lấy cả track_ids)
        bboxes, class_ids, scores, track_ids = od.detect(frame, conf=0.35, track=True, persist=True)

        
        validated_bboxes = []
        validated_ids = []
        validated_scores = []
        validated_tids = []
        
        # 5. BỘ LỌC TỔNG HỢP (KẾT HỢP NGẦM): Loại bỏ đối tượng tĩnh / nhiễu
        for box, cid, score, tid in zip(bboxes, class_ids, scores, track_ids):
            # Bỏ qua đối tượng chưa có ID ổn định
            if tid == -1:
                continue
                
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
                validated_tids.append(tid)
        
        # ==========================================================
        # 6. LOGIC CẬP NHẬT SORTER (QUỸ ĐẠO, SIZE & ĐẾM PHÂN LOẠI)
        # ==========================================================
        current_in_roi = 0
        filtered_boxes, filtered_ids, filtered_scores, filtered_tids = [], [], [], []
        
        for box_disp, cid, score, tid in zip(validated_bboxes, validated_ids, validated_scores, validated_tids):
            
            # 6.1 Cập nhật Quỹ đạo (Đuôi đỏ)
            cx, cy = box_disp[0] + box_disp[2]//2, box_disp[1] + box_disp[3]//2
            if tid not in trajectories:
                trajectories[tid] = deque(maxlen=30)
            trajectories[tid].append((cx, cy))
            
            # 6.2 Đo kích thước (Chỉ đo 1 lần cho mỗi trái cây)
            if tid not in fruit_sizes:
                fruit_sizes[tid] = determine_size(box_disp[2], box_disp[3])
                
            # 6.3 Logic Đếm & Ghi nhận vào ROI
            if is_inside_roi(box_disp, ROI_display):
                current_in_roi += 1
                filtered_boxes.append(box_disp)
                filtered_ids.append(cid)
                filtered_scores.append(score)
                filtered_tids.append(tid) # Gửi Track ID đi để vẽ lên màn hình
                
                # Đếm đối tượng nếu chưa được đếm
                if tid not in counted_ids:
                    counted_ids.add(tid)
                    total_unique_count += 1
                    
                    # Cập nhật Thống kê chi tiết (Sorter logic)
                    c_name = od.classes.get(cid, "Unknown")
                    f_size = fruit_sizes[tid]
                    if c_name not in detailed_stats:
                        detailed_stats[c_name] = {'S (Nho)': 0, 'M (Vua)': 0, 'L (Lon)': 0}
                    detailed_stats[c_name][f_size] += 1
        
        # ==========================================================
        # 7. ĐỒ HỌA GIAO DIỆN CÔNG NGHIỆP (DASHBOARD)
        # ==========================================================
        frame_result = frame_display.copy()
        
        # 7.1 Vẽ quỹ đạo (Đuôi bám theo trái cây)
        for tid in validated_tids: # Chỉ vẽ các quả đang lọt qua lớp lọc MOG2
            if tid in trajectories:
                pts = list(trajectories[tid])
                for i in range(1, len(pts)):
                    cv2.line(frame_result, pts[i-1], pts[i], (0, 0, 255), max(1, int(3 - i/10)))

        # 7.2 Vẽ Bounding Boxes bằng hàm của ObjectDetector
        # [QUAN TRỌNG]: Truyền thêm filtered_tids để hàm cũ hiện #ID
        frame_result = od.draw_boxes(frame_result, filtered_boxes, filtered_ids, filtered_scores, track_ids=filtered_tids, line_thickness=2)
        
        # 7.3 Vẽ Size trực tiếp cạnh Bounding Box
        for box_disp, tid in zip(filtered_boxes, filtered_tids):
            x, y, w, h = box_disp
            size_label = fruit_sizes.get(tid, "Unknown")
            cv2.putText(frame_result, f"| Size: {size_label}", (x + w - 10, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 7.4 Vẽ ROI đa giác
        cv2.polylines(frame_result, [ROI_display], True, (0, 255, 255), 2)
        
        # 7.5 Vẽ Bảng Thống kê chi tiết
        cv2.rectangle(frame_result, (10, 130), (320, 360), (0, 0, 0), -1)
        cv2.putText(frame_result, "THONG KE CHI TIET", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        y_pos = 190
        for fruit, sizes in detailed_stats.items():
            cv2.putText(frame_result, f"{fruit}:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
            for size, count in sizes.items():
                cv2.putText(frame_result, f"  - Size {size}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_pos += 25

        # 7.6 Vẽ đếm tổng (Giữ nguyên của bạn)
        cv2.putText(frame_result, f'Objects in ROI now: {current_in_roi}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame_result, f'TOTAL COUNT: {total_unique_count}', (10, 85), 
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
    #     counted_ids.clear()
    #     # Không cần tracked_objects và next_object_id nữa

# --- KẾT THÚC ---
print("\n" + "="*50)
print("KẾT QUẢ GIÁM SÁT SAU CÙNG:\n")
print(f"Tổng số frame hình đã được xử lý     : {frame_count}")
print(f"Tổng số đối tượng DUY NHẤT đã đi qua ROI: {total_unique_count}")
print("-" * 40)
for fruit, sizes in detailed_stats.items():
    print(f"[{fruit.upper()}]")
    for size, count in sizes.items():
        print(f" - Size {size}: {count}")
print("="*50)

cap.release()
cv2.destroyAllWindows()
od.close()
import cv2
from ultralytics import YOLO
import subprocess
import os
# Đọc video từ file m3u8
video_path = "D:\\FPT\\FALL2024\\m3u8\\output.m3u8"  # Thay đường dẫn tới file m3u8
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Xử lý frame ở đây (ví dụ: đưa vào mô hình phát hiện đối tượng)
    
    # Hiển thị frame để kiểm tra
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

# Khởi tạo model YOLO
model = YOLO("yolov8n.pt")

# Đọc video từ file M3U8
cap = cv2.VideoCapture("D:\\FPT\\FALL2024\\m3u8\\output.m3u8")

if not cap.isOpened():
    print("Không thể mở file video M3U8!")
    exit()

# Lấy thông tin về độ rộng, chiều cao và FPS của khung hình video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 60:
    fps = 25

print(f"Width: {width}, Height: {height}, FPS: {fps}")

# Khởi tạo VideoWriter để ghi video đã qua phát hiện
output_video = cv2.VideoWriter('output_with_detection.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))

if not output_video.isOpened():
    print("Không thể khởi tạo VideoWriter!")
    cap.release()
    exit()

# Loop qua từng frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame hoặc video đã hết.")
        break
    
    # Chạy mô hình phát hiện đối tượng trên frame
    results = model(frame)
    
    # Vẽ bounding box lên frame
    annotated_frame = results[0].plot()
    
    # Ghi lại frame đã qua phát hiện vào file video
    output_video.write(annotated_frame)
    
    # Hiển thị frame đã phát hiện đối tượng (có thể tắt nếu không cần)
    cv2.imshow('Detected Frame', annotated_frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng các tài nguyên
cap.release()
output_video.release()
cv2.destroyAllWindows()
import subprocess

# Chuyển đổi file MP4 đã qua phát hiện thành M3U8
input_video = 'D:\\FPT\\FALL2024\\output_with_detection.mp4'
output_m3u8 = 'output_with_detection.m3u8'

command = [
    'ffmpeg',
    '-i', input_video,
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-f', 'hls',
    '-hls_time', '10',
    '-hls_playlist_type', 'event',
    '-hls_list_size', '0',  # Giữ tất cả các phân đoạn
    output_m3u8
]

result = subprocess.run(command, stderr=subprocess.PIPE, text=True)

# Kiểm tra xem file M3U8 có tồn tại sau khi chuyển đổi không
if result.returncode == 0 and os.path.exists(output_m3u8):
    print(f"Chuyển đổi thành công: {output_m3u8}")
else:
    print("Đã có lỗi xảy ra trong quá trình chuyển đổi.")
    print("Chi tiết lỗi:", result.stderr)
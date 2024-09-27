from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import os

# Load your custom YOLOv8 model
model = YOLO("D:\\FPT\\FALL2024\\yolov8_best.pt")  # Đường dẫn tới model của bạn

# Open the RTSP stream
rtsp_link = "rtsp://long:Xsw!12345@nongdanonline.ddns.net:554/cam/realmonitor?channel=2&subtype=0"  # Thay bằng link RTSP của bạn
cap = cv2.VideoCapture(rtsp_link)

# Initialize VideoWriter
output_video_path = 'output_with_tracking.mp4'
desired_width = 1500  # Độ rộng mong muốn
desired_height = 800  # Độ cao mong muốn
output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'H264'), 10, (desired_width, desired_height))

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Resize frame to the desired resolution
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Count the number of detected boxes
        num_boxes = len(boxes)

        # Write the count on the frame
        cv2.putText(frame, f'Number of objects: {num_boxes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Write the annotated frame to video
        output_video.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Convert the MP4 file with detections to M3U8
output_m3u8 = 'output_with_tracking.m3u8'
command = [
    'ffmpeg',
    '-i', output_video_path,
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-f', 'hls',
    '-hls_time', '60',
    '-hls_playlist_type', 'event',
    '-hls_list_size', '0',  # Keep all segments
    output_m3u8
]

result = subprocess.run(command, stderr=subprocess.PIPE, text=True)

# Check if the M3U8 file was created successfully
if result.returncode == 0 and os.path.exists(output_m3u8):
    print(f"Chuyển đổi thành công: {output_m3u8}")
else:
    print("Đã có lỗi xảy ra trong quá trình chuyển đổi.")
    print("Chi tiết lỗi:", result.stderr)

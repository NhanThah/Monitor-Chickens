import cv2
import torch
import pathlib
import subprocess
import os

# Adjusting pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load your YOLOv5 model
model_path = 'D:\\FPT\\FALL2024\\yolov5\\best.pt'
model = torch.hub.load('D:\\FPT\\FALL2024\\yolov5', 'custom', path=model_path, source='local')

# RTSP stream URL
rtsp_url = "rtsp://internsys:Them1kynuanhe@nongdanonlnine.ddns.net:554/cam/realmonitor?channel=2&subtype=0"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Get frame width and height from the capture
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Directory to save output
output_dir = 'D:\\FPT\\FALL2024\\hls_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Setup FFmpeg command to convert to HLS (m3u8)
ffmpeg_command = [
    'ffmpeg',
    '-y',  # overwrite output file if it exists
    '-f', 'rawvideo',  # input format
    '-vcodec', 'rawvideo',  # input codec
    '-pix_fmt', 'bgr24',  # pixel format
    '-s', f'{width}x{height}',  # frame size
    '-r', '15',  # frame rate
    '-i', '-',  # input comes from stdin
    '-c:v', 'libx264',  # video codec
    '-f', 'hls',  # format is HLS
    '-hls_time', '10',  # duration of each segment (in seconds)
    '-hls_list_size', '0',  # keep all segments
    '-hls_flags', 'delete_segments',  # delete old segments
    os.path.join(output_dir, 'output_with_detection.m3u8')  # output file with full path
]

# Start the FFmpeg process
process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# Set up resizable window for display
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', width, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference with YOLOv5
    results = model(frame)

    # Render the results on the frame
    annotated_frame = results.render()[0]

    # Write frame to FFmpeg for HLS conversion
    process.stdin.write(annotated_frame.tobytes())

    # Display the frame with detection
    cv2.imshow('Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
process.stdin.close()
process.wait()
cv2.destroyAllWindows()
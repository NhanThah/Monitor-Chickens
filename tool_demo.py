import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import os
import time
import matplotlib.pyplot as plt
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


class VideoApp:
    def __init__(self, root, rtsp_url):
        self.root = root
        self.root.title("Video Annotation Tool")

        # Khởi tạo biến
        self.rtsp_url = rtsp_url
        self.capture = cv2.VideoCapture(rtsp_url)
        self.running = True
        self.current_frame = None
        self.drawing_box = False
        self.start_x = self.start_y = 0
        self.boxes = []
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Các nút điều khiển
        self.btn_play = tk.Button(root, text="Dừng", command=self.toggle_play_pause)
        self.btn_play.pack(side=tk.LEFT)

        self.btn_skip = tk.Button(root, text="Tua 5s", command=self.skip_5_seconds)
        self.btn_skip.pack(side=tk.LEFT)

        self.btn_create_box = tk.Button(root, text="Tạo khung", command=self.create_bounding_box)
        self.btn_create_box.pack(side=tk.LEFT)

        self.btn_delete_box = tk.Button(root, text="Xóa khung", command=self.delete_bounding_box)
        self.btn_delete_box.pack(side=tk.LEFT)

        self.btn_save = tk.Button(root, text="Lưu ảnh", command=self.save_frame)
        self.btn_save.pack(side=tk.LEFT)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        # Vẽ khung và chạy video trong luồng riêng
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()

    def update_frame(self):
        if self.running:
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame.copy()  # Lưu khung hình hiện tại

                # Phát hiện đối tượng bằng YOLO
                results = model(frame)

                # Vẽ các bounding box lên khung hình
                for *xyxy, conf, cls in results.xyxy[0].tolist():
                    label = results.names[int(cls)]  # Lấy tên đối tượng (ví dụ: 'bird', 'cat')
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Vẽ bounding box nếu phát hiện đối tượng
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Khung màu xanh lá
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)

                # Chuyển đổi khung hình từ BGR sang RGB để hiển thị trên canvas
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                # Cập nhật ảnh trên canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk  # Giữ tham chiếu để không bị dọn dẹp
            self.root.after(30, self.update_frame)

    def toggle_play_pause(self):
        self.running = not self.running
        self.btn_play.config(text="Chạy" if not self.running else "Dừng")

    def start_drawing_box(self):
        self.drawing_box = True

    def on_mouse_down(self, event):
        if self.drawing_box:
            self.start_x = event.x
            self.start_y = event.y
            print(f"Start: ({self.start_x}, {self.start_y})")

    def on_mouse_drag(self, event):
        if self.drawing_box:
            self.canvas.delete("rect")
            self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", tags="rect")
            print(f"Dragging: ({event.x}, {event.y})")

    def on_mouse_up(self, event):
        if self.drawing_box:
            x1, y1, x2, y2 = self.start_x, self.start_y, event.x, event.y
            print(f"Box drawn: ({x1}, {y1}), ({x2}, {y2})")
            self.boxes.append((x1, y1, x2, y2))
            self.drawing_box = False
            self.update_canvas()

    def skip_5_seconds(self):
        if self.capture.isOpened():
            current_pos = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            self.capture.set(cv2.CAP_PROP_POS_MSEC, current_pos + 5000)

    def create_bounding_box(self):
        if self.current_frame is not None:
            # Sử dụng YOLO để phát hiện đối tượng
            results = model(self.current_frame)
            print(results)
            for *xyxy, conf, cls in results.xyxy[0].tolist():
                if int(cls) == 3:  # Lớp "bird"
                    x1, y1, x2, y2 = map(int, xyxy)
                    self.boxes.append((x1, y1, x2, y2))
            self.update_canvas()

    def delete_bounding_box(self):
        if self.boxes:
            self.boxes.pop()  # Xóa khung cuối cùng
            self.update_canvas()

    def update_canvas(self):
        if self.current_frame is not None:
            frame_copy = self.current_frame.copy()
            for (x1, y1, x2, y2) in self.boxes:
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

    def save_frame(self):
        if self.current_frame is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if save_path:
                cv2.imwrite(save_path, self.current_frame)
                print(f"Ảnh đã được lưu tại: {save_path}")
if __name__ == "__main__":
    root = tk.Tk()
    rtsp_url = "rtsp://internsys:Them1kynuanhe@nongdanonlnine.ddns.net:554/cam/realmonitor?channel=2^&subtype=0"  # Đặt RTSP URL của bạn ở đây
    app = VideoApp(root, rtsp_url)
    root.mainloop()                                         
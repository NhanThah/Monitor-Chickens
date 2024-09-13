import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess
import os
import threading
import glob
import time
class VideoStreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor Chicken")

        # Tạo frame cho video
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(pady=10)

        # Nhãn để hiển thị video
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Tạo frame cho ô nhập và nút
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        # Nhãn và ô nhập địa chỉ RTSP
        self.rtsp_label = tk.Label(self.control_frame, text="Nhập URL RTSP:")
        self.rtsp_label.pack(side=tk.LEFT)

        self.rtsp_entry = tk.Entry(self.control_frame, width=40)
        self.rtsp_entry.pack(side=tk.LEFT)

        # Nhãn và ô nhập thư mục đầu ra cho M3U8
        self.m3u8_label = tk.Label(self.control_frame, text="Nhập thư mục lưu M3U8:")
        self.m3u8_label.pack(side=tk.LEFT)

        self.m3u8_entry = tk.Entry(self.control_frame, width=40)
        self.m3u8_entry.pack(side=tk.LEFT)

        # Nút để bắt đầu phát video
        self.start_button = tk.Button(self.control_frame, text="Bắt đầu phát video", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT)

        # Nút để chuyển đổi RTSP thành M3U8
        self.convert_button = tk.Button(self.control_frame, text="Chuyển đổi RTSP sang M3U8", command=self.start_conversion)
        self.convert_button.pack(side=tk.LEFT)

        # Nút để dừng quá trình
        self.stop_button = tk.Button(self.control_frame, text="Dừng", command=self.stop_conversion, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        self.vid = None
        self.frame_count = 0  # Đếm số khung hình đã lưu
        self.stop_flag = False  # Biến để kiểm soát việc dừng luồng

    def start_stream(self):
        rtsp_url = self.rtsp_entry.get()
        if rtsp_url:
            self.vid = cv2.VideoCapture(rtsp_url)
            if not self.vid.isOpened():
                messagebox.showerror("Error", "Không thể mở video từ URL đã cho.")
            else:
                self.show_frame()
        else:
            messagebox.showwarning("Warning", "Vui lòng nhập URL RTSP.")

    def show_frame(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Lặp lại để tiếp tục hiển thị các khung hình mới
            self.video_label.after(10, self.show_frame)
        else:
            messagebox.showerror("Error", "Không thể đọc khung hình từ video.")
            
    def start_conversion(self):
        rtsp_url = self.rtsp_entry.get()
        output_dir = self.m3u8_entry.get()
        if rtsp_url and output_dir:
            self.stop_flag = False
            self.stop_button.config(state=tk.NORMAL)  # Kích hoạt nút Dừng
            threading.Thread(target=self.convert_rtsp_to_m3u8, args=(rtsp_url, output_dir)).start()
        else:
            messagebox.showwarning("Warning", "Vui lòng nhập URL RTSP và thư mục đầu ra M3U8.")

    def stop_conversion(self):
        # Đặt cờ dừng thành True
        self.stop_flag = True
        self.stop_button.config(state=tk.DISABLED)  # Vô hiệu hóa nút Dừng
        messagebox.showinfo("Info", "Đã yêu cầu dừng quá trình chuyển đổi.")

    def convert_rtsp_to_m3u8(self, rtsp_url, output_dir):
        output_file = os.path.join(output_dir, 'output.m3u8')
        command = [
            'C:\\ffmpeg\\ffmpeg-7.0.2-full_build\\bin\\ffmpeg.exe',  # Đường dẫn tới ffmpeg.exe
            '-i', rtsp_url,          # Đường dẫn RTSP đầu vào
            '-c:v', 'copy',          # Giữ nguyên video codec
            '-c:a', 'aac',           # Chuyển đổi âm thanh sang AAC
            '-f', 'hls',             # Định dạng đầu ra là HLS (M3U8)
            '-hls_time', '10',       # Thời gian mỗi phân đoạn .ts (10 giây)
            '-hls_playlist_type', 'event',# Playlist theo kiểu event
            '-hls_list_size', '10',    
            '-hls_flags', 'delete_segments',  # Xóa các phân đoạn cũ
            output_file              # File đầu ra M3U8
        ]

        messagebox.showinfo("Info", "Đang chuyển đổi RTSP sang M3U8, vui lòng chờ...")

        try:
            # Tạo một luồng để thực hiện xóa file cũ liên tục trong khi ffmpeg đang chạy
            threading.Thread(target=self.clean_old_segments, args=(output_dir,)).start()

            # Thực hiện lệnh ffmpeg
            process = subprocess.Popen(command)

            while process.poll() is None:
                if self.stop_flag:
                    process.terminate()  # Dừng quá trình ffmpeg
                    messagebox.showinfo("Info", "Quá trình đã bị dừng.")
                    break
                time.sleep(1)

            if not self.stop_flag:
                messagebox.showinfo("Success", f"Chuyển đổi thành công: {output_file}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Có lỗi xảy ra: {e}")

    def clean_old_segments(self, output_dir, max_segments=10):
        # Liên tục kiểm tra và xóa các phân đoạn cũ
        while not self.stop_flag:
            ts_files = glob.glob(os.path.join(output_dir, '*.ts'))
            ts_files.sort(key=os.path.getmtime)

            # In danh sách các file hiện có để kiểm tra
            print(f"Các file hiện tại: {ts_files}")

            # Nếu số lượng file vượt quá giới hạn, xóa những file cũ hơn
            if len(ts_files) > max_segments:
                files_to_delete = ts_files[:-max_segments]  # Giữ lại các file mới nhất
                for ts_file in files_to_delete:
                    try:
                        print(f"Đang xóa file: {ts_file}")
                        os.remove(ts_file)
                    except Exception as e:
                        print(f"Lỗi khi xóa {ts_file}: {e}")

            # Đợi một khoảng thời gian ngắn rồi kiểm tra lại (ví dụ 10 giây)
            time.sleep(10)

    def __del__(self):
        if self.vid is not None:
            self.vid.release()
# Tạo cửa sổ GUI
root = tk.Tk()
app = VideoStreamApp(root)

# Chạy vòng lặp chính
root.mainloop()
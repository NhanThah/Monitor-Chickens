import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Biến toàn cục
rect_start = None
rect_end = None
drawing = False
image = None
frame = None
paused = False
displayed_frame = None
scale = 0.5


def on_mouse_down(event):
    global rect_start, drawing
    # Điều chỉnh tọa độ chuột theo tỷ lệ thu nhỏ
    rect_start = (int(event.x / scale), int(event.y / scale))
    drawing = True


def on_mouse_move(event):
    global rect_end, drawing, displayed_frame
    if drawing:
        # Điều chỉnh tọa độ chuột theo tỷ lệ thu nhỏ
        rect_end = (int(event.x / scale), int(event.y / scale))
        img_copy = displayed_frame.copy()
        cv2.rectangle(img_copy, rect_start, rect_end, (0, 255, 0), 2)
        show_frame(img_copy)


def on_mouse_up(event):
    global drawing
    drawing = False
    confirm_save_button.config(state="normal")  # Kích hoạt nút Lưu


def save_label_and_image():
    global frame, rect_start, rect_end

    if rect_start and rect_end:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])

        if not file_path:
            return

        # Lưu hình ảnh
        cv2.imwrite(file_path, frame)

        # Tính toán tọa độ bounding box theo YOLO format
        x_center = (rect_start[0] + rect_end[0]) / 2 / frame.shape[1]
        y_center = (rect_start[1] + rect_end[1]) / 2 / frame.shape[0]
        width = abs(rect_start[0] - rect_end[0]) / frame.shape[1]
        height = abs(rect_start[1] - rect_end[1]) / frame.shape[0]

        label_path = file_path.replace('.jpg', '.txt')
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {width} {height}\n")

        messagebox.showinfo("Thông báo", "Lưu thành công!")
        confirm_save_button.config(state="disabled")


def pause_video():
    global paused
    paused = True


def resume_video():
    global paused
    paused = False


def show_frame(img, scale=0.5):
    # Thu nhỏ khung hình video với tỷ lệ `scale`
    img_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)


def update_frame():
    global frame, displayed_frame, paused

    if not paused:
        ret, frame = cap.read()
        if ret:
            displayed_frame = frame.copy()
            show_frame(displayed_frame)

    video_label.after(10, update_frame)


# Giao diện Tkinter
root = tk.Tk()
root.title("Video Labeling Tool")

# Tạo nhãn video để hiển thị luồng video
video_label = tk.Label(root)
video_label.grid(row=0, column=0, columnspan=3)
video_label.bind("<Button-1>", on_mouse_down)  # Bắt đầu vẽ khi nhấn chuột trái
video_label.bind("<B1-Motion>", on_mouse_move)  # Vẽ bounding box khi di chuyển chuột
video_label.bind("<ButtonRelease-1>", on_mouse_up)  # Kết thúc vẽ khi nhả chuột

# Nút bắt đầu
start_button = tk.Button(root, text="Chạy Video", command=resume_video)
start_button.grid(row=1, column=0, padx=10, pady=10)

# Nút dừng video
pause_button = tk.Button(root, text="Dừng Video", command=pause_video)
pause_button.grid(row=1, column=1, padx=10, pady=10)

# Nút lưu bounding box và hình ảnh
confirm_save_button = tk.Button(root, text="Lưu", command=save_label_and_image, state="disabled")
confirm_save_button.grid(row=1, column=2, padx=10, pady=10)

# Mở stream RTSP (thay đổi URL)
rtsp_url = "rtsp://internsys:Them1kynuanhe@nongdanonlnine.ddns.net:554/cam/realmonitor?channel=2^&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    messagebox.showerror("Lỗi", "Không thể mở stream RTSP")

# Cập nhật khung hình mỗi 10ms
update_frame()

# Chạy giao diện Tkinter
root.mainloop()

# Giải phóng tài nguyên khi kết thúc
cap.release()
cv2.destroyAllWindows()

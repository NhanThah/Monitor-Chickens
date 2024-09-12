import os
import cv2

def is_bbox_valid(xmin, ymin, xmax, ymax, img_width, img_height):
    # Kiểm tra xem bounding box có nằm trong phạm vi ảnh không
    return not (xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height or xmin >= xmax or ymin >= ymax)

def check_and_delete_invalid_files(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        lbl_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(lbl_path):
            continue  # Không có nhãn cho ảnh này
        
        # Đọc ảnh để lấy kích thước
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh: {img_path}")
            continue
        img_height, img_width = img.shape[:2]

        # Đọc file nhãn và kiểm tra từng bounding box
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        num_boxes = int(lines[0].strip())
        valid = True

        for line in lines[1:]:
            coords = line.strip().split()
            if len(coords) != 4:
                print(f"Lỗi định dạng tọa độ bounding box trong file nhãn: {lbl_path}")
                valid = False
                break
            
            xmin, ymin, xmax, ymax = map(int, coords)
            if not is_bbox_valid(xmin, ymin, xmax, ymax, img_width, img_height):
                print(f"Bounding box ngoài phạm vi ảnh trong file nhãn: {lbl_path}")
                valid = False
                break
        
        # Nếu không hợp lệ, xóa cả file ảnh và nhãn
        if not valid:
            os.remove(img_path)
            os.remove(lbl_path)
            print(f"Đã xóa file không hợp lệ: {img_path} và {lbl_path}")

    print("Đã kiểm tra và xóa các file không hợp lệ.")

def check_image_label_consistency(image_file, label_file):
    # Kiểm tra kích thước ảnh
    img = cv2.imread(image_file)
    if img is None:
        print(f"Không thể đọc ảnh: {image_file}")
        return False
    img_height, img_width = img.shape[:2]

    # Đọc file nhãn
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # Dòng đầu tiên là số lượng bounding box
    try:
        num_boxes = int(lines[0].strip())
    except ValueError:
        print(f"Lỗi định dạng dòng đầu tiên trong file nhãn: {label_file}")
        return False
    
    # Kiểm tra số lượng bounding box có khớp với số dòng còn lại hay không
    if num_boxes != len(lines) - 1:
        print(f"Số lượng bounding box không khớp trong file nhãn: {label_file}")
        return False
    
    # Kiểm tra bounding box có nằm trong kích thước ảnh không
    for line in lines[1:]:
        coords = line.strip().split()
        if len(coords) != 4:
            print(f"Lỗi định dạng tọa độ bounding box trong file nhãn: {label_file}")
            return False
        
        xmin, ymin, xmax, ymax = map(int, coords)

        # Kiểm tra các bounding box có nằm trong kích thước ảnh không
        if not is_bbox_valid(xmin, ymin, xmax, ymax, img_width, img_height):
            print(f"Bounding box ngoài phạm vi ảnh: {label_file}")
            return False

    return True

def check_datasets(images_dir, labels_dir):
    # Danh sách file ảnh và nhãn
    image_files = set(os.listdir(images_dir))
    label_files = set(os.listdir(labels_dir))

    # Kiểm tra sự đồng nhất giữa ảnh và nhãn
    for img_file in image_files:
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            img_path = os.path.join(images_dir, img_file)
            lbl_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(lbl_path):
                print(f"Thiếu nhãn cho ảnh: {img_file}")
            else:
                if not check_image_label_consistency(img_path, lbl_path):
                    print(f"Lỗi đồng nhất giữa ảnh và nhãn cho file: {img_file}")

    for lbl_file in label_files:
        if lbl_file.endswith('.txt'):
            img_file = lbl_file.replace('.txt', '.jpg').replace('.txt', '.png')
            lbl_path = os.path.join(labels_dir, lbl_file)
            img_path = os.path.join(images_dir, img_file)
            
            if not os.path.exists(img_path):
                print(f"Thiếu ảnh cho nhãn: {lbl_file}")

    print("Đã kiểm tra xong dữ liệu.")

# Đường dẫn đến thư mục chứa ảnh và nhãn
images_dir = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Images'
labels_dir = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels'

# Xóa các file ảnh và nhãn không hợp lệ
check_and_delete_invalid_files(images_dir, labels_dir)

# Kiểm tra tính đồng nhất và hợp lệ của dữ liệu
check_datasets(images_dir, labels_dir)
def delete_invalid_files_from_report(report_files):
    for lbl_path in report_files:
        # Xóa file nhãn
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
            print(f"Đã xóa nhãn không hợp lệ: {lbl_path}")
        else:
            print(f"File nhãn không tồn tại: {lbl_path}")

        # Xóa file ảnh tương ứng
        img_path = lbl_path.replace('.txt', '.jpg').replace('.txt', '.png')
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Đã xóa ảnh không hợp lệ: {img_path}")
        else:
            print(f"File ảnh không tồn tại: {img_path}")

# Danh sách các file nhãn có lỗi đồng nhất
report_files = [
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0472.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0317.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0469.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0350.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0550.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0396.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0365.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0277.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0167.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0488.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0304.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0415.txt',
    'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\frame_0592.txt'
]

# Xóa các file không hợp lệ
delete_invalid_files_from_report(report_files)
import os

def delete_images_without_labels(images_dir, labels_dir):
    # Danh sách các ảnh không có nhãn
    image_files = set(f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png')))
    
    for img_file in image_files:
        lbl_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        lbl_path = os.path.join(labels_dir, lbl_file)
        
        if not os.path.exists(lbl_path):
            img_path = os.path.join(images_dir, img_file)
            
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Đã xóa ảnh không có nhãn: {img_path}")
            else:
                print(f"File ảnh không tồn tại: {img_path}")

# Đường dẫn đến thư mục chứa ảnh và nhãn
images_dir = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Images'
labels_dir = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels'

# Xóa các ảnh không có nhãn tương ứng
delete_images_without_labels(images_dir, labels_dir)
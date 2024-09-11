import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
class CCDataset(Dataset):
    def __init__(self, image_dir, label_dir, augmentation=None):
        self.image_dir = image_dir  # Thư mục chứa ảnh
        self.label_dir = label_dir  # Thư mục chứa nhãn bounding box
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]  # Lọc ảnh
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Đọc ảnh và nhãn
        image = Image.open(image_path).convert('RGB')
        boxes, labels = self.get_boxes(label_path)
        
        # Chuyển bounding boxes sang dạng torch tensor
        boxes = np.array(boxes, dtype=np.float32).tolist()
        labels = np.array(labels, dtype=np.int64).tolist()
        targets = {'boxes': boxes, 'labels': labels}  # Giả sử tất cả các đối tượng thuộc về cùng một lớp (class 0)

        # Áp dụng biến đổi
        if self.augmentation:
            augmented = self.augmentation(image=np.array(image), bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']

        # Cập nhật targets
        targets['boxes'] = boxes
        targets['labels'] = labels

        # Chuyển ảnh sang tensor
        image = ToTensorV2()(image=image)
        return image, targets

    def get_boxes(self, label_file):
        boxes = []
        labels = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Bỏ qua dòng đầu tiên (số lượng bounding boxes)
                parts = line.strip().split()
                if len(parts) == 4:
                    x_min, y_min, x_max, y_max = map(float, parts)
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)
        return boxes, labels
augmentation = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.BboxParams(format='pascal_voc', label_fields=['labels']),  # Đảm bảo bbox_params có cấu hình đúng
    ToTensorV2()
])
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # Load Faster R-CNN model pretrained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one (adjusting the number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
TRAIN_DIR = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Images'  # Đường dẫn tới thư mục ảnh
LABEL_DIR = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels\\000'  # Đường dẫn tới thư mục nhãn bounding box

train_dataset = CCDataset(image_dir=TRAIN_DIR, label_dir=LABEL_DIR, augmentation=augmentation)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
try:
    for images, targets in train_loader:
        print(f"Number of images in batch: {len(images)}")
        print(f"Number of targets in batch: {len(targets)}")
        break  # Chỉ kiểm tra một batch đầu tiên
except Exception as e:
    print(f"An error occurred: {e}")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the model and move it to the device
model = get_model(num_classes=2)  # num_classes=2 (1 class for "chicken" + 1 background)
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, targets in train_loader:
        # Move images and targets to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
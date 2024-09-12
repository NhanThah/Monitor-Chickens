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
from torch.utils.data.dataloader import default_collate
class CCDataset(Dataset):
    def __init__(self, image_dir, label_dir, augmentation=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(image_path).convert('RGB')
        boxes, labels = self.get_boxes(label_path)

        image = np.array(image)
        if self.augmentation:
            augmented = self.augmentation(image=image, bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']

        image = F.to_tensor(image)
        return {'image': image, 'targets': {'boxes': boxes, 'labels': labels}}

    def get_boxes(self, label_file):
        boxes = []
        labels = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) == 4:
                    x_min, y_min, x_max, y_max = map(float, parts)
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)
        return boxes, labels
bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])

# Tạo danh sách các biến đổi
augmentation = A.Compose([
    A.RandomCrop(width=512, height=512, p=1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
], bbox_params=bbox_params)
TRAIN_DIR = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Images'
LABEL_DIR = 'D:\\FPT\\FALL2024\\dataset\\Chickens_Dataset\\Labels'
train_dataset = CCDataset(image_dir=TRAIN_DIR, label_dir=LABEL_DIR, augmentation=augmentation)


def collate_fn(batch):
    images = [item['image'] for item in batch]
    targets = [item['targets'] for item in batch]

    # Gộp các images thành một tensor
    images = torch.stack(images)

    # Tạo list rỗng để chứa các bounding boxes và labels
    boxes = []
    labels = []

    for target in targets:
        boxes.append(torch.tensor(target['boxes'], dtype=torch.float32))
        labels.append(torch.tensor(target['labels'], dtype=torch.long))

    # Lấy kích thước lớn nhất của các bounding boxes và labels để padding
    max_boxes = max([len(b) for b in boxes])

    # Padding các bounding boxes và labels để có cùng kích thước
    padded_boxes = [torch.cat([b, torch.zeros((max_boxes - len(b), 4))], dim=0) for b in boxes]
    padded_labels = [torch.cat([l, torch.zeros(max_boxes - len(l), dtype=torch.long)], dim=0) for l in labels]

    # Chuyển đổi list thành tensor
    boxes_tensor = torch.stack(padded_boxes)
    labels_tensor = torch.stack(padded_labels)

    return images, {'boxes': boxes_tensor, 'labels': labels_tensor}


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

try:
    for images, targets in train_loader:
        print(f"Number of images in batch: {images.size(0)}")
        print(f"Number of targets in batch: {len(targets['boxes'])}")
        print(f"Batch image sizes: {images.shape}")
        print(f"Batch target sizes: {[len(boxes) for boxes in targets['boxes']]}")
        break  # Chỉ kiểm tra một batch đầu tiên
except Exception as e:
    print(f"An error occurred: {e}")
def check_image_sizes(dataset):
    sizes = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image']
        sizes.add(image.shape)
    return sizes

# Kiểm tra kích thước ảnh trong dataset
sizes = check_image_sizes(train_dataset)
print(f"Unique image sizes in dataset: {sizes}")
def check_dataset_samples(dataset):
    for i in range(10):  # Xem 5 mẫu đầu tiên
        sample = dataset[i]
        print(f"Sample {i}: Image size {sample['image'].shape}, Targets {sample['targets']}")

# Kiểm tra mẫu dữ liệu
check_dataset_samples(train_dataset)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def get_model(num_classes):
    # Load Faster R-CNN model pretrained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one (adjusting the number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


try:
    for batch in train_loader:
        images, targets = batch  # batch là tuple (images, targets)

        # Kiểm tra kích thước của hình ảnh
        print(f"Batch image sizes: {images.shape}")

        # Kiểm tra kích thước của nhãn
        if isinstance(targets, dict):
            # Đối với nhãn, chúng là dictionary với các tensor con
            num_boxes_per_image = [len(targets['boxes'][i]) for i in range(targets['boxes'].size(0))]
            print(f"Batch target sizes (boxes per image): {num_boxes_per_image}")
        else:
            print(f"Unexpected targets format: {targets}")

        break  # Chỉ kiểm tra một batch đầu tiên
except Exception as e:
    print(f"An error occurred: {e}")


def check_image_sizes(image_dir):
    sizes = set()
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path)
            sizes.add(img.size)
    return sizes


print("Sizes of images in dataset:", check_image_sizes(TRAIN_DIR))
try:
    for images, targets in train_loader:
        print(f"Number of images in batch: {len(images)}")
        print(f"Number of targets in batch: {len(targets)}")
        break  # Chỉ kiểm tra một batch đầu tiên
except Exception as e:
    print(f"An error occurred: {e}")
for images, targets in train_loader:
    print("Targets format:", type(targets))
    print("Targets sample:", targets)
    break  # In thử một batch để kiểm tra
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the model and move it to the device
num_classes = 2  # 1 class (chicken) + 1 background
model = get_model(num_classes)
model.to(device)

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
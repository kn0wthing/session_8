import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

# Dataset parameters
NUM_CLASSES = 10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Transform parameters
train_transforms = A.Compose([
    A.RandomCrop(32, 32, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(
        max_holes=3, max_height=8, max_width=8,
        min_holes=1, min_height=8, min_width=8,
        fill_value=[x * 255 for x in CIFAR_MEAN],
        p=0.5
    ),
    A.OneOf([
        A.RandomBrightnessContrast(p=1.0),
        A.RandomGamma(p=1.0),
    ], p=0.3),
    A.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ToTensorV2()
]) 
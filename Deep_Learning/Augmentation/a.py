import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load image
img_path = "data/hercules.jpg"  # Replace with your image path
image = Image.open(img_path).convert("RGB")

# Resize for consistency
base_transform = transforms.Resize((256, 256))
image = base_transform(image)

# Define individual transforms
transform_list = {
    "Original": transforms.Compose([]),
    "Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
    "Vertical Flip": transforms.RandomVerticalFlip(p=1.0),
    "Rotation": transforms.RandomRotation(degrees=45),
    "Crop": transforms.RandomCrop(size=(128, 128)),
    "Color Jitter": transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
    "Perspective": transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
    "Affine": transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    "Gaussian Blur": transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 2.0)),
    "Random Erasing": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.05, 1), value=0.45),
        transforms.ToPILImage()
    ])
}

# Show each image in a separate window
for name, transform in transform_list.items():
    if name == "Original":
        transformed_img = image
    else:
        transformed_img = transform(image)

    plt.figure()
    plt.imshow(transformed_img)
    plt.title(name)
    plt.axis("off")
    plt.show()

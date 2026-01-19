from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root="/Users/mac/Desktop/config/tiny-imagenet-200/train",
    transform=transform
)

print("Loaded:", len(train_dataset), "training images")
print("Classes:", len(train_dataset.classes))


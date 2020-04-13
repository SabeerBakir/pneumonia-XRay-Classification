from torchvision import transforms


def CNN_Transforms():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])



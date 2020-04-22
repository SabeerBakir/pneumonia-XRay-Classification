from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2 as AToTensor


def CNN_Train_Augments():
    return Default_Train_Augments()


def CNN_Eval_Augments():
    return Default_Eval_Augments()


def AlexNet_Train_Augments():
    return Default_Train_Augments()


def AlexNet_Eval_Augments():
    return Default_Eval_Augments()


def ResNext_Train_Augments():
    return Default_Train_Augments()


def ResNext_Eval_Augments():
    return Default_Eval_Augments()


def MLP_Train_Augments():
    return Default_Train_Augments()


def MLP_Eval_Augments():
    return Default_Eval_Augments()


def Default_Train_Augments():
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(p=0.2),  # Probability 20%
        A.ShiftScaleRotate(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AToTensor()
    ])


def Default_Eval_Augments():
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AToTensor()
    ])

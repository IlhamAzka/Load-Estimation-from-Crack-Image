import torch
import os
import matplotlib.pyplot as plt
import albumentations as A
import torchvision.transforms as transforms
import math
import numpy as np
import cv2

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from pathlib import Path

# MARK: Actual helper function

def crop_image(input_file, height, width):
    img = Image.open(input_file, mode='r')
    img_width, img_height = img.size
    for i in range(img_height//height):
        for j in range(img_width//width):
            box = (j * width, i * height, (j+1) * width, (i+1) * height)
            yield img.crop(box)

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)

def round_to_nearest_power_of_two(n):
    if n <= 0:
        raise ValueError("Must be positive number")
    
    log_n = math.log2(n)

    floor_log = math.floor(log_n)
    ceil_log = math.ceil(log_n)

    # Lower and Upper bound in 2^(x) form
    lower_bound = 2 ** floor_log
    upper_bound = 2 ** ceil_log

    if abs(n - lower_bound) <= abs(n - upper_bound):
        return lower_bound
    else:
        return upper_bound

def moment_of_area(img_bin):
    moment = cv2.moments(img_bin)
    return moment["m10"], moment["m01"], moment["m11"], moment["m00"]

def centroid(img_bin):
    moment = cv2.moments(img_bin)
    centroid_x = moment["m10"] / moment["m00"]
    centroid_y = moment["m01"] / moment["m00"]
    return centroid_x, centroid_y

def polar_moment_of_area(img_bin): 
    moa = moment_of_area(img_bin)
    Ixx = moa[0]
    Iyy = moa[1]
    polar_moment_of_area = Ixx + Iyy
    return polar_moment_of_area

# def mask_to_image(mask: np.ndarray, mask_values):
#     ...

# MARK: DATASET CLASS

class CrackSeg9kDataset(Dataset):
    def __init__(self, root_path: str, img_dim=None, limit=None, augment=False):
        self.image_path = root_path + "Image/"
        self.mask_path = root_path + "Mask/"
        self.img_dim = img_dim
        self.limit = limit
        self.augment = augment
        self.images = sorted([self.image_path + fname for fname in os.listdir(self.image_path)])[:self.limit]
        self.masks = sorted([self.mask_path + fname for fname in os.listdir(self.mask_path)])[:self.limit]

        if img_dim is not None:
            self.transform = transforms.Compose([
                transforms.Resize(img_dim),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        if self.limit is None:
            # get all images without size limit
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        mask = mask.point(lambda x: 255 if x > 128 else 0, '1')

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return min(len(self.images), self.limit)

class Crack500Dataset(Dataset):
    def __init__(self, root_path, img_dim=(128, 128), limit=None, augment=False):
        self.root_path = root_path
        self.img_dim = img_dim
        self.limit = limit
        self.augment = augment
        self.images = sorted([root_path + fname for fname in os.listdir(root_path) if 'jpg' in fname])[:self.limit]
        self.masks = sorted([root_path + fname for fname in os.listdir(root_path) if 'png' in fname])[:self.limit]

        self.transform = transforms.Compose([
            # TODO: understand the need for size transformation
            transforms.Resize(self.img_dim),
            transforms.ToTensor()
        ])

        if self.limit is None:
            # get all images without size limit
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        img = self.transform(img)
        mask = self.transform(mask) 

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        pil_mask = to_pil(mask)
        mask = pil_mask.point(lambda x: 255 if x > 128 else 0, '1')
        mask = to_tensor(mask)

        if self.augment is True:
            img = np.array(img)
            mask = np.array(mask)
            aug = self.augmentations(image=img, mask=mask)
            aug["mask"] = aug["mask"].unsqueeze(0)
            return aug["image"]/255, aug["mask"]/1

        return img, mask

    def __len__(self):
        return min(len(self.images), self.limit)

class CrackConglomerateDataset(Dataset):
    def __init__(self, root_path, img_dim=None, limit=None, augment=False):
        self.root_path = root_path
        self.img_dim = img_dim
        self.limit = limit
        self.augment = augment
        self.images = sorted([root_path + 'images/' + fname for fname in os.listdir(root_path + 'images/')])[:self.limit]
        self.masks = sorted([root_path + 'masks/' + fname for fname in os.listdir(root_path + 'masks/')])[:self.limit]

        if img_dim is not None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_dim),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])


        if self.limit is None:
            # get all images without size limit
            self.limit = len(self.images)

    # augmentations = A.Compose(
    #     [                   
    #     A.HorizontalFlip(p=0.7),
    #     A.Rotate(p=0.7),
    #     A.OneOf([
    #             A.RandomGamma(),
    #             A.RandomBrightnessContrast()
    #             ], p=0.3),
    #     A.OneOf([
    #             A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
    #             A.GridDistortion(),
    #             A.OpticalDistortion(distort_limit=2, shift_limit=0.5)
    #             ], p=0.3),
    #     A.Resize(height=128, width=128),
    #     A.ToTensorV2()
    #     ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        mask = mask.point(lambda x: 255 if x > 128 else 0, '1')
        
        if self.augment is True:
            img = np.array(img)
            mask = np.array(mask)
            aug = self.augmentations(image=img, mask=mask)
            aug["mask"] = aug["mask"].unsqueeze(0)
            return aug["image"]/255, aug["mask"]/1

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return min(len(self.images), self.limit)

class OmniCrack30kDataset(Dataset):
    def __init__(self, image_path, mask_path, img_dim=(128, 128), limit=None, augment=False):
        # self.root_path = root_path
        self.img_dim = img_dim
        self.limit = limit
        self.augment = augment
        self.images = sorted([image_path + fname for fname in os.listdir(image_path)])[:self.limit]
        self.masks = sorted([mask_path + fname for fname in os.listdir(mask_path)])[:self.limit]

        self.transform = transforms.Compose([
            # TODO: understand the need for size transformation
            transforms.Resize(self.img_dim),
            transforms.ToTensor()
        ])

        if self.limit is None:
            # get all images without size limit
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        img = self.transform(img)
        mask = self.transform(mask) 

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        pil_mask = to_pil(mask)
        mask = pil_mask.point(lambda x: 255 if x < 128 else 0, '1')
        mask = to_tensor(mask)

        if self.augment is True:
            img = np.array(img)
            mask = np.array(mask)
            aug = self.augmentations(image=img, mask=mask)
            aug["mask"] = aug["mask"].unsqueeze(0)
            return aug["image"]/255, aug["mask"]/1

        return img, mask

    def __len__(self):
        return min(len(self.images), self.limit)

# MARK: DATASET SETTER

def set_dataset(dataset, img_dim):
    if dataset == "crackseg9k":
        ROOT_DIR = "./CrackDatabase/crackseg9k/"
        TRAIN_DIR = ROOT_DIR + "Train/"
        TEST_DIR = ROOT_DIR + "Test/"
        SEED = 10

        generator = torch.Generator().manual_seed(SEED)
        train_dataset = CrackSeg9kDataset(TRAIN_DIR, img_dim=img_dim)
        test_dataset = CrackSeg9kDataset(TEST_DIR, img_dim=img_dim)
        test_dataset, validation_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    elif dataset == "CRACK500":
        ROOT_DIR = 'CRACK500/'
        TRAIN_DIR = ROOT_DIR + 'traincrop/traincrop/'
        TEST_DIR = ROOT_DIR + 'testcrop/testcrop/'
        VAL_DIR = ROOT_DIR + 'valcrop/valcrop/'

        train_dataset = Crack500Dataset(TRAIN_DIR, augment=False)
        test_dataset = Crack500Dataset(TEST_DIR)
        validation_dataset = Crack500Dataset(VAL_DIR, augment=False)

    elif dataset == "CC":
        ROOT_DIR = './CrackDatabase/crack-conglomerate/Conglomerate Concrete Crack Detection/Conglomerate Concrete Crack Detection/'
        TRAIN_DIR = ROOT_DIR + 'Train/'
        TEST_DIR = ROOT_DIR + 'Test/'

        SEED = 15

        generator = torch.Generator().manual_seed(SEED)
        train_dataset = CrackConglomerateDataset(TRAIN_DIR, img_dim=img_dim)
        test_dataset = CrackConglomerateDataset(TEST_DIR, img_dim=img_dim)
        test_dataset, validation_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    elif dataset == "omnicrack30k":
        ROOT_DIR = './CrackDatabase/omnicrack30k/'
        TRAIN_DIR = 'training/'
        TEST_DIR = 'test/'
        VALIDATION_DIR = 'validation/'
        IMAGE_DIR = ROOT_DIR + 'images/'
        MASK_DIR = ROOT_DIR + 'annotations/'

        train_dataset = OmniCrack30kDataset(
            IMAGE_DIR+TRAIN_DIR, 
            MASK_DIR+TRAIN_DIR
            )
        test_dataset = OmniCrack30kDataset(
            IMAGE_DIR+TEST_DIR,
            MASK_DIR+TEST_DIR 
            )
        validation_dataset = OmniCrack30kDataset(
            IMAGE_DIR+VALIDATION_DIR,
            MASK_DIR+VALIDATION_DIR
            )

    else:
        raise Exception("Dataset Not Available")

    return train_dataset, test_dataset, validation_dataset

# MARK: MODEL METRIC

def dice_coeff(prediction, target, epsilon=1):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

def willmott_index_of_agreement(observed, predicted):
    """
    Calculates Willmott's Index of Agreement (d).
    
    Args:
        observed (array-like): Array of observed values.
        predicted (array-like): Array of predicted values.
        
    Returns:
        float: Willmott's Index of Agreement.
    """
    if len(observed) != len(predicted):
        raise ValueError("Observed and predicted arrays must have the same length.")
        
    n = len(observed)
    obs_mean = np.mean(observed)
    
    numerator = np.sum((predicted - observed)**2)
    denominator = np.sum((np.abs(predicted - obs_mean) + np.abs(observed - obs_mean))**2)
    
    if denominator == 0:
        return 1.0 # Perfect agreement if no variability in observed data
    
    d = 1 - (numerator / denominator)
    return d

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, pred, target):
        prediction_copy = pred.clone()

        prediction_copy[prediction_copy < 0] = 0
        prediction_copy[prediction_copy > 0] = 1

        intersection = abs(torch.sum(prediction_copy * target))
        union = abs(torch.sum(prediction_copy) + torch.sum(target))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

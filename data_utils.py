"""
data_utils.py - Data loading and preprocessing utilities  
Supports CIFAR-10 and ImageNet datasets
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from typing import Tuple, Dict, Any, Optional
from config import Config, DatasetType
from model_utils import get_logits

class MiniImageNetDataset(Dataset):
    """Mini-ImageNet数据集类，支持CSV标签文件"""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, test_ratio: float = 0.2):
        """
        Initialize the Mini-ImageNet dataset

        Args:
            root_dir: Root directory of the dataset
            split: 'train' or 'test'
            transform: Data transformation
            test_ratio: Test set ratio (used when split='test')
        """

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.test_ratio = test_ratio
        
        # 构建样本列表
        self.samples = []
        self.class_to_idx = {}
        self.images_dir = os.path.join(root_dir, 'mini-imagenet', 'images')
        
        # 读取训练集CSV文件
        train_csv_file = os.path.join(root_dir, 'train.csv')
        if not os.path.exists(train_csv_file):
            raise FileNotFoundError(f"Training CSV file does not exist.: {train_csv_file}")
        
        import pandas as pd
        train_df = pd.read_csv(train_csv_file)
        
        # 创建类别映射
        unique_labels = train_df['label'].unique()
        for idx, label in enumerate(sorted(unique_labels)):
            self.class_to_idx[label] = idx
        
        # 构建样本
        all_samples = []
        for _, row in train_df.iterrows():
            filename = row['filename']
            label = row['label']
            img_path = os.path.join(self.images_dir, filename)
            
            if os.path.exists(img_path):
                class_idx = self.class_to_idx[label]
                all_samples.append((img_path, class_idx))
        
        # 根据split分割数据 - 按类别分层分割
        if split == 'train':
            # 按类别分层分割，确保每个类别都有训练和测试样本
            self.samples = []
            for class_idx in range(len(self.class_to_idx)):
                # 获取该类别的所有样本
                class_samples = [sample for sample in all_samples if sample[1] == class_idx]
                # 取前80%作为训练集
                train_count = int(len(class_samples) * (1 - test_ratio))
                self.samples.extend(class_samples[:train_count])
        else:  # split == 'test'
            # 按类别分层分割，确保每个类别都有训练和测试样本
            self.samples = []
            for class_idx in range(len(self.class_to_idx)):
                # 获取该类别的所有样本
                class_samples = [sample for sample in all_samples if sample[1] == class_idx]
                # 取后20%作为测试集
                train_count = int(len(class_samples) * (1 - test_ratio))
                self.samples.extend(class_samples[train_count:])
        
        print(f"Loaded {len(self.samples)} {split} samples with {len(self.class_to_idx)} classes in total.")
        
        # 保存类别数到配置中（如果可能的话）
        self.num_classes = len(self.class_to_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                # 转换为tensor
                image = torchvision.transforms.ToTensor()(image)
        except Exception as e:
            print(f"Failed to load image. {img_path}: {e}")
            # 创建一个黑色图像作为占位符，而不是随机图像
            if self.transform:
                # 创建一个黑色图像并应用变换
                black_image = Image.new('RGB', (224, 224), (0, 0, 0))
                image = self.transform(black_image)
            else:
                image = torch.zeros(3, 224, 224)
        
        return image, label

class ImageNetDataset(Dataset):
    """ImageNet dataset class，loading from folder is supported"""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Initialize the ImageNet dataset

        Args:
           root_dir: Root directory of the dataset
           split: 'train' or 'val'
           transform: Data transformation
       """

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 构建数据路径
        self.data_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"ImageNet {split} data directory does not exist: {self.data_dir}")
        
        # 扫描所有图像文件
        self.samples = []
        self.class_to_idx = {}
        
        if split == 'train':
            # 训练集：每个类别一个文件夹
            for class_name in sorted(os.listdir(self.data_dir)):
                class_path = os.path.join(self.data_dir, class_name)
                if os.path.isdir(class_path):
                    class_idx = len(self.class_to_idx)
                    self.class_to_idx[class_name] = class_idx
                    
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_path, img_name)
                            self.samples.append((img_path, class_idx))
        else:
            # 验证集：所有图像在一个文件夹中
            for img_name in os.listdir(self.data_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.data_dir, img_name)
                    # 从文件名推断类别（这里简化处理）
                    class_idx = hash(img_name) % 1000  # 临时方案
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} {split} samples.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = torchvision.io.read_image(img_path)
            image = image.float() / 255.0  # 归一化到[0,1]
        except Exception as e:
            print(f"Failed to load image. {img_path}: {e}")
            # 返回一个随机图像作为占位符
            image = torch.randn(3, 224, 224)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    get the transition of data
    
    Args:
        config: config for the object
        
    Returns:
        train_transform, test_transform
    """
    mean, std = config.get_normalize_params()
    
    # 训练集变换
    train_transforms = []
    for transform_name in config.dataset_config["train_transform"]:
        if transform_name == "RandomCrop":
            # 对于Mini-ImageNet，使用224x224输入
            input_size = config.model_config["input_size"]
            train_transforms.append(transforms.RandomCrop(input_size))
        elif transform_name == "RandomResizedCrop":
            # 对于Mini-ImageNet，使用224x224输入
            input_size = config.model_config["input_size"]
            train_transforms.append(transforms.RandomResizedCrop(input_size))
        elif transform_name == "RandomHorizontalFlip":
            train_transforms.append(transforms.RandomHorizontalFlip())
        elif transform_name == "Resize":
            train_transforms.append(transforms.Resize(256))
        elif transform_name == "ToTensor":
            train_transforms.append(transforms.ToTensor())
        elif transform_name == "Normalize":
            train_transforms.append(transforms.Normalize(mean, std))
    
    # 如果模型需要调整输入尺寸（如ViT处理CIFAR-10）
    if config.requires_resize():
        # 在ToTensor之前添加Resize
        resize_idx = -2 if "ToTensor" in config.dataset_config["train_transform"] else -1
        train_transforms.insert(resize_idx, transforms.Resize(config.model_config["input_size"]))
    
    train_transform = transforms.Compose(train_transforms)
    
    # 测试集变换
    test_transforms = []
    for transform_name in config.dataset_config["test_transform"]:
        if transform_name == "Resize":
            test_transforms.append(transforms.Resize(256))
        elif transform_name == "CenterCrop":
            input_size = config.model_config["input_size"]
            test_transforms.append(transforms.CenterCrop(input_size))
        elif transform_name == "ToTensor":
            test_transforms.append(transforms.ToTensor())
        elif transform_name == "Normalize":
            test_transforms.append(transforms.Normalize(mean, std))
    
    # 如果模型需要调整输入尺寸（如ViT处理CIFAR-10）
    if config.requires_resize():
        test_transforms.insert(-2, transforms.Resize(config.model_config["input_size"]))
    
    test_transform = transforms.Compose(test_transforms)
    
    return train_transform, test_transform

def load_dataset(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    load dataset
    
    Args:
        config: configs for the object
        
    Returns:
        train_loader, test_loader
    """
    train_transform, test_transform = get_transforms(config)
    
    if config.dataset == DatasetType.CIFAR10:
        # CIFAR-10数据集
        trainset = torchvision.datasets.CIFAR10(
            root=config.dataset_config["data_path"],
            train=True,
            download=False,  # 不下载，使用本地数据
            transform=train_transform
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=config.dataset_config["data_path"],
            train=False,
            download=False,  # 不下载，使用本地数据
            transform=test_transform
        )
    
    elif config.dataset == DatasetType.IMAGENET:
        # Mini-ImageNet数据集（使用CSV标签文件）
        trainset = MiniImageNetDataset(
            root_dir=config.dataset_config["data_path"],
            split='train',
            transform=train_transform
        )
        
        # 使用训练集的一部分作为测试集，确保类别一致
        testset = MiniImageNetDataset(
            root_dir=config.dataset_config["data_path"],
            split='test',
            transform=test_transform
        )
        
        # 动态更新配置中的类别数
        max_classes = max(trainset.num_classes, testset.num_classes)
        config.dataset_config["num_classes"] = max_classes
        config.model_config["num_classes"] = max_classes
    
    else:
        raise ValueError(f"Unsupported dataset type.: {config.dataset}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        trainset,
        batch_size=config.training_config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=config.training_config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training set: {len(trainset)} samples")
    print(f"Test set: {len(testset)} samples")
    return train_loader, test_loader

def normalize_images(images: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Normalize images

    Args:
        images: Input image tensor [B, C, H, W]
        config: Configuration object
    
    Returns:
        Normalized images
    """

    mean, std = config.get_normalize_params()
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
    return (images - mean) / std

def unnormalize_images(images: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Denormalize images

    Args:
        images: Normalized image tensor [B, C, H, W]
        config: Configuration object
    
    Returns:
        Denormalized images
    """

    mean, std = config.get_normalize_params()
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
    return images * std + mean

def get_class_names(config: Config) -> list:
    """
    Get class names

    Args:
        config: Configuration object
    
    Returns:
        List of class names
    """

    return config.get_class_names()

def organize_images_by_class(dataloader: DataLoader, model: torch.nn.Module, 
                           config: Config, max_per_class: int = 200) -> Dict[int, list]:
    """
    Organize images by class

    Args:
        dataloader: Data loader
        model: Model type
        config: Configuration object
        max_per_class: Maximum number of samples per class
    
    Returns:
    Dictionary of images organized by class
    """

    from collections import defaultdict
    
    images_by_class = defaultdict(list)
    model.eval()
    
    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):
            img_tensor = img.to(config.device)
            label_tensor = label.to(config.device)
            
            # 归一化图像
            img_norm = normalize_images(img_tensor, config)
            
            out = model(img_norm)
            logits = get_logits(out)
            pred = logits.argmax(dim=1)

            #pred = model(img_norm).argmax(dim=1)
            
            # 只保留正确预测的样本
            for i in range(len(img_tensor)):
                if pred[i] == label_tensor[i]:
                    images_by_class[label[i].item()].append((img_tensor[i:i+1], idx * dataloader.batch_size + i))
                    
                    # 检查是否收集足够的样本
                    if all(len(imgs) >= max_per_class for imgs in images_by_class.values()) and len(images_by_class) == config.get_num_classes():
                        break
            
            if all(len(imgs) >= max_per_class for imgs in images_by_class.values()) and len(images_by_class) == config.get_num_classes():
                break
    
    return images_by_class

# def create_mini_imagenet_dataset(config: Config, num_classes: int = 100, 
#                                 samples_per_class: int = 600) -> Tuple[DataLoader, DataLoader]:
#     """
#     创建Mini-ImageNet数据集（用于快速实验）
    
#     Args:
#         config: 配置对象
#         num_classes: 类别数量
#         samples_per_class: 每个类别样本数
        
#     Returns:
#         train_loader, test_loader
#     """
#     # 这里可以实现Mini-ImageNet的加载逻辑
#     # 目前返回None，需要根据实际需求实现
#     print("Mini-ImageNet数据集功能待实现")
#     return None, None
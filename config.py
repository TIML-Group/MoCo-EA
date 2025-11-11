"""
config.py - Unified configuration management supporting multiple datasets and architectures  
Supports CIFAR-10/ImageNet datasets and ResNet/ViT architectures
"""


import torch
from enum import Enum
from typing import Dict, Any, Tuple, Optional

class DatasetType(Enum):
    """Dataset type enumeration"""
    CIFAR10 = "cifar10"
    IMAGENET = "imagenet"

class ModelType(Enum):
    """Model architecture type enumeration"""
    RESNET = "resnet"
    VIT = "vit"

class TrainingMode(Enum):
    """Training mode enumeration"""
    PRETRAINED = "pretrained"  # 仅使用预训练模型
    FINE_TUNE = "fine_tune"    # 预训练+微调
    FROM_SCRATCH = "from_scratch"  # 从头训练

class Config:
    """Unified configuration class"""
    
    def __init__(self, 
                 dataset: DatasetType = DatasetType.CIFAR10,
                 model: ModelType = ModelType.RESNET,
                 training_mode: TrainingMode = TrainingMode.FROM_SCRATCH,
                 device: str = "auto"):
        """
        Initialize configuration
        
        Args:
            dataset: Type of dataset
            model: Type of model architecture
            training_mode: Training mode
            device: Device selection
        """
        self.dataset = dataset
        self.model = model
        self.training_mode = training_mode
        
        # 设备配置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 数据集配置
        self.dataset_config = self._get_dataset_config()
        
        # 模型配置
        self.model_config = self._get_model_config()
        
        # 训练配置
        self.training_config = self._get_training_config()
        
        # 实验配置
        self.experiment_config = self._get_experiment_config()
    
    def _get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset-related configuration"""
        if self.dataset == DatasetType.CIFAR10:
            return {
                "name": "CIFAR-10",
                "num_classes": 10,
                "image_size": 32,
                "channels": 3,
                "class_names": ['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck'],
                "normalize_mean": [0.4914, 0.4822, 0.4465],
                "normalize_std": [0.2023, 0.1994, 0.2010],
                "data_path": "./data",
                "train_transform": [
                    "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"
                ],
                "test_transform": ["ToTensor", "Normalize"]
            }
        elif self.dataset == DatasetType.IMAGENET:
            return {
                "name": "Mini-ImageNet",
                "num_classes": 64,  # Mini-ImageNet有64个类别
                "image_size": 224,
                "channels": 3,
                "class_names": None,  # Mini-ImageNet有64个类别，不在这里列出
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
                "data_path": "./data",  # 使用data目录
                "train_transform": [
                    "Resize", "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"
                ],
                "test_transform": [
                    "Resize", "CenterCrop", "ToTensor", "Normalize"
                ]
            }
        else:
            raise ValueError(f"Unsupported dataset type.: {self.dataset}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-related configuration"""
        if self.model == ModelType.RESNET:
            if self.dataset == DatasetType.CIFAR10:
                return {
                    "name": "ResNet-18",
                    "architecture": "resnet18",
                    "pretrained_available": True,
                    "input_size": 32,
                    "num_classes": self.dataset_config["num_classes"],
                    "modifications": {
                        "conv1": "3x3_conv",  # 修改第一层卷积
                        "maxpool": "identity",  # 移除最大池化
                        "fc": "custom"  # 自定义全连接层
                    }
                }
            elif self.dataset == DatasetType.IMAGENET:
                return {
                    "name": "ResNet-18",
                    "architecture": "resnet18",
                    "pretrained_available": True,
                    "input_size": 224,
                    "num_classes": self.dataset_config["num_classes"],
                    "modifications": None  # 使用原始架构
                }
        elif self.model == ModelType.VIT:
            if self.dataset == DatasetType.CIFAR10:
                return {
                    "name": "ViT-Base",
                    "architecture": "vit_base_patch16",
                    "pretrained_available": True,
                    "input_size": 224,  # ViT需要224x224输入
                    "num_classes": self.dataset_config["num_classes"],
                    "patch_size": 16,
                    "embed_dim": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "modifications": {
                        "resize": True,  # 需要将32x32调整到224x224
                        "head": "custom"  # 自定义分类头
                    }
                }
            elif self.dataset == DatasetType.IMAGENET:
                return {
                    "name": "ViT-Base",
                    "architecture": "vit_base_patch16",
                    "pretrained_available": True,
                    "input_size": 224,
                    "num_classes": self.dataset_config["num_classes"],
                    "patch_size": 16,
                    "embed_dim": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "modifications": {
                        "head": "custom"  # 自定义分类头
                    }
                }
        else:
            raise ValueError(f"Unsupported model type.: {self.model}")
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get training-related configuration"""
        base_config = {
            "batch_size": 128 if self.dataset == DatasetType.CIFAR10 else 64,
            "num_epochs": 200 if self.dataset == DatasetType.CIFAR10 else 100,
            "learning_rate": 0.1 if self.dataset == DatasetType.CIFAR10 else 0.01,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "scheduler": "MultiStepLR",
            "milestones": [60, 120, 160] if self.dataset == DatasetType.CIFAR10 else [30, 60, 80],
            "gamma": 0.1,
            "save_best": True,
            "patience": 10
        }
        
        # 根据训练模式调整配置
        if self.training_mode == TrainingMode.PRETRAINED:
            base_config["num_epochs"] = 0  # 不训练
            base_config["learning_rate"] = 0.0
        elif self.training_mode == TrainingMode.FINE_TUNE:
            base_config["learning_rate"] *= 0.1  # 微调使用更小的学习率
            base_config["num_epochs"] = min(base_config["num_epochs"], 50)  # 微调轮数较少
        
        return base_config
    
    def _get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment-related configuration"""
        # 固定类别配置（与原始实验保持一致）
        fixed_classes = {
            'setting_A': 3,        # cat (单图像)
            'setting_B': 3,        # cat (同类)
            'setting_C': (3, 5)    # cat 和 dog (不同类)
        }
        
        # 根据数据集调整类别ID
        if self.dataset == DatasetType.IMAGENET:
            # Mini-ImageNet使用不同的类别ID
            # 使用前几个类别作为示例（可以根据实际需要调整）
            fixed_classes = {
                'setting_A': 0,        # 第一个类别 (n01532829)
                'setting_B': 0,        # 第一个类别 (n01532829)
                'setting_C': (0, 1)    # 前两个类别 (n01532829, n01558993)
            }
        
        return {
            "fixed_classes": fixed_classes,
            "target_samples": 25,
            "pgd_iterations": 40,
            "bezier_iterations": 30,
            "pgd_alpha_factors": {
                'linf': 4.0,
                'l2': 5.0,
                'l1': 10.0
            },
            "epsilons": {
                'linf': 8/255 if self.dataset == DatasetType.CIFAR10 else 4/255,
                'l2': 0.5 if self.dataset == DatasetType.CIFAR10 else 0.25,
                'l1': 10.0 if self.dataset == DatasetType.CIFAR10 else 5.0
            },
            "test_set_size": 100,
            "auxiliary_pool_size": 25,
            "training_pool_start": 130
        }
    
    def get_model_filename(self) -> str:
        """Get model filename"""
        dataset_name = self.dataset.value
        model_name = self.model.value
        mode_name = self.training_mode.value
        
        if self.training_mode == TrainingMode.PRETRAINED:
            return f"{model_name}_{dataset_name}_pretrained.pth"
        elif self.training_mode == TrainingMode.FINE_TUNE:
            return f"{model_name}_{dataset_name}_finetuned.pth"
        else:
            return f"{model_name}_{dataset_name}_trained.pth"
    
    def get_normalize_params(self) -> Tuple[list, list]:
        """Get normalization parameters"""
        mean = self.dataset_config["normalize_mean"]
        std = self.dataset_config["normalize_std"]
        return mean, std
    
    def get_class_names(self) -> list:
        """Get class names"""
        return self.dataset_config["class_names"]
    
    def is_pretrained_available(self) -> bool:
        """Check whether a pretrained model is available"""
        return self.model_config["pretrained_available"]
    
    def requires_resize(self) -> bool:
        """Check whether image resizing is required"""
        if self.model == ModelType.VIT and self.dataset == DatasetType.CIFAR10:
            return True
        return False
    
    def get_input_size(self) -> int:
        """Get model input size"""
        return self.model_config["input_size"]
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return self.dataset_config["num_classes"]
    
    def print_config(self):
        """Print configuration information"""
        print("="*60)
        print("Experiment Configuration Information")
        print("="*60)
        print(f"Dataset: {self.dataset_config['name']}")
        print(f"Model Architecture: {self.model_config['name']}")
        print(f"Training Mode: {self.training_mode.value}")
        print(f"Device: {self.device}")
        print(f"Input Size: {self.get_input_size()}")
        print(f"Number of Classes: {self.get_num_classes()}")
        print(f"Requires Resize: {self.requires_resize()}")
        print(f"Pretrained Model Available: {self.is_pretrained_available()}")
        print("="*60)


# 预定义配置
CIFAR10_RESNET_CONFIG = Config(DatasetType.CIFAR10, ModelType.RESNET, TrainingMode.FROM_SCRATCH)
CIFAR10_VIT_CONFIG = Config(DatasetType.CIFAR10, ModelType.VIT, TrainingMode.FINE_TUNE)
IMAGENET_RESNET_CONFIG = Config(DatasetType.IMAGENET, ModelType.RESNET, TrainingMode.PRETRAINED)
IMAGENET_VIT_CONFIG = Config(DatasetType.IMAGENET, ModelType.VIT, TrainingMode.PRETRAINED)
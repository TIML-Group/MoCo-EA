"""
model_utils.py - Model creation and management utilities  
Supports ResNet and ViT architectures, as well as pretrained model loading
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTForImageClassification, ViTConfig
from typing import Optional, Dict, Any
import os
from config import Config, ModelType, TrainingMode

def create_resnet_model(config: Config) -> nn.Module:
    if config.model_config["architecture"] == "resnet18":
        model = models.resnet18(pretrained=False)
        
        # 根据数据集调整模型
        if config.dataset.value == "cifar10":
            # CIFAR-10调整
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.fc = nn.Linear(512, config.get_num_classes())
        else:
            # ImageNet使用原始架构
            model.fc = nn.Linear(512, config.get_num_classes())
    
    else:
        raise ValueError(f"Unsupported ResNet Architecture: {config.model_config['architecture']}")
    
    return model

def create_vit_model(config: Config) -> nn.Module:
    # 创建ViT配置
    vit_config = ViTConfig(
        image_size=config.model_config["input_size"],
        patch_size=config.model_config["patch_size"],
        num_channels=3,
        hidden_size=config.model_config["embed_dim"],
        num_hidden_layers=config.model_config["num_layers"],
        num_attention_heads=config.model_config["num_heads"],
        num_labels=config.get_num_classes(),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    # 创建模型
    model = ViTForImageClassification(vit_config)
    
    return model

def load_pretrained_model(config: Config) -> nn.Module:
    if config.model == ModelType.RESNET:
        if config.dataset.value == "cifar10":
            # CIFAR-10使用ImageNet预训练权重
            try:
                # 尝试使用本地预训练模型
                local_path = './pretrained_models/cifar10/resnet18_cifar10.pth'
                if os.path.exists(local_path):
                    print("Use local CIFAR-10 ResNet model.")
                    model = models.resnet18(pretrained=False)
                    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    model.maxpool = nn.Identity()
                    model.fc = nn.Linear(512, config.get_num_classes())
                    model.load_state_dict(torch.load(local_path, map_location='cpu'))
                else:
                    # 使用torchvision预训练模型
                    print("Use torchvision pretrained model.")
                    model = models.resnet18(pretrained=True)
                    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                    model.maxpool = nn.Identity()
                    model.fc = nn.Linear(512, config.get_num_classes())
            except Exception as e:
                print(f"Failed to load local model, using torchvision pretrained model instead.: {e}")
                model = models.resnet18(pretrained=True)
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
                model.fc = nn.Linear(512, config.get_num_classes())
        else:
            # Mini-ImageNet使用ImageNet预训练权重
            try:
                # 尝试使用本地预训练模型
                local_path = './pretrained_models/mini_imagenet/resnet18_mini_imagenet.pth'
                if os.path.exists(local_path):
                    print("Use local Mini-ImageNet ResNet model.")
                    model = models.resnet18(pretrained=False)
                    model.fc = nn.Linear(512, config.get_num_classes())
                    model.load_state_dict(torch.load(local_path, map_location='cpu'))
                else:
                    # 使用torchvision预训练模型
                    print("Use torchvision pretrained model.")
                    model = models.resnet18(pretrained=True)
                    model.fc = nn.Linear(512, config.get_num_classes())
            except Exception as e:
                print(f"Failed to load local model, using torchvision pretrained model instead: {e}")
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(512, config.get_num_classes())
    
    elif config.model == ModelType.VIT:
        try:
            # 尝试使用本地ViT模型
            if config.dataset.value == "cifar10":
                local_path = './pretrained_models/cifar10/vit-base-cifar10'
            else:
                local_path = './pretrained_models/mini_imagenet/vit-base-mini-imagenet'
            
            if os.path.exists(local_path):
                print(f"Use local ViT model: {local_path}")
                model = ViTForImageClassification.from_pretrained(
                    local_path,
                    num_labels=config.get_num_classes(),
                    ignore_mismatched_sizes=True
                )
            else:
                # 创建新的ViT模型（无预训练权重）
                print("Create a new ViT model (without pretrained weights).")
                vit_config = ViTConfig(
                    image_size=config.model_config["input_size"],
                    patch_size=config.model_config["patch_size"],
                    num_channels=3,
                    hidden_size=config.model_config["embed_dim"],
                    num_hidden_layers=config.model_config["num_layers"],
                    num_attention_heads=config.model_config["num_heads"],
                    num_labels=config.get_num_classes(),
                )
                model = ViTForImageClassification(vit_config)
        except Exception as e:
            print(f"Failed to load the local ViT model, creating a new one: {e}")
            # 创建新的ViT模型（无预训练权重）
            vit_config = ViTConfig(
                image_size=config.model_config["input_size"],
                patch_size=config.model_config["patch_size"],
                num_channels=3,
                hidden_size=config.model_config["embed_dim"],
                num_hidden_layers=config.model_config["num_layers"],
                num_attention_heads=config.model_config["num_heads"],
                num_labels=config.get_num_classes(),
            )
            model = ViTForImageClassification(vit_config)
    
    else:
        raise ValueError(f"Unsupported model type: {config.model}")
    
    return model

def create_model(config: Config) -> nn.Module:
    if config.training_mode == TrainingMode.PRETRAINED:
        # 使用预训练模型
        model = load_pretrained_model(config)
        print(f"Load pretrained model.: {config.model_config['name']}")
    
    elif config.training_mode == TrainingMode.FINE_TUNE:
        # 加载预训练模型进行微调
        model = load_pretrained_model(config)
        print(f"Load pretrained model for fine-tuning: {config.model_config['name']}")
    
    else:
        # 从头训练
        if config.model == ModelType.RESNET:
            model = create_resnet_model(config)
        elif config.model == ModelType.VIT:
            model = create_vit_model(config)
        else:
            raise ValueError(f"Unsupported model type.: {config.model}")
        
        print(f"Create a new model.: {config.model_config['name']}")
    
    return model

def load_saved_model(config: Config, model_path: str) -> nn.Module:
    """
    加载已保存的模型
    
    Args:
        config: 配置对象
        model_path: 模型文件路径
        
    Returns:
        加载的模型
    """
    # 创建模型
    model = create_model(config)
    
    # 加载权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.device)
        
        # 1) 我们自己训练保存的标准格式（含 'model'）
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
            print(f"Loaded weights (acc={checkpoint.get('acc', 'N/A')}%), "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")

        # 2) ResultsManager 保存的富字典（含 'model_state_dict'）
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded weights from ResultsManager, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")

        # 3) 纯 state_dict（直接是权重字典）
        elif isinstance(checkpoint, dict):
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded raw state_dict, missing={len(missing)}, unexpected={len(unexpected)}")

        else:
            raise RuntimeError(f"Unrecognized checkpoint format at {model_path}")
    else:
        print(f"Model file does not exist.: {model_path}")
        return None
    
    return model

def save_model(model: nn.Module, config: Config, accuracy: float, 
               epoch: int, additional_info: Optional[Dict[str, Any]] = None) -> str:
    """
    保存模型
    
    Args:
        model: 模型
        config: 配置对象
        accuracy: 准确率
        epoch: 训练轮数
        additional_info: 额外信息
        
    Returns:
        保存的文件路径
    """
    filename = config.get_model_filename()
    
    checkpoint = {
        'model': model.state_dict(),
        'config': {
            'dataset': config.dataset.value,
            'model': config.model.value,
            'training_mode': config.training_mode.value,
            'num_classes': config.get_num_classes(),
            'input_size': config.get_input_size()
        },
        'acc': accuracy,
        'epoch': epoch
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filename)
    print(f"Model saved to: {filename}")
    
    return filename

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型信息
    
    Args:
        model: 模型
        
    Returns:
        模型信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        'architecture': str(type(model).__name__)
    }

def freeze_model_layers(model: nn.Module, freeze_pattern: str = "backbone") -> None:
    """
    冻结模型层
    
    Args:
        model: 模型
        freeze_pattern: 冻结模式 ("backbone", "all", "none")
    """
    if freeze_pattern == "backbone":
        # 冻结骨干网络，只训练分类头
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
        print("Backbone network frozen; only the classification head will be trained.")
    
    elif freeze_pattern == "all":
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        print("All parameters have been frozen.")
    
    elif freeze_pattern == "none":
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
        print("All parameters have been unfrozen.")
    
    else:
        raise ValueError(f"Unsupported freeze mode: {freeze_pattern}")

def count_parameters(model: nn.Module) -> tuple:
    """
    统计模型参数数量
    
    Args:
        model: 模型
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_model_summary(model: nn.Module, config: Config) -> Dict[str, Any]:
    """
    打印模型摘要
    
    Args:
        model: 模型
        config: 配置对象
        
    Returns:
        模型信息字典
    """
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Architecture: {config.model_config['name']}")
    print(f"Dataset: {config.dataset_config['name']}")
    print(f"Training Mode: {config.training_mode.value}")
    print(f"Input Size: {config.get_input_size()}")
    print(f"Number of Classes: {config.get_num_classes()}")

    
    total_params, trainable_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print("="*60)


    # 返回模型信息
    return {
        'architecture': config.model_config['name'],
        'dataset': config.dataset_config['name'],
        'training_mode': config.training_mode.value,
        'input_size': config.get_input_size(),
        'num_classes': config.get_num_classes(),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }

# 兼容性函数，保持与原始代码的接口一致
def create_resnet18_cifar10() -> nn.Module:
    """创建CIFAR-10 ResNet-18模型（兼容性函数）"""
    config = Config()
    return create_resnet_model(config)

def load_model() -> nn.Module:
    """加载模型（兼容性函数）"""
    config = Config()
    model_path = config.get_model_filename()
    return load_saved_model(config, model_path)

def get_logits(output):
    """Return logits Tensor from a model forward output."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unsupported model output type: {type(output)}")

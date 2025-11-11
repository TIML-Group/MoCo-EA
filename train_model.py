"""
train_model.py - Model training script  
Supports CIFAR-10/ImageNet datasets and ResNet/ViT architectures  
Supports three modes: pretrained, fine-tuning, and training from scratch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
from tqdm import tqdm
from config import Config, DatasetType, ModelType, TrainingMode
from model_utils import create_model, save_model, print_model_summary
from data_utils import load_dataset
from utils import evaluate_accuracy
from results_manager import create_results_manager

def train_model(config: Config, save_results: bool = True):
    """
    Train the model

    Args:
        config: Configuration object
        save_results: Whether to save the results
    """

    print(f"Start Training {config.model_config['name']} On dataset {config.dataset_config['name']}")
    config.print_config()
    
    # 创建结果管理器
    results_manager = create_results_manager() if save_results else None
    
    # 加载数据集
    train_loader, test_loader = load_dataset(config)
    
    # 创建模型
    model = create_model(config)
    model = model.to(config.device)
    
    # 打印模型摘要
    model_info = print_model_summary(model, config)
    
    # 如果是预训练模式，直接返回
    if config.training_mode == TrainingMode.PRETRAINED:
        print("Use the pretrained model and skip training.")
        if save_results and results_manager:
            experiment_name = f"{config.dataset.value}_{config.model.value}_{config.training_mode.value}"
            results_manager.save_model(model, config, experiment_name, model_info)
            results_manager.save_experiment_config(config, experiment_name)
        return model
    
    # 设置优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training_config["learning_rate"],
        momentum=config.training_config["momentum"],
        weight_decay=config.training_config["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.training_config["milestones"],
        gamma=config.training_config["gamma"]
    )
    
    # 训练参数
    num_epochs = config.training_config["num_epochs"]
    best_acc = 0
    patience_counter = 0
    
    # 训练历史记录
    training_history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'learning_rate': []
    }
    
    print(f"Start training for {num_epochs} epochs.")
    print("="*60)
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 处理ViT模型的输出格式
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = model(inputs)
                
                # 处理ViT模型的输出格式
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                loss = criterion(logits, targets)
                
                test_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{test_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        acc = 100.*correct/total
        print(f'Epoch {epoch+1}: Test accuracy = {acc:.2f}%')
        
        # 记录训练历史
        training_history['train_loss'].append(train_loss / len(train_loader))
        training_history['test_loss'].append(test_loss / len(test_loader))
        training_history['train_acc'].append(100. * correct / total)
        training_history['test_acc'].append(acc)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 保存最佳模型
        if acc > best_acc:
            print(f'Saving best model (accuracy: {acc:.2f}%)...')
            save_model(model, config, acc, epoch)
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= config.training_config["patience"]:
            print(f"Early stopping: no improvement for {config.training_config['patience']} consecutive epochs.")
            break
        
        scheduler.step()
    
    print(f'Training completed! Best accuracy: {best_acc:.2f}%')
    
    # 保存训练结果
    if save_results and results_manager:
        experiment_name = f"{config.dataset.value}_{config.model.value}_{config.training_mode.value}"
        
        # 保存训练日志
        log_file = results_manager.save_training_log(
            config, training_history, model_info, experiment_name
        )
        print(f"Training log has been saved to: {log_file}")
        
        # 保存可视化图表
        plot_files = results_manager.save_training_plots(
            training_history, experiment_name
        )
        print(f"Visualization charts have been saved to: {plot_files}")
        
        # 保存模型
        model_file = results_manager.save_model(
            model, config, experiment_name, model_info
        )
        print(f"Model has been saved to: {model_file}")
        
        # 保存实验配置
        config_file = results_manager.save_experiment_config(config, experiment_name)
        print(f"Experiment configuration has been saved to: {config_file}")
        
        # 创建实验总结
        summary_file = results_manager.create_experiment_summary(
            experiment_name, config, training_history, model_info
        )
        print(f"Experiment summary has been saved to: {summary_file}")
    
    return model

def train_model_legacy():
    print("Starting ResNet-18 training on CIFAR-10...")
    
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Create model
    model = create_resnet18_cifar10().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
    
    # Training parameters
    num_epochs = 200
    best_acc = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(testloader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{test_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        acc = 100.*correct/total
        print(f'Epoch {epoch+1}: Test Accuracy = {acc:.2f}%')
        
        # Save checkpoint
        if acc > best_acc:
            print(f'Saving best model (accuracy: {acc:.2f}%)...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, 'resnet18_cifar10_best.pth')
            best_acc = acc
        
        torch.save(model.state_dict(), 'resnet18_cifar10_latest.pth')
        scheduler.step()
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')

def main():
    """主函数，支持命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'imagenet'], help='Dataset selection')
    parser.add_argument('--model', type=str, default='resnet', 
                       choices=['resnet', 'vit'], help='Model architecture selection')
    parser.add_argument('--mode', type=str, default='from_scratch',
                       choices=['pretrained', 'fine_tune', 'from_scratch'],
                       help='Training mode selection')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device selection (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建配置
    dataset = DatasetType.CIFAR10 if args.dataset == 'cifar10' else DatasetType.IMAGENET
    model = ModelType.RESNET if args.model == 'resnet' else ModelType.VIT
    mode = TrainingMode.PRETRAINED if args.mode == 'pretrained' else \
           TrainingMode.FINE_TUNE if args.mode == 'fine_tune' else TrainingMode.FROM_SCRATCH
    
    config = Config(dataset=dataset, model=model, training_mode=mode, device=args.device)
    
    # 检查模型文件是否已存在
    model_filename = config.get_model_filename()
    if os.path.exists(model_filename):
        response = input(f"Model file {model_filename} already exists. Retrain?(y/n): ")
        if response.lower() != 'y':
            print("Exit training.")
            return
    
    # 创建数据目录
    os.makedirs('./data', exist_ok=True)
    
    # 开始训练
    train_model(config)

if __name__ == "__main__":
    main()
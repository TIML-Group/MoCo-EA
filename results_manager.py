#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果管理器
负责保存实验结果、可视化图表和训练过程
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
from typing import Dict, List, Any, Optional
import seaborn as sns

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResultsManager:
    """实验结果管理器"""
    
    def __init__(self, base_dir: str = "results"):
        """
        初始化结果管理器
        
        Args:
            base_dir: 结果保存的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.dirs = {
            'models': self.base_dir / 'models',
            'plots': self.base_dir / 'plots',
            'logs': self.base_dir / 'logs',
            'data': self.base_dir / 'data',
            'configs': self.base_dir / 'configs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
    
    def save_training_log(self, 
                         config: Any, 
                         training_history: Dict[str, List[float]], 
                         model_info: Dict[str, Any],
                         experiment_name: Optional[str] = None) -> str:
        """
        保存训练日志
        
        Args:
            config: 配置对象
            training_history: 训练历史记录
            model_info: 模型信息
            experiment_name: 实验名称
            
        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name is None:
            experiment_name = f"{config.dataset.value}_{config.model.value}_{config.training_mode.value}"
        
        log_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config': {
                'dataset': config.dataset.value,
                'model': config.model.value,
                'training_mode': config.training_mode.value,
                'device': str(config.device),
                'input_size': config.model_config['input_size'],
                'num_classes': config.get_num_classes(),
                'batch_size': config.training_config['batch_size'],
                'learning_rate': config.training_config['learning_rate'],
                'num_epochs': config.training_config['num_epochs']
            },
            'model_info': model_info,
            'training_history': training_history,
            'metrics': {
                'final_train_loss': training_history.get('train_loss', [0])[-1],
                'final_test_loss': training_history.get('test_loss', [0])[-1],
                'final_train_acc': training_history.get('train_acc', [0])[-1],
                'final_test_acc': training_history.get('test_acc', [0])[-1],
                'best_test_acc': max(training_history.get('test_acc', [0])),
                'total_epochs': len(training_history.get('train_loss', []))
            }
        }
        
        # 保存JSON日志
        log_file = self.dirs['logs'] / f"{experiment_name}_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # 保存pickle格式（用于后续分析）
        pickle_file = self.dirs['data'] / f"{experiment_name}_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(log_data, f)
        
        return str(log_file)
    
    def save_training_plots(self, 
                           training_history: Dict[str, List[float]], 
                           experiment_name: str,
                           save_format: str = 'png') -> List[str]:
        """
        保存训练过程可视化图表
        
        Args:
            training_history: 训练历史记录
            experiment_name: 实验名称
            save_format: 保存格式
            
        Returns:
            保存的文件路径列表
        """
        saved_files = []
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 损失函数曲线
        if 'train_loss' in training_history and 'test_loss' in training_history:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(training_history['train_loss']) + 1)
            
            plt.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, training_history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
            
            plt.title(f'Training and Validation Loss - {experiment_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            loss_file = self.dirs['plots'] / f"{experiment_name}_loss.{save_format}"
            plt.savefig(loss_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(str(loss_file))
        
        # 2. 准确率曲线
        if 'train_acc' in training_history and 'test_acc' in training_history:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(training_history['train_acc']) + 1)
            
            plt.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            plt.plot(epochs, training_history['test_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            
            plt.title(f'Training and Validation Accuracy - {experiment_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.tight_layout()
            
            acc_file = self.dirs['plots'] / f"{experiment_name}_accuracy.{save_format}"
            plt.savefig(acc_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(str(acc_file))
        
        # 3. 综合训练过程图
        if len(training_history) > 0:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(training_history.get('train_loss', [])) + 1)
            
            # 损失函数
            if 'train_loss' in training_history and 'test_loss' in training_history:
                ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
                ax1.plot(epochs, training_history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
                ax1.set_title('Loss Curves', fontweight='bold')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 准确率
            if 'train_acc' in training_history and 'test_acc' in training_history:
                ax2.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
                ax2.plot(epochs, training_history['test_acc'], 'r-', label='Validation Accuracy', linewidth=2)
                ax2.set_title('Accuracy Curves', fontweight='bold')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
            
            # 学习率（如果有）
            if 'learning_rate' in training_history:
                ax3.plot(epochs, training_history['learning_rate'], 'g-', linewidth=2)
                ax3.set_title('Learning Rate Schedule', fontweight='bold')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.grid(True, alpha=0.3)
                ax3.set_yscale('log')
            else:
                ax3.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Learning Rate Schedule', fontweight='bold')
            
            # 训练统计
            stats_text = f"""
            Final Training Loss: {training_history.get('train_loss', [0])[-1]:.4f}
            Final Validation Loss: {training_history.get('test_loss', [0])[-1]:.4f}
            Final Training Accuracy: {training_history.get('train_acc', [0])[-1]:.2f}%
            Final Validation Accuracy: {training_history.get('test_acc', [0])[-1]:.2f}%
            Best Validation Accuracy: {max(training_history.get('test_acc', [0])):.2f}%
            Total Epochs: {len(training_history.get('train_loss', []))}
            """
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10, 
                    verticalalignment='center', fontfamily='monospace')
            ax4.set_title('Training Statistics', fontweight='bold')
            ax4.axis('off')
            
            plt.suptitle(f'Training Process Overview - {experiment_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            overview_file = self.dirs['plots'] / f"{experiment_name}_overview.{save_format}"
            plt.savefig(overview_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(str(overview_file))
        
        return saved_files
    
    def save_model(self, 
                  model: torch.nn.Module, 
                  config: Any, 
                  experiment_name: str,
                  additional_info: Optional[Dict] = None) -> str:
        """
        保存模型
        
        Args:
            model: 训练好的模型
            config: 配置对象
            experiment_name: 实验名称
            additional_info: 额外信息
            
        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_info = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config': {
                'dataset': config.dataset.value,
                'model': config.model.value,
                'training_mode': config.training_mode.value,
                'device': str(config.device),
                'input_size': config.model_config['input_size'],
                'num_classes': config.get_num_classes()
            },
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        if additional_info:
            model_info.update(additional_info)
        
        model_file = self.dirs['models'] / f"{experiment_name}_{timestamp}.pth"
        torch.save(model_info, model_file)
        
        return str(model_file)
    
    def save_experiment_config(self, 
                              config: Any, 
                              experiment_name: str) -> str:
        """
        保存实验配置
        
        Args:
            config: 配置对象
            experiment_name: 实验名称
            
        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        config_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'dataset_config': config.dataset_config,
            'model_config': config.model_config,
            'training_config': config.training_config,
            'experiment_config': config.experiment_config
        }
        
        config_file = self.dirs['configs'] / f"{experiment_name}_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return str(config_file)
    
    def create_experiment_summary(self, 
                                 experiment_name: str,
                                 config: Any,
                                 training_history: Dict[str, List[float]],
                                 model_info: Dict[str, Any]) -> str:
        """
        创建实验总结报告
        
        Args:
            experiment_name: 实验名称
            config: 配置对象
            training_history: 训练历史记录
            model_info: 模型信息
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算统计信息
        final_train_loss = training_history.get('train_loss', [0])[-1]
        final_test_loss = training_history.get('test_loss', [0])[-1]
        final_train_acc = training_history.get('train_acc', [0])[-1]
        final_test_acc = training_history.get('test_acc', [0])[-1]
        best_test_acc = max(training_history.get('test_acc', [0]))
        
        # 生成报告
        report = f"""
# Experiment Summary Report

## Experiment Information
- **Experiment Name**: {experiment_name}
- **Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Dataset**: {config.dataset.value}
- **Model**: {config.model.value}
- **Training Mode**: {config.training_mode.value}
- **Device**: {config.device}

## Model Configuration
- **Input Size**: {config.model_config['input_size']}
- **Number of Classes**: {config.get_num_classes()}
- **Total Parameters**: {model_info.get('total_params', 'N/A'):,}
- **Trainable Parameters**: {model_info.get('trainable_params', 'N/A'):,}

## Training Configuration
- **Batch Size**: {config.training_config['batch_size']}
- **Learning Rate**: {config.training_config['learning_rate']}
- **Number of Epochs**: {config.training_config['num_epochs']}
- **Total Epochs Trained**: {len(training_history.get('train_loss', []))}

## Training Results
- **Final Training Loss**: {final_train_loss:.4f}
- **Final Validation Loss**: {final_test_loss:.4f}
- **Final Training Accuracy**: {final_train_acc:.2f}%
- **Final Validation Accuracy**: {final_test_acc:.2f}%
- **Best Validation Accuracy**: {best_test_acc:.2f}%

## Training History
- **Training Loss**: {training_history.get('train_loss', [])}
- **Validation Loss**: {training_history.get('test_loss', [])}
- **Training Accuracy**: {training_history.get('train_acc', [])}
- **Validation Accuracy**: {training_history.get('test_acc', [])}

## Files Generated
- Model: `{experiment_name}_{timestamp}.pth`
- Training Log: `{experiment_name}_{timestamp}.json`
- Plots: `{experiment_name}_*.png`
- Config: `{experiment_name}_{timestamp}.json`

---
*Report generated automatically by ResultsManager*
"""
        
        report_file = self.dirs['logs'] / f"{experiment_name}_{timestamp}_summary.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return str(report_file)
    
    def get_experiment_list(self) -> List[Dict[str, Any]]:
        """
        获取所有实验的列表
        
        Returns:
            实验信息列表
        """
        experiments = []
        
        # 扫描日志文件
        for log_file in self.dirs['logs'].glob("*.json"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    experiments.append({
                        'name': data.get('experiment_name', 'Unknown'),
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'dataset': data.get('config', {}).get('dataset', 'Unknown'),
                        'model': data.get('config', {}).get('model', 'Unknown'),
                        'training_mode': data.get('config', {}).get('training_mode', 'Unknown'),
                        'best_acc': data.get('metrics', {}).get('best_test_acc', 0),
                        'final_acc': data.get('metrics', {}).get('final_test_acc', 0),
                        'log_file': str(log_file)
                    })
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")
        
        return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
    
    def cleanup_old_experiments(self, keep_days: int = 30):
        """
        清理旧的实验文件
        
        Args:
            keep_days: 保留天数
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for dir_path in self.dirs.values():
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        try:
                            file_path.unlink()
                            print(f"Deleted old file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")

def create_results_manager(base_dir: str = "results") -> ResultsManager:
    """
    创建结果管理器实例
    
    Args:
        base_dir: 结果保存的基础目录
        
    Returns:
        ResultsManager实例
    """
    return ResultsManager(base_dir)
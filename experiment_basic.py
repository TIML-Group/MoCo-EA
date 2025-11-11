"""
experiment_basic.py - Basic Bézier-curve experiment
Adversarial attack experiments across multiple datasets and architectures
- Use a fixed class configuration (consistent with other experiments)
- Collect all available successful samples
- Support CIFAR-10/ImageNet datasets and ResNet/ViT architectures
"""


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import random
import argparse

from utils import PGDAttack, normalize_cifar10
from bezier_core import BezierAdversarialUnconstrained
from config import Config, DatasetType, ModelType, TrainingMode
from model_utils import create_model, load_saved_model
from data_utils import load_dataset, organize_images_by_class, normalize_images

def set_random_seeds(seed=42):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(config: Config):
    """加载模型"""
    model_filename = config.get_model_filename()
    
    if os.path.exists(model_filename):
        model = load_saved_model(config, model_filename)
        if model is None:
            print(f"Failed to load model file.: {model_filename}")
            return None
    else:
        print(f"Model file does not exist.: {model_filename}")
        print("Please run train_model.py first to train a model")
        return None
    
    return model.to(config.device).eval()

def organize_images_by_class_legacy(dataloader, model, config: Config, max_per_class=200):
    """按类别组织图像（兼容性函数）"""
    return organize_images_by_class(dataloader, model, config, max_per_class)

def evaluate_bezier_path(model, bezier_obj, delta1, theta, delta2, x1, x2, y1, y2, 
                        config: Config, setting_type='A', num_points=50):
    """评估贝塞尔路径在多个点上的成功率（排除端点）"""
    # 排除端点以避免评估偏差
    t_values = torch.linspace(0.02, 0.98, num_points).to(config.device)
    
    if setting_type == 'A':
        # Setting A: 单图像 - x1和x2相同
        x2 = x1
        y2 = y1
    
    # 跟踪每个图像的成功率
    success_x1 = 0
    success_x2 = 0
    success_both = 0
    
    with torch.no_grad():
        for t in t_values:
            delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
            delta_t = bezier_obj.project_norm_ball(delta_t)
            
            # 测试第一个图像
            x1_adv = torch.clamp(x1 + delta_t, 0, 1)
            x1_adv_norm = normalize_images(x1_adv, config)
            outputs1 = model(x1_adv_norm)
            
            # 处理ViT模型的输出格式
            if hasattr(outputs1, 'logits'):
                logits1 = outputs1.logits
            else:
                logits1 = outputs1
            pred1 = logits1.argmax(dim=1).item()
            s1 = pred1 != y1.item()
            
            if setting_type == 'A':
                # Setting A只测试一个图像
                if s1:
                    success_x1 += 1
                    success_x2 += 1  # Setting A中与x1相同
                    success_both += 1
            else:
                # Settings B和C测试第二个图像
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                x2_adv_norm = normalize_images(x2_adv, config)
                outputs2 = model(x2_adv_norm)
                
                # 处理ViT模型的输出格式
                if hasattr(outputs2, 'logits'):
                    logits2 = outputs2.logits
                else:
                    logits2 = outputs2
                pred2 = logits2.argmax(dim=1).item()
                s2 = pred2 != y2.item()
                
                if s1:
                    success_x1 += 1
                if s2:
                    success_x2 += 1
                if s1 and s2:
                    success_both += 1
    
    return {
        'success_rate_x1': success_x1 / num_points,
        'success_rate_x2': success_x2 / num_points,
        'success_rate_both': success_both / num_points,
        'success_rate_avg': (success_x1 + success_x2) / (2 * num_points)
    }

def collect_samples_setting_A(images_by_class, model, pgd_attack, bezier, config: Config, target_samples=25):
    """收集Setting A（单图像）的样本"""
    class_id = config.experiment_config["fixed_classes"]['setting_A']
    
    if class_id not in images_by_class:
        print(f"    Error: Class {class_id} is not available.")
        return []
    
    available_images = images_by_class[class_id]
    print(f"    Setting A: Using class {class_id}, {len(available_images)} images available.")
    
    samples = []
    attempts = 0
    max_attempts = min(len(available_images) * 10, 500)  # 合理限制
    
    pbar = tqdm(total=target_samples, desc=f"    Collecting Setting A samples.")
    
    while len(samples) < target_samples and attempts < max_attempts:
        # 选择一个图像
        img_idx = attempts % len(available_images)
        x = available_images[img_idx][0]
        y = torch.tensor([class_id]).to(config.device)
        
        attempts += 1
        
        # 为同一图像生成两个扰动
        delta1 = pgd_attack.perturb(x, y)
        delta2 = pgd_attack.perturb(x, y)
        
        # 验证两个端点都有效
        with torch.no_grad():
            x_adv_d1 = torch.clamp(x + delta1, 0, 1)
            x_adv_d2 = torch.clamp(x + delta2, 0, 1)
            x_adv_d1_norm = normalize_images(x_adv_d1, config)
            x_adv_d2_norm = normalize_images(x_adv_d2, config)
            # 处理ViT模型的输出格式
            outputs_d1 = model(x_adv_d1_norm)
            outputs_d2 = model(x_adv_d2_norm)
            
            if hasattr(outputs_d1, 'logits'):
                pred_d1 = outputs_d1.logits.argmax(dim=1)
            else:
                pred_d1 = outputs_d1.argmax(dim=1)
                
            if hasattr(outputs_d2, 'logits'):
                pred_d2 = outputs_d2.logits.argmax(dim=1)
            else:
                pred_d2 = outputs_d2.argmax(dim=1)
            
            if pred_d1 == y or pred_d2 == y:
                continue
        
        # 优化贝塞尔路径
        theta, _, _, theta_norms = bezier.optimize_setting_A(x, y, delta1, delta2)
        
        # 评估路径
        eval_results = evaluate_bezier_path(
            model, bezier, delta1, theta, delta2, 
            x, x, y, y, config, setting_type='A'
        )
        
        samples.append({
            'success_rate': eval_results['success_rate_avg'],
            'detailed_results': eval_results,
            'theta_norm': theta_norms[-1],
            'image_idx': img_idx
        })
        
        pbar.update(1)

    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting A (attempts: {attempts}).")
    
    return samples

def collect_samples_setting_B(images_by_class, model, pgd_attack, bezier, config: Config, target_samples=25):
    """收集Setting B（同类）的样本"""
    class_id = config.experiment_config["fixed_classes"]['setting_B']
    
    if class_id not in images_by_class:
        print(f"    Error: Class {class_id} is not available.")
        return []
    
    available_images = images_by_class[class_id]
    print(f"    Setting B: Using class {class_id}, {len(available_images)} images available.")
    
    if len(available_images) < 2:
        print(f"    Error: Setting B requires at least 2 images.")
        return []
    
    samples = []
    attempts = 0
    max_attempts = min(len(available_images) * len(available_images), 500)
    
    pbar = tqdm(total=target_samples, desc=f"    Collecting Setting B samples.")
    
    while len(samples) < target_samples and attempts < max_attempts:
        # 从同一类别选择两个不同的图像
        idx1 = attempts % len(available_images)
        idx2 = (attempts + 1 + (attempts // len(available_images))) % len(available_images)
        
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(available_images)
        
        x1 = available_images[idx1][0]
        x2 = available_images[idx2][0]
        y = torch.tensor([class_id]).to(config.device)
        
        attempts += 1
        
        # 为每个图像生成扰动
        delta1 = pgd_attack.perturb(x1, y)
        delta2 = pgd_attack.perturb(x2, y)
        
        # 验证端点有效
        with torch.no_grad():
            x1_adv = torch.clamp(x1 + delta1, 0, 1)
            x2_adv = torch.clamp(x2 + delta2, 0, 1)
            x1_adv_norm = normalize_images(x1_adv, config)
            x2_adv_norm = normalize_images(x2_adv, config)
            
            # 处理ViT模型的输出格式
            outputs1 = model(x1_adv_norm)
            outputs2 = model(x2_adv_norm)
            
            if hasattr(outputs1, 'logits'):
                pred1 = outputs1.logits.argmax(1)
            else:
                pred1 = outputs1.argmax(1)
                
            if hasattr(outputs2, 'logits'):
                pred2 = outputs2.logits.argmax(1)
            else:
                pred2 = outputs2.argmax(1)
            
            if pred1 == y or pred2 == y:
                continue
        
        # 优化贝塞尔路径
        theta, _, _, theta_norms = bezier.optimize_setting_B(x1, x2, y, delta1, delta2)
        
        # 评估路径
        eval_results = evaluate_bezier_path(
            model, bezier, delta1, theta, delta2, 
            x1, x2, y, y, config, setting_type='B'
        )
        
        samples.append({
            'success_rate': eval_results['success_rate_both'],
            'detailed_results': eval_results,
            'theta_norm': theta_norms[-1],
            'image_indices': (idx1, idx2)
        })
        
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting B (attempts: {attempts}).")
    
    return samples

def collect_samples_setting_C(images_by_class, model, pgd_attack, bezier, config: Config, target_samples=25):
    """收集Setting C（不同类）的样本"""
    class_id1, class_id2 = config.experiment_config["fixed_classes"]['setting_C']
    
    if class_id1 not in images_by_class or class_id2 not in images_by_class:
        print(f"    Error: Class {class_id1} or {class_id2} is not available.")
        return []
    
    available_images1 = images_by_class[class_id1]
    available_images2 = images_by_class[class_id2]
    print(f"    Setting C: Using class {class_id1} ({len(available_images1)} images). "
          f"and {class_id2} ({len(available_images2)} images).")
    
    samples = []
    attempts = 0
    max_attempts = min(len(available_images1) * len(available_images2), 500)
    
    pbar = tqdm(total=target_samples, desc=f"    Collecting Setting C samples.")
    
    while len(samples) < target_samples and attempts < max_attempts:
        # 从每个类别选择一张图像
        idx1 = attempts % len(available_images1)
        idx2 = attempts % len(available_images2)
        
        x1 = available_images1[idx1][0]
        x2 = available_images2[idx2][0]
        y1 = torch.tensor([class_id1]).to(config.device)
        y2 = torch.tensor([class_id2]).to(config.device)
        
        attempts += 1
        
        # 为每个图像生成扰动
        delta1 = pgd_attack.perturb(x1, y1)
        delta2 = pgd_attack.perturb(x2, y2)
        
        # 验证端点有效
        with torch.no_grad():
            x1_adv = torch.clamp(x1 + delta1, 0, 1)
            x2_adv = torch.clamp(x2 + delta2, 0, 1)
            x1_adv_norm = normalize_images(x1_adv, config)
            x2_adv_norm = normalize_images(x2_adv, config)
            
            # 处理ViT模型的输出格式
            outputs1 = model(x1_adv_norm)
            outputs2 = model(x2_adv_norm)
            
            if hasattr(outputs1, 'logits'):
                pred1 = outputs1.logits.argmax(1)
            else:
                pred1 = outputs1.argmax(1)
                
            if hasattr(outputs2, 'logits'):
                pred2 = outputs2.logits.argmax(1)
            else:
                pred2 = outputs2.argmax(1)
            
            if pred1 == y1 or pred2 == y2:
                continue
        
        # 优化贝塞尔路径
        theta, _, _, theta_norms = bezier.optimize_setting_C(x1, x2, y1, y2, delta1, delta2)
        
        # 评估路径
        eval_results = evaluate_bezier_path(
            model, bezier, delta1, theta, delta2, 
            x1, x2, y1, y2, config, setting_type='C'
        )
        
        samples.append({
            'success_rate': eval_results['success_rate_both'],
            'detailed_results': eval_results,
            'theta_norm': theta_norms[-1],
            'image_indices': (idx1, idx2)
        })
        
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting C (attempts: {attempts}).")
    
    return samples

def print_results_fixed(results, config):
    """Print results in fixed class format"""
    print("\n" + "="*120)
    print("BASIC EXPERIMENTS - FIXED CLASSES RESULTS")
    print("="*120)
    
    # 获取固定类别配置
    fixed_classes = config.experiment_config["fixed_classes"]
    
    # Print configuration
    if config.dataset == DatasetType.CIFAR10:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        print("\nFixed Configuration:")
        print(f"  Setting A: Class {fixed_classes['setting_A']} ({class_names[fixed_classes['setting_A']]})")
        print(f"  Setting B: Class {fixed_classes['setting_B']} ({class_names[fixed_classes['setting_B']]})")
        c1, c2 = fixed_classes['setting_C']
        print(f"  Setting C: Classes {c1} ({class_names[c1]}) and {c2} ({class_names[c2]})")
    else:
        # Mini-ImageNet没有预定义的类别名称
        print("\nFixed Configuration:")
        print(f"  Setting A: Class {fixed_classes['setting_A']}")
        print(f"  Setting B: Class {fixed_classes['setting_B']}")
        c1, c2 = fixed_classes['setting_C']
        print(f"  Setting C: Classes {c1} and {c2}")
    
    # Detailed results for each norm and setting
    for norm in ['linf', 'l2', 'l1']:
        if norm not in results:
            continue
            
        print(f"\n{'='*100}")
        print(f"{norm.upper()} NORM RESULTS")
        print(f"{'='*100}")
        
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            if setting not in results[norm]:
                continue
                
            setting_name = {
                'setting_A': 'Setting A (Single Image)',
                'setting_B': 'Setting B (Same Class)',
                'setting_C': 'Setting C (Different Classes)'
            }[setting]
            
            print(f"\n{setting_name}:")
            samples = results[norm][setting]
            
            if not samples:
                print("  No samples collected")
                continue
            
            # Extract metrics
            if setting == 'setting_A':
                success_rates = [s['success_rate'] for s in samples]
                theta_norms = [s['theta_norm'] for s in samples]
                
                avg_success = np.mean(success_rates) * 100
                std_success = np.std(success_rates) * 100
                avg_theta = np.mean(theta_norms)
                std_theta = np.std(theta_norms)
                
                print(f"  Number of samples:     {len(samples)}")
                print(f"  Path success rate:     {avg_success:>6.1f} ± {std_success:<5.1f}%")
                print(f"  Control point θ/ε:     {avg_theta:>6.2f} ± {std_theta:<5.2f}")
            
            else:
                # For Settings B and C
                x1_rates = [s['detailed_results']['success_rate_x1'] for s in samples]
                x2_rates = [s['detailed_results']['success_rate_x2'] for s in samples]
                both_rates = [s['detailed_results']['success_rate_both'] for s in samples]
                avg_rates = [s['detailed_results']['success_rate_avg'] for s in samples]
                theta_norms = [s['theta_norm'] for s in samples]
                
                # Calculate statistics
                avg_x1 = np.mean(x1_rates) * 100
                std_x1 = np.std(x1_rates) * 100
                avg_x2 = np.mean(x2_rates) * 100
                std_x2 = np.std(x2_rates) * 100
                avg_both = np.mean(both_rates) * 100
                std_both = np.std(both_rates) * 100
                avg_avg = np.mean(avg_rates) * 100
                std_avg = np.std(avg_rates) * 100
                avg_theta = np.mean(theta_norms)
                std_theta = np.std(theta_norms)
                
                print(f"  Number of samples:     {len(samples)}")
                print(f"  Image 1 success rate:  {avg_x1:>6.1f} ± {std_x1:<5.1f}%")
                print(f"  Image 2 success rate:  {avg_x2:>6.1f} ± {std_x2:<5.1f}%")
                print(f"  Both images success:   {avg_both:>6.1f} ± {std_both:<5.1f}%")
                print(f"  Average success rate:  {avg_avg:>6.1f} ± {std_avg:<5.1f}%")
                print(f"  Control point θ/ε:     {avg_theta:>6.2f} ± {std_theta:<5.2f}")
    
    # Summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE - FIXED CLASSES")
    print("="*120)
    print(f"\n{'Setting':<30} {'Norm':<8} {'Samples':<10} {'Img1':<18} {'Img2':<18} {'Both':<18} {'Average':<18}")
    print("-" * 110)
    
    for setting in ['setting_A', 'setting_B', 'setting_C']:
        setting_display = {
            'setting_A': 'Setting A (Single)',
            'setting_B': 'Setting B (Same Class)',
            'setting_C': 'Setting C (Diff Class)'
        }[setting]
        
        for norm in ['linf', 'l2', 'l1']:
            if norm not in results or setting not in results[norm]:
                continue
                
            norm_symbol = {'linf': 'ℓ∞', 'l2': 'ℓ₂', 'l1': 'ℓ₁'}[norm]
            samples = results[norm][setting]
            
            if not samples:
                continue
            
            if setting == 'setting_A':
                success_rates = [s['success_rate'] for s in samples]
                avg_success = np.mean(success_rates) * 100
                std_success = np.std(success_rates) * 100
                
                print(f"{setting_display:<30} {norm_symbol:<8} {len(samples):<10} "
                      f"{'N/A':<18} {'N/A':<18} "
                      f"{'N/A':<18} {avg_success:>6.1f}±{std_success:<5.1f}%")
            else:
                x1_rates = [s['detailed_results']['success_rate_x1'] for s in samples]
                x2_rates = [s['detailed_results']['success_rate_x2'] for s in samples]
                both_rates = [s['detailed_results']['success_rate_both'] for s in samples]
                avg_rates = [s['detailed_results']['success_rate_avg'] for s in samples]
                
                avg_x1 = np.mean(x1_rates) * 100
                std_x1 = np.std(x1_rates) * 100
                avg_x2 = np.mean(x2_rates) * 100
                std_x2 = np.std(x2_rates) * 100
                avg_both = np.mean(both_rates) * 100
                std_both = np.std(both_rates) * 100
                avg_avg = np.mean(avg_rates) * 100
                std_avg = np.std(avg_rates) * 100
                
                img1_str = f"{avg_x1:.1f}±{std_x1:.1f}%"
                img2_str = f"{avg_x2:.1f}±{std_x2:.1f}%"
                both_str = f"{avg_both:.1f}±{std_both:.1f}%"
                avg_str = f"{avg_avg:.1f}±{std_avg:.1f}%"
                
                print(f"{setting_display:<30} {norm_symbol:<8} {len(samples):<10} "
                      f"{img1_str:<18} {img2_str:<18} "
                      f"{both_str:<18} {avg_str:<18}")
    
    print("\nExperimental Framework:")
    print("• Fixed classes across all experiments for consistency")
    print("• No separate test set (evaluates on training path)")
    print("• Collects all available successful samples")
    print("• Aligned with multi_image and comprehensive experiments")

def run_basic_experiments_fixed(config: Config):
    """运行基础贝塞尔曲线实验（固定类别）"""
    # 设置随机种子
    set_random_seeds(42)
    
    # 加载模型
    model = load_model(config)
    if model is None:
        return None
    
    # 加载数据集
    _, test_loader = load_dataset(config)
    
    print("\nOrganize images by class...")
    images_by_class = organize_images_by_class(test_loader, model, config, max_per_class=200)
    
    # 检查固定类别的可用性
    class_names = config.get_class_names()
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(config.get_num_classes())]
    
    print("\nFixed class availability:")
    required_classes = set([config.experiment_config["fixed_classes"]['setting_A'], 
                           config.experiment_config["fixed_classes"]['setting_B']] + 
                           list(config.experiment_config["fixed_classes"]['setting_C']))
    
    for class_id in required_classes:
        if class_id in images_by_class:
            print(f"  Class {class_id} ({class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'}): {len(images_by_class[class_id])} images.")
        else:
            print(f"  Error: Class {class_id} is not available!")
            return None
    
    target_samples = config.experiment_config["target_samples"]
    print(f"\nTarget: {target_samples} samples per norm per setting")
    print(f"PGD attack iterations: {config.experiment_config['pgd_iterations']} (using community-standard α)")
    print(f"Bézier optimization: {config.experiment_config['bezier_iterations']} iterations, learning rate 0.01")
    print("="*80)

    
    all_results = {}
    norms = ['linf', 'l2', 'l1']
    
    for norm in norms:
        print(f"\n{'='*80}")
        print(f"Test {norm.upper()} norm (ε={config.experiment_config['epsilons'][norm]})")
        print(f"{'='*80}")
        
        # 创建PGD攻击
        pgd_attack = PGDAttack(model, config, norm=norm)
        
        # 创建贝塞尔优化器
        bezier = BezierAdversarialUnconstrained(model, config, norm=norm, 
                                               lr=0.01, num_iter=config.experiment_config['bezier_iterations'])
        
        norm_results = {}
        
        # Setting A: 单图像
        print(f"\n  Setting A (Simple Image):")
        samples_A = collect_samples_setting_A(images_by_class, model, pgd_attack, bezier, config, target_samples)
        norm_results['setting_A'] = samples_A
        
        # Setting B: 同类
        print(f"\n  Setting B (Same Classificaition):")
        samples_B = collect_samples_setting_B(images_by_class, model, pgd_attack, bezier, config, target_samples)
        norm_results['setting_B'] = samples_B
        
        # Setting C: 不同类
        print(f"\n  Setting C (Different Classificaition):")
        samples_C = collect_samples_setting_C(images_by_class, model, pgd_attack, bezier, config, target_samples)
        norm_results['setting_C'] = samples_C
        
        all_results[norm] = norm_results
        
        # 打印该范数的摘要
        print(f"\n  {norm.upper()} Abstract:")
        print(f"    Setting A: Collected {len(samples_A)} samples")
        print(f"    Setting B: Collected {len(samples_B)} samples")
        print(f"    Setting C: Collected {len(samples_C)} samples")
    
    return all_results

def save_results_fixed(results):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'bezier_basic_fixed_{timestamp}.json'
    
    # Convert to serializable format
    results_serializable = {}
    for norm in results:
        results_serializable[norm] = {}
        for setting in results[norm]:
            samples_serializable = []
            for sample in results[norm][setting]:
                sample_dict = {
                    'success_rate': float(sample['success_rate']),
                    'theta_norm': float(sample['theta_norm']),
                    'detailed_results': {
                        k: float(v) for k, v in sample['detailed_results'].items()
                    }
                }
                # Add indices information
                if 'image_idx' in sample:
                    sample_dict['image_idx'] = int(sample['image_idx'])
                if 'image_indices' in sample:
                    sample_dict['image_indices'] = [int(i) for i in sample['image_indices']]
                samples_serializable.append(sample_dict)
            
            results_serializable[norm][setting] = samples_serializable
    
    # Add configuration information
    results_with_config = {
        'results': results_serializable,
        'configuration': {
            'fixed_classes': FIXED_CLASSES,
            'target_samples': 25,
            'pgd_iterations': 40,
            'bezier_iterations': 30,
            'pgd_alpha_factors': {
                'linf': 4.0,
                'l2': 5.0,
                'l1': 10.0
            }
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results_with_config, f, indent=4)
    
    print(f"\nResults saved to {filename}")
    return filename

def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='Base Bézier-curve Experiment')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'imagenet'], help='Choosing Dataset')
    parser.add_argument('--model', type=str, default='resnet', 
                       choices=['resnet', 'vit'], help='Choosing Architecture')
    parser.add_argument('--mode', type=str, default='from_scratch',
                       choices=['pretrained', 'fine_tune', 'from_scratch'],
                       help='Choosing Training Mode')
    parser.add_argument('--device', type=str, default='auto',
                       help='Choosing Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建配置
    dataset = DatasetType.CIFAR10 if args.dataset == 'cifar10' else DatasetType.IMAGENET
    model = ModelType.RESNET if args.model == 'resnet' else ModelType.VIT
    mode = TrainingMode.PRETRAINED if args.mode == 'pretrained' else \
           TrainingMode.FINE_TUNE if args.mode == 'fine_tune' else TrainingMode.FROM_SCRATCH
    
    config = Config(dataset=dataset, model=model, training_mode=mode, device=args.device)
    
    print("Bézier adversarial curves - Basic experiment (fixed classes)")
    print("="*80)
    print("\nKey design (consistent with multi_image and comprehensive):")
    print("• Fixed classes for all settings:")
    print(f"  - Setting A: class {config.experiment_config['fixed_classes']['setting_A']} - single image")
    print(f"  - Setting B: class {config.experiment_config['fixed_classes']['setting_B']} - same-class pair")
    print(f"  - Setting C: class {config.experiment_config['fixed_classes']['setting_C']} - different-class pair")
    print("• Collect all available successful samples (target: 25 per setting)")
    print("• No separate test set (evaluate on the training paths)")
    print("• PGD attack: 40 iterations, using community-standard α")
    print("• Bézier optimization: 30 iterations, learning rate 0.01")
    print("="*80)

    
    # 检查模型文件是否存在
    model_filename = config.get_model_filename()
    if not os.path.exists(model_filename):
        print(f"\nError: Trained model not found!")
        print(f"Please run 'python train_model.py --dataset {args.dataset} --model {args.model} --mode {args.mode}' first")

        return
    
    # 运行实验
    results = run_basic_experiments_fixed(config)
    
    if results:
        # 打印结果
        print_results_fixed(results, config)
        
        # 保存结果
        results_file = save_results_fixed(results)
        
        print(f"\Experiment Complete!")
        print(f"The result will be saved in: {results_file}")
        
        print("\nConsistency with other experiments:")
        print("• Uses the same fixed classes as multi_image and comprehensive experiments")
        print("• No separate test set (basic experiment evaluates only on paths)")
        print("• Consistent PGD parameters (40 iterations, community α)")
        print("• Serves as a baseline for comparison with multi-image optimization")


if __name__ == "__main__":
    main()

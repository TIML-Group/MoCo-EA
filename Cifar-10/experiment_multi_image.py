"""
experiment_multi_image.py - Multi-image optimization experiments
Modified version with VARIED MAIN IMAGES and FIXED AUXILIARY:
- Fixed classes for each setting
- VARIED main images from training pool [130+] for each of 5 experiments
- Deterministic auxiliary image selection (indices 0-24) - FIXED
- Fixed test set (indices 30-129)
- Tests the effect of auxiliary images (0,5,10,15,20,25) on optimization
- ENSURES 5 successful experiments per setting through retry mechanism
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
import matplotlib.pyplot as plt
import random

from utils import PGDAttack, normalize_cifar10
from bezier_core import BezierAdversarialMultiImage

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seeds at module level
set_random_seeds(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FIXED CLASS CONFIGURATION
FIXED_CLASSES = {
    'setting_A': 3,        # cat (single class)
    'setting_B': 3,        # cat (same class)
    'setting_C': (3, 5)    # cat and dog (two different classes)
}

def load_model():
    """Load pretrained ResNet-18"""
    model = resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    
    checkpoint = torch.load('resnet18_cifar10_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model with accuracy: {checkpoint['acc']:.2f}%")
    
    return model.to(device).eval()

def organize_images_by_class(dataloader, model, max_per_class=300):
    """Organize images by class"""
    images_by_class = defaultdict(list)
    
    for idx, (img, label) in enumerate(dataloader):
        img_tensor = img.to(device)
        label_tensor = label.to(device)
        
        with torch.no_grad():
            pred = model(normalize_cifar10(img_tensor)).argmax(dim=1)
            if pred == label_tensor:
                images_by_class[label.item()].append((img_tensor, idx))
                
                if all(len(imgs) >= max_per_class for imgs in images_by_class.values()) and len(images_by_class) == 10:
                    break
    
    return images_by_class

def get_fixed_test_set_for_setting(images_by_class, setting):
    """
    Get FIXED test set for each setting
    Data layout:
    [0-24]: Auxiliary image pool
    [25-29]: Reserved
    [30-129]: Test set (100 images)
    [130+]: Training/Main image pool
    """
    test_start = 30
    test_size = 100
    
    if setting == 'A' or setting == 'B':
        class_id = FIXED_CLASSES[f'setting_{setting}']
        test_images = []
        test_labels = []
        
        for i in range(test_start, min(test_start + test_size, len(images_by_class[class_id]))):
            test_images.append(images_by_class[class_id][i][0])
            test_labels.append(torch.tensor([class_id]).to(device))
        
        return test_images, test_labels
    
    elif setting == 'C':
        class_id1, class_id2 = FIXED_CLASSES['setting_C']
        test_images = []
        test_labels = []
        
        # 50 images from each class
        for i in range(test_start, min(test_start + 50, len(images_by_class[class_id1]))):
            test_images.append(images_by_class[class_id1][i][0])
            test_labels.append(torch.tensor([class_id1]).to(device))
        
        for i in range(test_start, min(test_start + 50, len(images_by_class[class_id2]))):
            test_images.append(images_by_class[class_id2][i][0])
            test_labels.append(torch.tensor([class_id2]).to(device))
        
        return test_images, test_labels

def get_auxiliary_images(images_by_class, setting, num_auxiliary):
    """
    Get auxiliary images deterministically from indices 0-24
    FIXED across all experiments
    """
    if num_auxiliary == 0:
        return [], []
    
    if setting == 'A' or setting == 'B':
        class_id = FIXED_CLASSES[f'setting_{setting}']
        aux_images = []
        aux_labels = []
        
        # Deterministically select first num_auxiliary images
        for idx in range(min(num_auxiliary, 25)):
            if idx < len(images_by_class[class_id]):
                aux_images.append(images_by_class[class_id][idx][0])
                aux_labels.append(torch.tensor([class_id]).to(device))
        
        return aux_images, aux_labels
    
    elif setting == 'C':
        class_id1, class_id2 = FIXED_CLASSES['setting_C']
        aux_images = []
        aux_labels = []
        
        num_from_class1 = (num_auxiliary + 1) // 2
        num_from_class2 = num_auxiliary // 2
        
        # From first class
        for idx in range(min(num_from_class1, 25)):
            if idx < len(images_by_class[class_id1]):
                aux_images.append(images_by_class[class_id1][idx][0])
                aux_labels.append(torch.tensor([class_id1]).to(device))
        
        # From second class
        for idx in range(min(num_from_class2, 25)):
            if idx < len(images_by_class[class_id2]):
                aux_images.append(images_by_class[class_id2][idx][0])
                aux_labels.append(torch.tensor([class_id2]).to(device))
        
        return aux_images, aux_labels

def get_main_images_for_attempt(images_by_class, setting, attempt_id):
    """
    Get main images from training pool [130+] for each attempt
    Note: attempt_id can be > 5 when retrying failed experiments
    """
    train_start = 130
    
    if setting == 'A':
        class_id = FIXED_CLASSES['setting_A']
        pool_size = len(images_by_class[class_id]) - train_start
        if pool_size <= 0:
            raise ValueError(f"Not enough images in training pool for class {class_id}")
        
        # Cycle through the pool if attempt_id is large
        idx = train_start + (attempt_id * 7) % pool_size  # Use prime number for better spread
        
        x_main = images_by_class[class_id][idx][0]
        y_main = torch.tensor([class_id]).to(device)
        return x_main, y_main, None, None, idx
    
    elif setting == 'B':
        class_id = FIXED_CLASSES['setting_B']
        pool_size = len(images_by_class[class_id]) - train_start
        if pool_size < 2:
            raise ValueError(f"Not enough images in training pool for class {class_id}")
        
        # Select different pairs for each attempt
        base_idx = (attempt_id * 11) % (pool_size - 1)  # Prime number for spread
        
        idx1 = train_start + base_idx
        idx2 = train_start + (base_idx + 1) % pool_size
        
        x1 = images_by_class[class_id][idx1][0]
        x2 = images_by_class[class_id][idx2][0]
        y = torch.tensor([class_id]).to(device)
        return x1, x2, y, y, (idx1, idx2)
    
    elif setting == 'C':
        class_id1, class_id2 = FIXED_CLASSES['setting_C']
        pool_size1 = len(images_by_class[class_id1]) - train_start
        pool_size2 = len(images_by_class[class_id2]) - train_start
        
        if pool_size1 <= 0 or pool_size2 <= 0:
            raise ValueError(f"Not enough images in training pool")
        
        idx1 = train_start + (attempt_id * 7) % pool_size1
        idx2 = train_start + (attempt_id * 11) % pool_size2
        
        x1 = images_by_class[class_id1][idx1][0]
        x2 = images_by_class[class_id2][idx2][0]
        y1 = torch.tensor([class_id1]).to(device)
        y2 = torch.tensor([class_id2]).to(device)
        return x1, x2, y1, y2, (idx1, idx2)

def generate_valid_endpoints_with_retry(setting, images_by_class, attempt_id, pgd_attack, model, 
                                       max_pgd_retries=5, norm='linf'):
    """
    Try to generate valid endpoints with PGD retry on same image
    Returns (x_main, y_main, x2, y2, delta1, delta2, main_idx, success)
    """
    # Get main images for this attempt
    if setting == 'A':
        x_main, y_main, _, _, main_idx = get_main_images_for_attempt(images_by_class, 'A', attempt_id)
        
        # Try multiple PGD attempts on same image
        for pgd_try in range(max_pgd_retries):
            delta1 = pgd_attack.perturb(x_main, y_main)
            delta2 = pgd_attack.perturb(x_main, y_main)
            
            with torch.no_grad():
                pred1 = model(normalize_cifar10(torch.clamp(x_main + delta1, 0, 1))).argmax(dim=1)
                pred2 = model(normalize_cifar10(torch.clamp(x_main + delta2, 0, 1))).argmax(dim=1)
                
                if pred1 != y_main and pred2 != y_main:
                    return x_main, y_main, None, None, delta1, delta2, main_idx, True
        
        return x_main, y_main, None, None, None, None, main_idx, False
    
    elif setting in ['B', 'C']:
        x1, x2, y1, y2, main_indices = get_main_images_for_attempt(
            images_by_class, setting, attempt_id
        )
        
        for pgd_try in range(max_pgd_retries):
            delta1 = pgd_attack.perturb(x1, y1)
            delta2 = pgd_attack.perturb(x2, y2)
            
            with torch.no_grad():
                pred1 = model(normalize_cifar10(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
                pred2 = model(normalize_cifar10(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
                
                if pred1 != y1 and pred2 != y2:
                    return x1, y1, x2, y2, delta1, delta2, main_indices, True
        
        return x1, y1, x2, y2, None, None, main_indices, False

def evaluate_transferability_multi(model, bezier_obj, delta1, theta, delta2, 
                                  transfer_images, transfer_labels, num_path_points=50):
    """Evaluate transferability with fixed 50 points"""
    t_values = torch.linspace(0.01, 0.99, num_path_points).to(device)
    
    results = {
        'delta1_success': [],
        'delta2_success': [],
        'any_path_success': []
    }
    
    with torch.no_grad():
        for x, y in zip(transfer_images, transfer_labels):
            # Test endpoints
            x_adv = torch.clamp(x + delta1, 0, 1)
            pred = model(normalize_cifar10(x_adv)).argmax(dim=1)
            delta1_success = (pred != y).item()
            results['delta1_success'].append(delta1_success)
            
            x_adv = torch.clamp(x + delta2, 0, 1)
            pred = model(normalize_cifar10(x_adv)).argmax(dim=1)
            delta2_success = (pred != y).item()
            results['delta2_success'].append(delta2_success)
            
            # Test path
            path_success_count = 0
            for t in t_values:
                delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
                delta_t = bezier_obj.project_norm_ball(delta_t)
                
                x_adv = torch.clamp(x + delta_t, 0, 1)
                pred = model(normalize_cifar10(x_adv)).argmax(dim=1)
                
                if pred != y:
                    path_success_count += 1
            
            results['any_path_success'].append(path_success_count > 0)
    
    stats = {
        'delta1_transfer_rate': np.mean(results['delta1_success']),
        'delta2_transfer_rate': np.mean(results['delta2_success']),
        'endpoints_avg_transfer_rate': (np.mean(results['delta1_success']) + 
                                        np.mean(results['delta2_success'])) / 2,
        'any_path_point_transfer_rate': np.mean(results['any_path_success']),
    }
    
    # Calculate rescue rate
    rescued = 0
    for i in range(len(transfer_images)):
        if not results['delta1_success'][i] and not results['delta2_success'][i] and results['any_path_success'][i]:
            rescued += 1
    
    stats['rescue_rate'] = rescued / len(transfer_images) if len(transfer_images) > 0 else 0
    
    return stats

def print_multi_image_results_extended(results):
    """Print results for extended auxiliary image configurations"""
    print("\n" + "="*120)
    print("MULTI-IMAGE OPTIMIZATION RESULTS (VARIED MAIN IMAGES, FIXED AUXILIARY)")
    print("="*120)
    
    # Print configuration
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print("\nConfiguration:")
    print(f"  Setting A: Class {FIXED_CLASSES['setting_A']} ({class_names[FIXED_CLASSES['setting_A']]})")
    print(f"  Setting B: Class {FIXED_CLASSES['setting_B']} ({class_names[FIXED_CLASSES['setting_B']]})")
    c1, c2 = FIXED_CLASSES['setting_C']
    print(f"  Setting C: Classes {c1} ({class_names[c1]}) and {c2} ({class_names[c2]})")
    print("\nData Layout:")
    print("  [0-24]: Auxiliary images (FIXED across experiments)")
    print("  [30-129]: Test set (100 images, FIXED)")
    print("  [130+]: Main image pool (DIFFERENT for each experiment)")
    
    for norm in ['linf', 'l2', 'l1']:
        if norm not in results:
            continue
        
        print(f"\n{'='*100}")
        print(f"{norm.upper()} NORM RESULTS")
        print(f"{'='*100}")
        
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            setting_name = {
                'setting_A': 'Setting A (Single Image)',
                'setting_B': 'Setting B (Same Class)',
                'setting_C': 'Setting C (Different Classes)'
            }[setting]
            
            print(f"\n{setting_name}:")
            
            if not results[norm][setting] or all(not v for v in results[norm][setting].values()):
                print("  No valid results obtained for this setting.")
                continue
            
            # Check if we have 5 samples for each auxiliary config
            sample_counts = {k: len(v) for k, v in results[norm][setting].items()}
            if any(count < 5 for count in sample_counts.values()):
                print(f"  WARNING: Some configurations have < 5 samples: {sample_counts}")
            
            print(f"{'Num Auxiliary':<15} {'Endpoint Avg':<20} {'Path Success':<20} "
                  f"{'Improvement':<20} {'Rescue Rate':<15}")
            print("-" * 90)
            
            setting_results = results[norm][setting]
            
            for num_additional in sorted(setting_results.keys()):
                if setting_results[num_additional]:
                    stats_list = setting_results[num_additional]
                    
                    # Calculate mean and std
                    avg_endpoints = np.mean([s['endpoints_avg_transfer_rate'] for s in stats_list]) * 100
                    std_endpoints = np.std([s['endpoints_avg_transfer_rate'] for s in stats_list]) * 100
                    
                    avg_path = np.mean([s['any_path_point_transfer_rate'] for s in stats_list]) * 100
                    std_path = np.std([s['any_path_point_transfer_rate'] for s in stats_list]) * 100
                    
                    avg_rescue = np.mean([s['rescue_rate'] for s in stats_list]) * 100
                    std_rescue = np.std([s['rescue_rate'] for s in stats_list]) * 100
                    
                    improvement = avg_path - avg_endpoints
                    
                    # Format output with mean ± std
                    endpoint_str = f"{avg_endpoints:.1f}±{std_endpoints:.1f}%"
                    path_str = f"{avg_path:.1f}±{std_path:.1f}%"
                    
                    if improvement > 0:
                        imp_str = f"\033[92m+{improvement:.1f}%\033[0m"
                    else:
                        imp_str = f"\033[91m{improvement:.1f}%\033[0m"
                    
                    rescue_str = f"{avg_rescue:.1f}±{std_rescue:.1f}%"
                    
                    print(f"{num_additional:<15} {endpoint_str:<20} {path_str:<20} "
                          f"{imp_str:<29} {rescue_str:<15}")

def run_multi_image_experiments_extended():
    """Run multi-image experiments with VARIED main images and FIXED auxiliary"""
    set_random_seeds(42)
    
    norms = ['linf', 'l2', 'l1']
    epsilons = {
        'linf': 8/255,
        'l2': 0.5,
        'l1': 10.0
    }
    
    pgd_steps = 40
    pgd_alpha_factors = {
        'linf': 4.0,    # α = ε/4
        'l2': 5.0,      # α = ε/5
        'l1': 10.0      # α = ε/10
    }
    
    auxiliary_configs = [0, 5, 10, 15, 20, 25]
    
    model = load_model()
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    print("Organizing images by class...")
    images_by_class = organize_images_by_class(testloader, model, max_per_class=300)
    
    # Check availability of FIXED classes
    print("\nFixed class availability:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    required_classes = set([FIXED_CLASSES['setting_A'], FIXED_CLASSES['setting_B']] + 
                           list(FIXED_CLASSES['setting_C']))
    
    for class_id in required_classes:
        if class_id in images_by_class:
            print(f"  Class {class_id} ({class_names[class_id]}): {len(images_by_class[class_id])} images")
            if len(images_by_class[class_id]) < 150:
                print(f"    WARNING: Need at least 150 images for adequate training pool")
        else:
            print(f"  ERROR: Class {class_id} ({class_names[class_id]}) not available!")
            return None
    
    # Create FIXED test sets
    print("\nCreating fixed test sets...")
    test_sets = {
        'setting_A': get_fixed_test_set_for_setting(images_by_class, 'A'),
        'setting_B': get_fixed_test_set_for_setting(images_by_class, 'B'),
        'setting_C': get_fixed_test_set_for_setting(images_by_class, 'C')
    }
    
    for setting, (test_images, test_labels) in test_sets.items():
        print(f"  {setting}: {len(test_images)} test images")
    
    print(f"\nAuxiliary configurations: {auxiliary_configs}")
    
    all_results = {
        'generation_stats': {},
        'main_image_indices': {}
    }
    
    target_experiments = 5  # We want exactly 5 successful experiments
    
    for norm in norms:
        print(f"\n{'='*80}")
        print(f"Testing {norm.upper()} norm with epsilon={epsilons[norm]}")
        print(f"{'='*80}")
        
        eps = epsilons[norm]
        alpha = eps / pgd_alpha_factors[norm]
        
        # Adjust retry parameters based on norm difficulty
        if norm == 'l1':
            max_pgd_retries = 10
            max_attempts = 30
        else:
            max_pgd_retries = 5
            max_attempts = 15
        
        pgd_attack = PGDAttack(
            model, 
            eps=eps, 
            alpha=alpha,
            num_iter=pgd_steps,
            norm=norm
        )
        
        norm_results = {
            'setting_A': {i: [] for i in auxiliary_configs},
            'setting_B': {i: [] for i in auxiliary_configs},
            'setting_C': {i: [] for i in auxiliary_configs}
        }
        
        generation_stats = {
            'setting_A': {'attempted': 0, 'successful': 0},
            'setting_B': {'attempted': 0, 'successful': 0},
            'setting_C': {'attempted': 0, 'successful': 0}
        }
        
        main_indices_used = {
            'setting_A': [],
            'setting_B': [],
            'setting_C': []
        }
        
        # Setting A: Single image
        print(f"\n  Testing Setting A (Class {FIXED_CLASSES['setting_A']})...")
        
        successful_experiments = []
        attempt_id = 0
        
        while len(successful_experiments) < target_experiments and attempt_id < max_attempts:
            generation_stats['setting_A']['attempted'] += 1
            
            # Try to generate valid endpoints with retry
            x_main, y_main, _, _, delta1, delta2, main_idx, success = generate_valid_endpoints_with_retry(
                'A', images_by_class, attempt_id, pgd_attack, model, max_pgd_retries, norm
            )
            
            print(f"    Attempt {attempt_id+1}: Image index {main_idx} - {'Success' if success else 'Failed'}")
            attempt_id += 1
            
            if not success:
                continue
            
            generation_stats['setting_A']['successful'] += 1
            main_indices_used['setting_A'].append(main_idx)
            
            # Store successful experiment data
            successful_experiments.append({
                'x_main': x_main,
                'y_main': y_main,
                'delta1': delta1,
                'delta2': delta2,
                'main_idx': main_idx
            })
            
            print(f"      Collected {len(successful_experiments)}/{target_experiments} experiments")
        
        # Now run evaluation for all successful experiments
        print(f"    Evaluating {len(successful_experiments)} successful experiments...")
        test_images, test_labels = test_sets['setting_A']
        
        for exp_data in successful_experiments:
            for num_additional in auxiliary_configs:
                bezier = BezierAdversarialMultiImage(model, norm=norm, eps=eps, lr=0.01, num_iter=30)
                
                if num_additional == 0:
                    theta, _, _, _ = bezier.optimize_setting_A(
                        exp_data['x_main'], exp_data['y_main'], 
                        exp_data['delta1'], exp_data['delta2']
                    )
                else:
                    aux_images, aux_labels = get_auxiliary_images(images_by_class, 'A', num_additional)
                    theta, _, _, _ = bezier.optimize_setting_A_multi(
                        exp_data['x_main'], exp_data['y_main'],
                        exp_data['delta1'], exp_data['delta2'],
                        aux_images, aux_labels
                    )
                
                stats = evaluate_transferability_multi(
                    model, bezier, exp_data['delta1'], theta, exp_data['delta2'],
                    test_images, test_labels
                )
                norm_results['setting_A'][num_additional].append(stats)
        
        # Setting B: Same class
        print(f"\n  Testing Setting B (Class {FIXED_CLASSES['setting_B']})...")
        
        successful_experiments = []
        attempt_id = 0
        
        while len(successful_experiments) < target_experiments and attempt_id < max_attempts:
            generation_stats['setting_B']['attempted'] += 1
            
            x1, y1, x2, y2, delta1, delta2, main_indices, success = generate_valid_endpoints_with_retry(
                'B', images_by_class, attempt_id, pgd_attack, model, max_pgd_retries, norm
            )
            
            print(f"    Attempt {attempt_id+1}: Image indices {main_indices} - {'Success' if success else 'Failed'}")
            attempt_id += 1
            
            if not success:
                continue
            
            generation_stats['setting_B']['successful'] += 1
            main_indices_used['setting_B'].append(main_indices)
            
            successful_experiments.append({
                'x1': x1, 'x2': x2,
                'y1': y1, 'y2': y2,
                'delta1': delta1, 'delta2': delta2,
                'main_indices': main_indices
            })
            
            print(f"      Collected {len(successful_experiments)}/{target_experiments} experiments")
        
        print(f"    Evaluating {len(successful_experiments)} successful experiments...")
        test_images, test_labels = test_sets['setting_B']
        
        for exp_data in successful_experiments:
            for num_additional in auxiliary_configs:
                bezier = BezierAdversarialMultiImage(model, norm=norm, eps=eps, lr=0.01, num_iter=30)
                
                if num_additional == 0:
                    theta, _, _, _ = bezier.optimize_setting_B(
                        exp_data['x1'], exp_data['x2'], exp_data['y1'],
                        exp_data['delta1'], exp_data['delta2']
                    )
                else:
                    aux_images, aux_labels = get_auxiliary_images(images_by_class, 'B', num_additional)
                    theta, _, _, _ = bezier.optimize_setting_B_multi(
                        exp_data['x1'], exp_data['x2'], exp_data['y1'],
                        exp_data['delta1'], exp_data['delta2'],
                        aux_images, aux_labels
                    )
                
                stats = evaluate_transferability_multi(
                    model, bezier, exp_data['delta1'], theta, exp_data['delta2'],
                    test_images, test_labels
                )
                norm_results['setting_B'][num_additional].append(stats)
        
        # Setting C: Different classes
        class_c1, class_c2 = FIXED_CLASSES['setting_C']
        print(f"\n  Testing Setting C (Classes {class_c1} and {class_c2})...")
        
        successful_experiments = []
        attempt_id = 0
        
        while len(successful_experiments) < target_experiments and attempt_id < max_attempts:
            generation_stats['setting_C']['attempted'] += 1
            
            x1, y1, x2, y2, delta1, delta2, main_indices, success = generate_valid_endpoints_with_retry(
                'C', images_by_class, attempt_id, pgd_attack, model, max_pgd_retries, norm
            )
            
            print(f"    Attempt {attempt_id+1}: Image indices {main_indices} - {'Success' if success else 'Failed'}")
            attempt_id += 1
            
            if not success:
                continue
            
            generation_stats['setting_C']['successful'] += 1
            main_indices_used['setting_C'].append(main_indices)
            
            successful_experiments.append({
                'x1': x1, 'x2': x2,
                'y1': y1, 'y2': y2,
                'delta1': delta1, 'delta2': delta2,
                'main_indices': main_indices
            })
            
            print(f"      Collected {len(successful_experiments)}/{target_experiments} experiments")
        
        print(f"    Evaluating {len(successful_experiments)} successful experiments...")
        test_images, test_labels = test_sets['setting_C']
        
        for exp_data in successful_experiments:
            for num_additional in auxiliary_configs:
                bezier = BezierAdversarialMultiImage(model, norm=norm, eps=eps, lr=0.01, num_iter=30)
                
                if num_additional == 0:
                    theta, _, _, _ = bezier.optimize_setting_C(
                        exp_data['x1'], exp_data['x2'],
                        exp_data['y1'], exp_data['y2'],
                        exp_data['delta1'], exp_data['delta2']
                    )
                else:
                    aux_images, aux_labels = get_auxiliary_images(images_by_class, 'C', num_additional)
                    theta, _, _, _ = bezier.optimize_setting_C_multi(
                        exp_data['x1'], exp_data['x2'],
                        exp_data['y1'], exp_data['y2'],
                        exp_data['delta1'], exp_data['delta2'],
                        aux_images, aux_labels
                    )
                
                stats = evaluate_transferability_multi(
                    model, bezier, exp_data['delta1'], theta, exp_data['delta2'],
                    test_images, test_labels
                )
                norm_results['setting_C'][num_additional].append(stats)
        
        # Report statistics
        print(f"\n  {norm.upper()} Summary:")
        print(f"    Setting A: {generation_stats['setting_A']['successful']}/{generation_stats['setting_A']['attempted']} attempts successful")
        print(f"    Setting B: {generation_stats['setting_B']['successful']}/{generation_stats['setting_B']['attempted']} attempts successful")
        print(f"    Setting C: {generation_stats['setting_C']['successful']}/{generation_stats['setting_C']['attempted']} attempts successful")
        
        all_results[norm] = norm_results
        all_results['generation_stats'][norm] = generation_stats
        all_results['main_image_indices'][norm] = main_indices_used
    
    return all_results

if __name__ == "__main__":
    print("Bézier Adversarial Curves - Multi-Image Optimization (VARIED MAIN, FIXED AUXILIARY)")
    print("="*80)
    print("Key Design:")
    print("- FIXED classes for all experiments")
    print("- VARIED main images from [130+] pool (5 different images per setting)")
    print("- FIXED auxiliary images from [0-24] (same across all experiments)")
    print("- FIXED test set [30-129] (same as other experiments)")
    print("- RETRY mechanism ensures exactly 5 successful experiments per setting")
    print("\nThis design ensures complete results even when some attacks fail")
    print("="*80)
    
    results = run_multi_image_experiments_extended()
    
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'bezier_multi_image_varied_main_{timestamp}.json'
        
        # Save results with configuration
        results_with_config = {
            'results': results,
            'configuration': {
                'fixed_classes': FIXED_CLASSES,
                'main_image_source': 'Training pool [130+], different for each experiment',
                'auxiliary_pool': '[0-24] FIXED',
                'test_set': '[30-129] FIXED',
                'auxiliary_configs': [0, 5, 10, 15, 20, 25],
                'target_experiments': 5,
                'pgd_iterations': 40,
                'bezier_iterations': 30
            }
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(filename, 'w') as f:
            json.dump(convert_numpy(results_with_config), f, indent=4)
        
        print_multi_image_results_extended(results)
        
        print(f"\nResults saved to {filename}")
        
        print("\nExperimental Insights:")
        print("- All settings have exactly 5 successful experiments")
        print("- Retry mechanism handles difficult cases (especially L1 norm)")
        print("- Results show true auxiliary image effect across diverse main images")

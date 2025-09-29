"""
experiment_basic.py - Basic Bézier curve experiments
Modified version with FIXED CLASSES to align with other experiments:
- Uses fixed classes for each setting (same as multi_image and comprehensive)
- Collects all available successful samples (no 5 experiments)
- No separate test set (evaluates on training path only)
- Maintains consistency with the experimental framework
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

from utils import PGDAttack, normalize_cifar10
from bezier_core import BezierAdversarialUnconstrained

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

# FIXED CLASS CONFIGURATION (same as other experiments)
FIXED_CLASSES = {
    'setting_A': 3,        # cat (single class)
    'setting_B': 3,        # cat (same class)
    'setting_C': (3, 5)    # cat and dog (two different classes)
}

def load_model():
    """Load pretrained ResNet-18 model"""
    model = resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    
    if os.path.exists('resnet18_cifar10_best.pth'):
        checkpoint = torch.load('resnet18_cifar10_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model with accuracy: {checkpoint['acc']:.2f}%")
    else:
        print("ERROR: No pretrained model found! Please run train_model.py first.")
        exit(1)
    
    return model.to(device).eval()

def organize_images_by_class(dataloader, model, max_per_class=200):
    """Organize images by class"""
    images_by_class = defaultdict(list)
    
    for idx, (img, label) in enumerate(dataloader):
        img_tensor = img.to(device)
        label_tensor = label.to(device)
        
        with torch.no_grad():
            pred = model(normalize_cifar10(img_tensor)).argmax(dim=1)
            if pred == label_tensor:
                images_by_class[label.item()].append((img_tensor, idx))
                
                # Need at least max_per_class images per class
                if all(len(imgs) >= max_per_class for imgs in images_by_class.values()) and len(images_by_class) == 10:
                    break
    
    return images_by_class

def evaluate_bezier_path(model, bezier_obj, delta1, theta, delta2, x1, x2, y1, y2, 
                        setting_type='A', num_points=50):
    """Evaluate Bézier path at multiple points (excluding endpoints)"""
    # Exclude endpoints to avoid evaluation bias
    t_values = torch.linspace(0.02, 0.98, num_points).to(device)
    
    if setting_type == 'A':
        # Setting A: Single image - both x1 and x2 are the same
        x2 = x1
        y2 = y1
    
    # Track success for each image
    success_x1 = 0
    success_x2 = 0
    success_both = 0
    
    with torch.no_grad():
        for t in t_values:
            delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
            delta_t = bezier_obj.project_norm_ball(delta_t)
            
            # Test on first image
            x1_adv = torch.clamp(x1 + delta_t, 0, 1)
            outputs1 = model(normalize_cifar10(x1_adv))
            pred1 = outputs1.argmax(dim=1).item()
            s1 = pred1 != y1.item()
            
            if setting_type == 'A':
                # For Setting A, only one image to test
                if s1:
                    success_x1 += 1
                    success_x2 += 1  # Same as x1 for Setting A
                    success_both += 1
            else:
                # Test on second image for Settings B and C
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                outputs2 = model(normalize_cifar10(x2_adv))
                pred2 = outputs2.argmax(dim=1).item()
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

def collect_samples_setting_A(images_by_class, model, pgd_attack, bezier, norm, target_samples=25):
    """Collect samples for Setting A (single image)"""
    class_id = FIXED_CLASSES['setting_A']
    
    if class_id not in images_by_class:
        print(f"    ERROR: Class {class_id} not available")
        return []
    
    available_images = images_by_class[class_id]
    print(f"    Setting A: Using class {class_id} with {len(available_images)} available images")
    
    samples = []
    attempts = 0
    max_attempts = min(len(available_images) * 10, 500)  # Reasonable limit
    
    pbar = tqdm(total=target_samples, desc=f"    Collecting Setting A samples")
    
    while len(samples) < target_samples and attempts < max_attempts:
        # Select an image
        img_idx = attempts % len(available_images)
        x = available_images[img_idx][0]
        y = torch.tensor([class_id]).to(device)
        
        attempts += 1
        
        # Generate two perturbations for the same image
        delta1 = pgd_attack.perturb(x, y)
        delta2 = pgd_attack.perturb(x, y)
        
        # Verify both endpoints work
        with torch.no_grad():
            x_adv_d1 = torch.clamp(x + delta1, 0, 1)
            x_adv_d2 = torch.clamp(x + delta2, 0, 1)
            pred_d1 = model(normalize_cifar10(x_adv_d1)).argmax(dim=1)
            pred_d2 = model(normalize_cifar10(x_adv_d2)).argmax(dim=1)
            
            if pred_d1 == y or pred_d2 == y:
                continue
        
        # Optimize Bézier path
        theta, _, _, theta_norms = bezier.optimize_setting_A(x, y, delta1, delta2)
        
        # Evaluate path
        eval_results = evaluate_bezier_path(
            model, bezier, delta1, theta, delta2, 
            x, x, y, y, setting_type='A'
        )
        
        samples.append({
            'success_rate': eval_results['success_rate_avg'],
            'detailed_results': eval_results,
            'theta_norm': theta_norms[-1],
            'image_idx': img_idx
        })
        
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting A (attempts: {attempts})")
    
    return samples

def collect_samples_setting_B(images_by_class, model, pgd_attack, bezier, norm, target_samples=25):
    """Collect samples for Setting B (same class)"""
    class_id = FIXED_CLASSES['setting_B']
    
    if class_id not in images_by_class:
        print(f"    ERROR: Class {class_id} not available")
        return []
    
    available_images = images_by_class[class_id]
    print(f"    Setting B: Using class {class_id} with {len(available_images)} available images")
    
    if len(available_images) < 2:
        print(f"    ERROR: Need at least 2 images for Setting B")
        return []
    
    samples = []
    attempts = 0
    max_attempts = min(len(available_images) * len(available_images), 500)
    
    pbar = tqdm(total=target_samples, desc=f"    Collecting Setting B samples")
    
    while len(samples) < target_samples and attempts < max_attempts:
        # Select two different images from the same class
        idx1 = attempts % len(available_images)
        idx2 = (attempts + 1 + (attempts // len(available_images))) % len(available_images)
        
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(available_images)
        
        x1 = available_images[idx1][0]
        x2 = available_images[idx2][0]
        y = torch.tensor([class_id]).to(device)
        
        attempts += 1
        
        # Generate perturbations for each image
        delta1 = pgd_attack.perturb(x1, y)
        delta2 = pgd_attack.perturb(x2, y)
        
        # Verify endpoints work
        with torch.no_grad():
            pred1 = model(normalize_cifar10(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
            pred2 = model(normalize_cifar10(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
            
            if pred1 == y or pred2 == y:
                continue
        
        # Optimize Bézier path
        theta, _, _, theta_norms = bezier.optimize_setting_B(x1, x2, y, delta1, delta2)
        
        # Evaluate path
        eval_results = evaluate_bezier_path(
            model, bezier, delta1, theta, delta2, 
            x1, x2, y, y, setting_type='B'
        )
        
        samples.append({
            'success_rate': eval_results['success_rate_both'],
            'detailed_results': eval_results,
            'theta_norm': theta_norms[-1],
            'image_indices': (idx1, idx2)
        })
        
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting B (attempts: {attempts})")
    
    return samples

def collect_samples_setting_C(images_by_class, model, pgd_attack, bezier, norm, target_samples=25):
    """Collect samples for Setting C (different classes)"""
    class_id1, class_id2 = FIXED_CLASSES['setting_C']
    
    if class_id1 not in images_by_class or class_id2 not in images_by_class:
        print(f"    ERROR: Classes {class_id1} or {class_id2} not available")
        return []
    
    available_images1 = images_by_class[class_id1]
    available_images2 = images_by_class[class_id2]
    print(f"    Setting C: Using classes {class_id1} ({len(available_images1)} images) "
          f"and {class_id2} ({len(available_images2)} images)")
    
    samples = []
    attempts = 0
    max_attempts = min(len(available_images1) * len(available_images2), 500)
    
    pbar = tqdm(total=target_samples, desc=f"    Collecting Setting C samples")
    
    while len(samples) < target_samples and attempts < max_attempts:
        # Select one image from each class
        idx1 = attempts % len(available_images1)
        idx2 = attempts % len(available_images2)
        
        x1 = available_images1[idx1][0]
        x2 = available_images2[idx2][0]
        y1 = torch.tensor([class_id1]).to(device)
        y2 = torch.tensor([class_id2]).to(device)
        
        attempts += 1
        
        # Generate perturbations for each image
        delta1 = pgd_attack.perturb(x1, y1)
        delta2 = pgd_attack.perturb(x2, y2)
        
        # Verify endpoints work
        with torch.no_grad():
            pred1 = model(normalize_cifar10(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
            pred2 = model(normalize_cifar10(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
            
            if pred1 == y1 or pred2 == y2:
                continue
        
        # Optimize Bézier path
        theta, _, _, theta_norms = bezier.optimize_setting_C(x1, x2, y1, y2, delta1, delta2)
        
        # Evaluate path
        eval_results = evaluate_bezier_path(
            model, bezier, delta1, theta, delta2, 
            x1, x2, y1, y2, setting_type='C'
        )
        
        samples.append({
            'success_rate': eval_results['success_rate_both'],
            'detailed_results': eval_results,
            'theta_norm': theta_norms[-1],
            'image_indices': (idx1, idx2)
        })
        
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting C (attempts: {attempts})")
    
    return samples

def print_results_fixed(results):
    """Print results in fixed class format"""
    print("\n" + "="*120)
    print("BASIC EXPERIMENTS - FIXED CLASSES RESULTS")
    print("="*120)
    
    # Print configuration
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print("\nFixed Configuration:")
    print(f"  Setting A: Class {FIXED_CLASSES['setting_A']} ({class_names[FIXED_CLASSES['setting_A']]})")
    print(f"  Setting B: Class {FIXED_CLASSES['setting_B']} ({class_names[FIXED_CLASSES['setting_B']]})")
    c1, c2 = FIXED_CLASSES['setting_C']
    print(f"  Setting C: Classes {c1} ({class_names[c1]}) and {c2} ({class_names[c2]})")
    
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

def run_basic_experiments_fixed():
    """Run basic Bézier curve experiments with fixed classes"""
    model = load_model()
    
    norms = ['linf', 'l2', 'l1']
    epsilons = {
        'linf': 8/255,
        'l2': 0.5,
        'l1': 10.0
    }
    
    pgd_steps = 40
    
    # Community standard alpha factors for 40-step PGD
    pgd_alpha_factors = {
        'linf': 4.0,    # α = ε/4 (community standard for 40 steps)
        'l2': 5.0,      # α = ε/5 (moderate attack)
        'l1': 10.0      # α = ε/10 (stable optimization)
    }
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    print("\nOrganizing images by class...")
    images_by_class = organize_images_by_class(testloader, model, max_per_class=200)
    
    # Check availability of fixed classes
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nFixed class availability:")
    required_classes = set([FIXED_CLASSES['setting_A'], FIXED_CLASSES['setting_B']] + 
                           list(FIXED_CLASSES['setting_C']))
    
    for class_id in required_classes:
        if class_id in images_by_class:
            print(f"  Class {class_id} ({class_names[class_id]}): {len(images_by_class[class_id])} images")
        else:
            print(f"  ERROR: Class {class_id} ({class_names[class_id]}) not available!")
            return None
    
    target_samples = 25
    print(f"\nTarget: {target_samples} samples per setting per norm")
    print(f"PGD attack iterations: {pgd_steps} (with community standard α)")
    print(f"Bézier optimization: 30 iterations with lr=0.01")
    print("="*80)
    
    all_results = {}
    
    for norm in norms:
        print(f"\n{'='*80}")
        print(f"Testing {norm.upper()} norm (ε={epsilons[norm]})")
        print(f"{'='*80}")
        
        eps = epsilons[norm]
        alpha = eps / pgd_alpha_factors[norm]
        
        pgd_attack = PGDAttack(model, eps=eps, alpha=alpha, 
                              num_iter=pgd_steps, norm=norm)
        
        bezier = BezierAdversarialUnconstrained(model, norm=norm, eps=eps, 
                                               lr=0.01, num_iter=30)
        
        norm_results = {}
        
        # Setting A: Single Image
        print(f"\n  Setting A (Single Image):")
        samples_A = collect_samples_setting_A(images_by_class, model, pgd_attack, bezier, norm, target_samples)
        norm_results['setting_A'] = samples_A
        
        # Setting B: Same Class
        print(f"\n  Setting B (Same Class):")
        samples_B = collect_samples_setting_B(images_by_class, model, pgd_attack, bezier, norm, target_samples)
        norm_results['setting_B'] = samples_B
        
        # Setting C: Different Classes
        print(f"\n  Setting C (Different Classes):")
        samples_C = collect_samples_setting_C(images_by_class, model, pgd_attack, bezier, norm, target_samples)
        norm_results['setting_C'] = samples_C
        
        all_results[norm] = norm_results
        
        # Print summary for this norm
        print(f"\n  {norm.upper()} Summary:")
        print(f"    Setting A: {len(samples_A)} samples collected")
        print(f"    Setting B: {len(samples_B)} samples collected")
        print(f"    Setting C: {len(samples_C)} samples collected")
    
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

if __name__ == "__main__":
    print("Bézier Adversarial Curves - Basic Experiments (FIXED CLASSES)")
    print("="*80)
    print("\nKey Design (aligned with multi_image and comprehensive):")
    print("• FIXED classes for all settings:")
    print("  - Setting A: Class 3 (cat) - single image")
    print("  - Setting B: Class 3 (cat) - same class pairs")
    print("  - Setting C: Classes 3 & 5 (cat & dog) - different classes")
    print("• Collect all available successful samples (target: 25 per setting)")
    print("• No separate test set (evaluates on training path)")
    print("• PGD attack with 40 iterations and community standard α")
    print("• Bézier optimization: 30 iterations with lr=0.01")
    print("="*80)
    
    if not os.path.exists('resnet18_cifar10_best.pth'):
        print("\nERROR: No trained model found!")
        print("Please run 'python train_model.py' first.")
        exit(1)
    
    # Run experiments
    results = run_basic_experiments_fixed()
    
    if results:
        # Print results
        print_results_fixed(results)
        
        # Save results
        results_file = save_results_fixed(results)
        
        print(f"\nExperiments completed!")
        print(f"Results saved to: {results_file}")
        
        print("\nAlignment with other experiments:")
        print("• Uses same fixed classes as multi_image and comprehensive")
        print("• No test set separation (basic experiment evaluates on path only)")
        print("• Consistent PGD parameters (40 iterations, community α)")
        print("• Provides baseline for comparison with multi-image optimization")

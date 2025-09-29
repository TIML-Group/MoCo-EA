"""
experiment_transferability.py - Test transferability on unseen images
Community standard version with:
- 25 different training samples/pairs (deterministic selection for reproducibility)
- Each sample generates its own PGD endpoints
- Fixed test set of 20 images for evaluation
- Tracks average successful points per image
- Aligned with experimental framework
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import random
import timm

from utils import PGDAttack, normalize_imagenet
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

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

valset = ImageFolder(
    root="./dataset/imagenet/val",  
    transform=transform_val
)

cat = "n02124075"   # Egyptian cat
dog = "n02099712"   # Labrador retriever

cat_id = valset.class_to_idx[cat]
dog_id = valset.class_to_idx[dog]

FIXED_CLASSES = {
    "setting_A": cat_id,
    "setting_B": cat_id,
    "setting_C": (cat_id, dog_id)
}

def load_model():
    """Load pretrained ViT model"""
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.eval()
    return model.to(device)

def organize_images_by_class(dataloader, model, max_per_class=50):
    """Organize images by class"""
    images_by_class = defaultdict(list)
    needed_classes = [cat_id, dog_id]
    
    for idx, (img, label) in enumerate(dataloader):
        img_tensor = img.to(device)
        label_tensor = label.to(device)

        if idx == 0:
            print("organize_images_by_class: raw batch min/max:", img.min().item(), img.max().item())
        
        with torch.no_grad():
            preds = model(normalize_imagenet(img_tensor)).argmax(dim=1)

        for j in range(img.size(0)):
            lbl = int(label[j].item())
            pred_lbl = int(preds[j].item())
            if pred_lbl == lbl:
                abs_idx = idx * dataloader.batch_size + j
                # store CPU tensor to save GPU memory
                images_by_class[lbl].append((img[j].cpu(), abs_idx))

        if all(len(images_by_class[c]) >= max_per_class for c in needed_classes):
            break

    return images_by_class

def get_fixed_test_set_for_setting(images_by_class, setting):
    """
    Data layout alignment:
    []: Reserved (auxiliary)
    [0-9]: Fixed test set (10 images)
    [10+]: Training pool
    """
    test_start = 0
    test_size = 10
    
    if setting == 'A' or setting == 'B':
        # Single class for settings A and B
        class_id = FIXED_CLASSES[f'setting_{setting}']
        test_images = []
        test_labels = []
        
        for i in range(test_start, min(test_start + test_size, len(images_by_class[class_id]))):
            test_images.append(images_by_class[class_id][i][0])
            test_labels.append(int(class_id)) 
        return test_images, test_labels
    
    elif setting == 'C':
        # Two classes for setting C
        class_id1, class_id2 = FIXED_CLASSES['setting_C']
        test_images = []
        test_labels = []
        
        # 10 images from each class
        for i in range(test_start, min(test_start + 10, len(images_by_class[class_id1]))):
            test_images.append(images_by_class[class_id1][i][0])
            test_labels.append(int(class_id1))
        
        for i in range(test_start, min(test_start + 10, len(images_by_class[class_id2]))):
            test_images.append(images_by_class[class_id2][i][0])
            test_labels.append(int(class_id2))
        
        return test_images, test_labels

def get_training_pool_for_setting(images_by_class, setting):
    """
    Get training pool for each setting
    Uses images from index 25 onwards (after test set)
    """
    train_start = 10
    
    if setting == 'A' or setting == 'B':
        class_id = FIXED_CLASSES[f'setting_{setting}']
        training_pool = []
        
        for i in range(train_start, len(images_by_class[class_id])):
            training_pool.append((images_by_class[class_id][i][0], int(class_id), images_by_class[class_id][i][1])) 
        
        return training_pool
    
    elif setting == 'C':
        class_id1, class_id2 = FIXED_CLASSES['setting_C']
        training_pool1 = []
        training_pool2 = []
        
        for i in range(train_start, len(images_by_class[class_id1])):
            training_pool1.append((images_by_class[class_id1][i][0], int(class_id1), images_by_class[class_id1][i][1]))
        
        for i in range(train_start, len(images_by_class[class_id2])):
            training_pool2.append((images_by_class[class_id2][i][0], int(class_id2), images_by_class[class_id2][i][1]))
        
        return training_pool1, training_pool2

def evaluate_transferability(model, bezier_obj, delta1, theta, delta2, 
                             transfer_images, transfer_labels, num_path_points=50):
    t_values = torch.linspace(0.01, 0.99, num_path_points).to(device)

    results = {
        'delta1_success': [],
        'delta2_success': [],
        'any_path_success': [],
        'successful_points_per_image': []
    }

    with torch.no_grad():
        for x_cpu, y in zip(transfer_images, transfer_labels):
            x = x_cpu.to(device)                     # [C,H,W]
            y_scalar = int(y)                       # store labels as int earlier

            # ensure delta shapes are compatible: squeeze first dim if present
            d1 = delta1.squeeze(0) if delta1.dim() == 4 else delta1
            d2 = delta2.squeeze(0) if delta2.dim() == 4 else delta2

            # Test delta1
            x_adv = torch.clamp(x + d1, 0.0, 1.0)
            pred = model(normalize_imagenet(x_adv.unsqueeze(0))).argmax(dim=1).item()
            delta1_success = int(pred != y_scalar)
            results['delta1_success'].append(delta1_success)

            # Test delta2
            x_adv = torch.clamp(x + d2, 0.0, 1.0)
            pred = model(normalize_imagenet(x_adv.unsqueeze(0))).argmax(dim=1).item()
            delta2_success = int(pred != y_scalar)
            results['delta2_success'].append(delta2_success)

            # Path
            path_success_count = 0
            for t in t_values:
                delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
                delta_t = bezier_obj.project_norm_ball(delta_t)
                # normalize delta_t shape
                delta_t_s = delta_t.squeeze(0) if delta_t.dim() == 4 else delta_t
                x_adv = torch.clamp(x + delta_t_s, 0.0, 1.0)
                pred = model(normalize_imagenet(x_adv.unsqueeze(0))).argmax(dim=1).item()
                if pred != y_scalar:
                    path_success_count += 1

            results['any_path_success'].append(int(path_success_count > 0))
            results['successful_points_per_image'].append(path_success_count)

    # statistics (same as before)
    stats = {
        'delta1_transfer_rate': np.mean(results['delta1_success']) if results['delta1_success'] else 0.0,
        'delta2_transfer_rate': np.mean(results['delta2_success']) if results['delta2_success'] else 0.0,
        'endpoints_avg_transfer_rate': (np.mean(results['delta1_success']) + np.mean(results['delta2_success'])) / 2 if results['delta1_success'] else 0.0,
        'any_path_point_transfer_rate': np.mean(results['any_path_success']) if results['any_path_success'] else 0.0,
        'avg_successful_points': np.mean(results['successful_points_per_image']) if results['successful_points_per_image'] else 0.0,
        'std_successful_points': np.std(results['successful_points_per_image']) if results['successful_points_per_image'] else 0.0,
    }

    # rescue rate
    rescued = 0
    for i in range(len(results['delta1_success'])):
        if not results['delta1_success'][i] and not results['delta2_success'][i] and results['any_path_success'][i]:
            rescued += 1
    stats['rescue_rate'] = rescued / len(results['delta1_success']) if len(results['delta1_success'])>0 else 0.0

    return stats

def collect_samples_setting_A(training_pool, model, pgd_attack, bezier, test_images, test_labels, 
                              target_samples=25):
    """Collect samples for Setting A using FIXED training samples for reproducibility"""
    samples = []
    
    pbar = tqdm(total=target_samples, desc="    Collecting Setting A samples")
    
    # Use FIXED, deterministic indices for reproducibility
    # Spread evenly across available training pool
    step = max(1, len(training_pool) // target_samples)
    
    sample_count = 0
    for i in range(min(target_samples * 2, len(training_pool))):  # Try more indices in case some fail
        if sample_count >= target_samples:
            break
            
        # FIXED index selection - same every run
        idx = (i * step) % len(training_pool)
        x_train_cpu, y_train_int, abs_idx = training_pool[idx]
        x_train = x_train_cpu.to(device)
        y_train = torch.tensor([y_train_int], device=device)  # shape [1]
        if x_train.dim() == 3:
            x_train = x_train.unsqueeze(0)
        
        # Try to generate valid endpoints for this fixed training sample
        max_pgd_attempts = 10
        success = False
        
        for pgd_attempt in range(max_pgd_attempts):
            # Generate two perturbations for this image
            delta1 = pgd_attack.perturb(x_train, y_train)
            delta2 = pgd_attack.perturb(x_train, y_train)
            
            # Verify endpoints work on training image
            with torch.no_grad():
                pred1 = model(normalize_imagenet(torch.clamp(x_train + delta1, 0, 1))).argmax(dim=1)
                pred2 = model(normalize_imagenet(torch.clamp(x_train + delta2, 0, 1))).argmax(dim=1)
                
                if pred1 != y_train and pred2 != y_train:
                    success = True
                    break

                # print("PGD attempt", pgd_attempt, "pred1:", pred1.item(), "pred2:", pred2.item())

        
        if not success:
            continue
        
        # Optimize Bézier path
        theta, _, _, _ = bezier.optimize_setting_A(x_train, y_train, delta1, delta2)
        
        # Evaluate transferability on test set
        stats = evaluate_transferability(
            model, bezier, delta1, theta, delta2,
            test_images, test_labels
        )
        
        samples.append({
            'training_idx': idx,
            'training_image_index': int(abs_idx),  # actual dataset index
            'stats': stats
        })
        
        sample_count += 1
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples using fixed training indices")
    
    return samples

def collect_samples_setting_B(training_pool, model, pgd_attack, bezier, test_images, test_labels,
                              target_samples=25):
    """Collect samples for Setting B using FIXED training pairs for reproducibility"""
    samples = []
    
    pbar = tqdm(total=target_samples, desc="    Collecting Setting B samples")
    
    # Generate FIXED pairs of indices for reproducibility
    # Use deterministic pairing: (0,1), (2,3), (4,5), etc.
    sample_count = 0
    for pair_idx in range(min(target_samples * 2, len(training_pool) // 2)):
        if sample_count >= target_samples:
            break
            
        # FIXED pair selection - same every run
        idx1 = (pair_idx * 2) % len(training_pool)
        idx2 = (pair_idx * 2 + 1) % len(training_pool)
        
        x1_cpu, y1_int, abs1 = training_pool[idx1]
        x2_cpu, y2_int, abs2 = training_pool[idx2]

        x1_train = x1_cpu.unsqueeze(0).to(device)
        x2_train = x2_cpu.unsqueeze(0).to(device)
        y_train = torch.tensor([y1_int], device=device)
      
        # Try to generate valid endpoints
        max_pgd_attempts = 10
        success = False
        
        for pgd_attempt in range(max_pgd_attempts):
            # Generate perturbations
            delta1 = pgd_attack.perturb(x1_train, y_train)
            delta2 = pgd_attack.perturb(x2_train, y_train)
            
            # Verify endpoints work
            with torch.no_grad():
                pred1 = model(normalize_imagenet(torch.clamp(x1_train + delta1, 0, 1))).argmax(dim=1)
                pred2 = model(normalize_imagenet(torch.clamp(x2_train + delta2, 0, 1))).argmax(dim=1)
                
                if pred1 != y_train and pred2 != y_train:
                    success = True
                    break
        
        if not success:
            continue
        
        # Optimize Bézier path
        theta, _, _, _ = bezier.optimize_setting_B(x1_train, x2_train, y_train, delta1, delta2)
        
        # Evaluate transferability
        stats = evaluate_transferability(
            model, bezier, delta1, theta, delta2,
            test_images, test_labels
        )
        
        samples.append({
            'training_indices': (idx1, idx2),
            'training_image_indices': (int(abs1), int(abs2)),
            'stats': stats
        })
        
        sample_count += 1
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples using fixed training pairs")
    
    return samples

def collect_samples_setting_C(training_pool1, training_pool2, model, pgd_attack, bezier, 
                              test_images, test_labels, target_samples=25):
    """Collect samples for Setting C using FIXED training pairs for reproducibility"""
    samples = []
    
    pbar = tqdm(total=target_samples, desc="    Collecting Setting C samples")
    
    # Generate FIXED cross-class pairs for reproducibility
    # Use deterministic pairing: pool1[0] with pool2[0], pool1[1] with pool2[1], etc.
    sample_count = 0
    max_pairs = min(target_samples * 2, len(training_pool1), len(training_pool2))
    
    for pair_idx in range(max_pairs):
        if sample_count >= target_samples:
            break
            
        # FIXED pair selection - same every run
        idx1 = pair_idx % len(training_pool1)
        idx2 = pair_idx % len(training_pool2)

        x1_cpu, y1_int, abs1 = training_pool1[idx1]
        x2_cpu, y2_int, abs2 = training_pool2[idx2]

        x1_train = x1_cpu.unsqueeze(0).to(device)
        x2_train = x2_cpu.unsqueeze(0).to(device)
        y1_train = torch.tensor([y1_int], device=device)
        y2_train = torch.tensor([y2_int], device=device)
      
        # Try to generate valid endpoints
        max_pgd_attempts = 10
        success = False
        
        for pgd_attempt in range(max_pgd_attempts):
            # Generate perturbations
            delta1 = pgd_attack.perturb(x1_train, y1_train)
            delta2 = pgd_attack.perturb(x2_train, y2_train)
            
            # Verify endpoints work
            with torch.no_grad():
                pred1 = model(normalize_imagenet(torch.clamp(x1_train + delta1, 0, 1))).argmax(dim=1)
                pred2 = model(normalize_imagenet(torch.clamp(x2_train + delta2, 0, 1))).argmax(dim=1)
                
                if pred1 != y1_train and pred2 != y2_train:
                    success = True
                    break
        
        if not success:
            continue
        
        # Optimize Bézier path
        theta, _, _, _ = bezier.optimize_setting_C(x1_train, x2_train, y1_train, y2_train, delta1, delta2)
        
        # Evaluate transferability
        stats = evaluate_transferability(
            model, bezier, delta1, theta, delta2,
            test_images, test_labels
        )
        
        samples.append({
            'training_indices': (idx1, idx2),
            'training_image_indices': (int(abs1), int(abs2)),
            'stats': stats
        })
        
        sample_count += 1
        pbar.update(1)
    
    pbar.close()
    print(f"    Collected {len(samples)} samples using fixed training pairs")
    
    return samples

def print_transferability_results(results):
    """Print transferability results in mean±std format"""
    print("\n" + "="*120)
    print("TRANSFERABILITY EXPERIMENT RESULTS (REPRODUCIBLE)")
    print("="*120)
    
    # Print configuration
    class_names = {
        cat_id: "egyptian cat",
        dog_id: "labrador retriever"
    }
    print(f"  Setting A: Class {FIXED_CLASSES['setting_A']} "
      f"({class_names[FIXED_CLASSES['setting_A']]})")
    print(f"  Setting B: Class {FIXED_CLASSES['setting_B']} "
      f"({class_names[FIXED_CLASSES['setting_B']]})")
    c1, c2 = FIXED_CLASSES['setting_C']
    print(f"  Setting C: Classes {c1} ({class_names[c1]}) and {c2} ({class_names[c2]})")
    print("  []:      Reserved (auxiliary pool for multi_image)")
    print("  [2-11]:  Fixed test set (10 images)")
    print("  [12+]:   Training pool")
        
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
            
            if not results[norm][setting]:
                print("  No samples collected")
                continue
            
            # Extract statistics from samples
            samples = results[norm][setting]
            delta1_rates = [s['stats']['delta1_transfer_rate'] for s in samples]
            delta2_rates = [s['stats']['delta2_transfer_rate'] for s in samples]
            endpoints_avg_rates = [s['stats']['endpoints_avg_transfer_rate'] for s in samples]
            path_rates = [s['stats']['any_path_point_transfer_rate'] for s in samples]
            rescue_rates = [s['stats']['rescue_rate'] for s in samples]
            avg_points_list = [s['stats']['avg_successful_points'] for s in samples]
            
            # Calculate mean and std
            avg_delta1 = np.mean(delta1_rates) * 100
            std_delta1 = np.std(delta1_rates) * 100
            
            avg_delta2 = np.mean(delta2_rates) * 100
            std_delta2 = np.std(delta2_rates) * 100
            
            avg_endpoints = np.mean(endpoints_avg_rates) * 100
            std_endpoints = np.std(endpoints_avg_rates) * 100
            
            avg_path = np.mean(path_rates) * 100
            std_path = np.std(path_rates) * 100
            
            avg_rescue = np.mean(rescue_rates) * 100
            std_rescue = np.std(rescue_rates) * 100
            
            avg_points = np.mean(avg_points_list)
            std_points = np.std(avg_points_list)
            
            improvement = avg_path - avg_endpoints
            
            print(f"  Number of samples:           {len(samples)}")
            print(f"  Endpoint δ₁ transfer:        {avg_delta1:>6.1f} ± {std_delta1:<5.1f}%")
            print(f"  Endpoint δ₂ transfer:        {avg_delta2:>6.1f} ± {std_delta2:<5.1f}%")
            print(f"  Endpoints average:           {avg_endpoints:>6.1f} ± {std_endpoints:<5.1f}%")
            print(f"  Any path point succeeds:     {avg_path:>6.1f} ± {std_path:<5.1f}%")
            print(f"  Avg successful points/image: {avg_points:>6.1f} ± {std_points:<5.1f} / 50")
            print(f"  Images rescued by path:      {avg_rescue:>6.1f} ± {std_rescue:<5.1f}%")
            
            if improvement > 0:
                print(f"  \033[92mImprovement over endpoints:  +{improvement:>5.1f}%\033[0m")
            else:
                print(f"  \033[91mImprovement over endpoints:  {improvement:>6.1f}%\033[0m")
    
    # Summary table with points metric
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(f"\n{'Setting':<30} {'Norm':<8} {'Samples':<10} {'Endpoint Avg':<20} {'Path Success':<20} {'Avg Points':<15} {'Improvement':<15}")
    print("-" * 118)
    
    for norm in ['linf', 'l2', 'l1']:
        if norm not in results:
            continue
        norm_symbol = {'linf': 'ℓ∞', 'l2': 'ℓ₂', 'l1': 'ℓ₁'}[norm]
        
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            setting_name = {
                'setting_A': 'Setting A (Single)',
                'setting_B': 'Setting B (Same Class)',
                'setting_C': 'Setting C (Diff Class)'
            }[setting]
            
            if results[norm][setting]:
                samples = results[norm][setting]
                
                endpoints_avg_rates = [s['stats']['endpoints_avg_transfer_rate'] for s in samples]
                path_rates = [s['stats']['any_path_point_transfer_rate'] for s in samples]
                avg_points_list = [s['stats']['avg_successful_points'] for s in samples]
                
                avg_endpoints = np.mean(endpoints_avg_rates) * 100
                std_endpoints = np.std(endpoints_avg_rates) * 100
                
                avg_path = np.mean(path_rates) * 100
                std_path = np.std(path_rates) * 100
                
                avg_points = np.mean(avg_points_list)
                std_points = np.std(avg_points_list)
                
                improvement = avg_path - avg_endpoints
                
                endpoints_str = f"{avg_endpoints:.1f}±{std_endpoints:.1f}%"
                path_str = f"{avg_path:.1f}±{std_path:.1f}%"
                points_str = f"{avg_points:.1f}±{std_points:.1f}"
                
                if improvement > 0:
                    imp_str = f"\033[92m+{improvement:.1f}%\033[0m"
                else:
                    imp_str = f"\033[91m{improvement:.1f}%\033[0m"
                
                print(f"{setting_name:<30} {norm_symbol:<8} {len(samples):<10} "
                      f"{endpoints_str:<20} {path_str:<20} {points_str:<15} {imp_str}")

def run_transferability_experiments():
    """Run transferability experiments using reproducible approach"""
    set_random_seeds(42)
    
    norms = ['linf', 'l2', 'l1']
    epsilons = {
        'linf': 8/255, # 4/255
        'l2': 4.0, #2.0  
        'l1': 300.0 #75.0 
    }
    
    pgd_steps = 40
    pgd_alpha_factors = {
        'linf': 4.0,    # α = ε/4 (community standard)
        'l2': 5.0,      # α = ε/5
        'l1': 10.0      # α = ε/10
    }
    
    model = load_model()
    
    needed_classes = [cat_id, dog_id]
    indices = [i for i, y in enumerate(valset.targets) if y in needed_classes]
    val_subset = torch.utils.data.Subset(valset, indices)

    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    for inputs, _ in valloader:
        print("raw batch range:", inputs.min().item(), inputs.max().item())
        break

    print("\nOrganizing images by class...")
    images_by_class = organize_images_by_class(valloader, model, max_per_class=50)
    
    # Check class availability
    print("\nClass availability:")
    class_names = {
        cat_id: "egyptian cat",
        dog_id: "labrador retriever"
    }
    
    required_classes = set([FIXED_CLASSES['setting_A'], FIXED_CLASSES['setting_B']] + 
                           list(FIXED_CLASSES['setting_C']))
    
    for class_id in required_classes:
        if class_id in images_by_class:
            print(f"  Class {class_id} ({class_names[class_id]}): {len(images_by_class[class_id])} images")
        else:
            print(f"  ERROR: Class {class_id} ({class_names[class_id]}) not available!")
            return None
    
    # Create fixed test sets
    print("\nCreating fixed test sets...")
    test_sets = {
        'setting_A': get_fixed_test_set_for_setting(images_by_class, 'A'),
        'setting_B': get_fixed_test_set_for_setting(images_by_class, 'B'),
        'setting_C': get_fixed_test_set_for_setting(images_by_class, 'C')
    }
    
    for setting, (test_images, test_labels) in test_sets.items():
        print(f"  {setting}: {len(test_images)} test images")
    
    # Create training pools
    print("\nCreating training pools...")
    training_pools = {}
    training_pools['setting_A'] = get_training_pool_for_setting(images_by_class, 'A')
    training_pools['setting_B'] = get_training_pool_for_setting(images_by_class, 'B')
    training_pools['setting_C'] = get_training_pool_for_setting(images_by_class, 'C')
    
    print(f"  Setting A: {len(training_pools['setting_A'])} training images")
    print(f"  Setting B: {len(training_pools['setting_B'])} training images")
    pool1, pool2 = training_pools['setting_C']
    print(f"  Setting C: {len(pool1)} + {len(pool2)} training images")
    
    target_samples = 25
    print(f"\nTarget: {target_samples} samples per setting per norm")
    print(f"PGD: {pgd_steps} iterations with community standard α")
    print(f"Bézier: 30 iterations with lr=0.01")
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
        
        # Setting A
        print(f"\n  Setting A (Single Image, Class {FIXED_CLASSES['setting_A']}):")
        test_images_A, test_labels_A = test_sets['setting_A']
        samples_A = collect_samples_setting_A(
            training_pools['setting_A'], model, pgd_attack, bezier,
            test_images_A, test_labels_A, target_samples
        )
        norm_results['setting_A'] = samples_A
        
        # Setting B
        print(f"\n  Setting B (Same Class, Class {FIXED_CLASSES['setting_B']}):")
        test_images_B, test_labels_B = test_sets['setting_B']
        samples_B = collect_samples_setting_B(
            training_pools['setting_B'], model, pgd_attack, bezier,
            test_images_B, test_labels_B, target_samples
        )
        norm_results['setting_B'] = samples_B
        
        # Setting C
        c1, c2 = FIXED_CLASSES['setting_C']
        print(f"\n  Setting C (Different Classes, {c1} and {c2}):")
        test_images_C, test_labels_C = test_sets['setting_C']
        pool1, pool2 = training_pools['setting_C']
        samples_C = collect_samples_setting_C(
            pool1, pool2, model, pgd_attack, bezier,
            test_images_C, test_labels_C, target_samples
        )
        norm_results['setting_C'] = samples_C
        
        all_results[norm] = norm_results
        
        # Print summary
        print(f"\n  {norm.upper()} Summary:")
        print(f"    Setting A: {len(samples_A)} samples collected")
        print(f"    Setting B: {len(samples_B)} samples collected")
        print(f"    Setting C: {len(samples_C)} samples collected")
    
    return all_results

def save_results(results):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'bezier_transferability_reproducible_{timestamp}.json'
    
    # Convert to serializable format
    results_serializable = {}
    for norm in results:
        results_serializable[norm] = {}
        for setting in results[norm]:
            samples_serializable = []
            for sample in results[norm][setting]:
                sample_dict = {
                    'stats': {k: float(v) for k, v in sample['stats'].items()}
                }
                if 'training_idx' in sample:
                    sample_dict['training_idx'] = int(sample['training_idx'])
                if 'training_indices' in sample:
                    sample_dict['training_indices'] = [int(i) for i in sample['training_indices']]
                if 'training_image_index' in sample:
                    sample_dict['training_image_index'] = int(sample['training_image_index'])
                if 'training_image_indices' in sample:
                    sample_dict['training_image_indices'] = [int(i) for i in sample['training_image_indices']]
                samples_serializable.append(sample_dict)
            
            results_serializable[norm][setting] = samples_serializable
    
    # Add configuration
    results_with_config = {
        'results': results_serializable,
        'configuration': {
            'fixed_classes': FIXED_CLASSES,
            'data_layout': {
                'auxiliary_pool': '[]',
                'test_set': '[0-9]',
                'training_pool': '[10+]'
            },
            'target_samples': 25,
            'test_set_size': 10,
            'pgd_iterations': 40,
            'bezier_iterations': 30,
            'path_points': 50,
            'random_seed': 42
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results_with_config, f, indent=4)
    
    print(f"\nResults saved to {filename}")
    return filename

if __name__ == "__main__":
    print("Bézier Adversarial Curves - Transferability Experiments (REPRODUCIBLE)")
    print("="*80)
    print("\nKey Design (Reproducible Community Standard):")
    print("• 25 FIXED training samples per setting (deterministic selection)")
    print("• Each sample generates its own PGD endpoints (with fixed seed)")
    print("• Fixed test set of 10 images for evaluation")
    print("• FULLY REPRODUCIBLE - same results every run")
    print("\nData Layout (Aligned with other experiments):")
    print("• []: Reserved for auxiliary (multi_image)")
    print("• [0-9]: Fixed test set")
    print("• [10+]: Training pool")
    print("\nTraining Sample Selection (FIXED):")
    print("• Setting A: Evenly spaced indices from training pool")
    print("• Setting B: Pairs [(0,1), (2,3), ...]")
    print("• Setting C: Cross-class pairs [(pool1[0],pool2[0]), ...]")
    print("\nConfiguration:")
    print("• PGD: 40 iterations with community standard α")
    print("• Bézier: 30 iterations with lr=0.01")
    print("• 50 path sampling points")
    print("• Random seed: 42 (fixed)")
    print("="*80)
    
    results = run_transferability_experiments()
    
    if results:
        print_transferability_results(results)
        filename = save_results(results)
        
        print("\nExperimental Framework Summary:")
        print("• experiment_basic: Tests on training images (no generalization)")
        print("• experiment_transferability: Tests on unseen images (generalization)")
        print("• experiment_multi_image: Tests auxiliary image effects")
        print("• experiment_comprehensive: Tests convergence and sampling")
        print("\nThis forms a coherent system for evaluating Bézier adversarial curves.")

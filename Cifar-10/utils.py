"""
utils.py - Utility functions for Bézier experiments
"""

import torch
import torch.nn as nn
import numpy as np

def normalize_cifar10(x):
    """Normalize CIFAR-10 images"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std

def unnormalize_cifar10(x):
    """Unnormalize CIFAR-10 images"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean

def project_l1_ball(x, eps):
    """Compute Euclidean projection onto L1 ball of radius eps"""
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    batch_size = x.shape[0]
    
    projected = []
    for i in range(batch_size):
        v = x[i]
        u = torch.abs(v)
        if u.sum() <= eps:
            projected.append(v)
        else:
            # L1 projection via soft thresholding
            u_sorted, _ = torch.sort(u, descending=True)
            cumsum = torch.cumsum(u_sorted, dim=0)
            k = torch.arange(1, len(u_sorted) + 1, device=x.device, dtype=torch.float32)
            
            condition = u_sorted > (cumsum - eps) / k
            rho = len(condition) - torch.flip(condition, [0]).long().argmax().item()
            theta = (cumsum[rho-1] - eps) / rho if rho > 0 else 0
            
            projected_v = torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0)
            projected.append(projected_v)
    
    projected = torch.stack(projected)
    return projected.view(original_shape)

class PGDAttack:
    """PGD Attack implementation supporting L1, L2, and Linf norms"""
    
    def __init__(self, model, eps, alpha=None, num_iter=None, norm='linf', randomize=True):
        self.model = model
        self.eps = eps
        self.norm = norm
        self.randomize = randomize
        
        if alpha is None:
            self.alpha = eps / 4 if norm == 'linf' else eps / 10
        else:
            self.alpha = alpha
            
        if num_iter is None:
            self.num_iter = 40 if norm == 'linf' else 40
        else:
            self.num_iter = num_iter
    
    def project_perturbation(self, delta, norm):
        """Project perturbation onto the norm ball"""
        if norm == 'linf':
            return torch.clamp(delta, -self.eps, self.eps)
        elif norm == 'l2':
            delta_flat = delta.view(delta.size(0), -1)
            norm_delta = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            scale = torch.clamp(norm_delta / self.eps, min=1.0)
            delta_flat = delta_flat / scale
            return delta_flat.view_as(delta)
        elif norm == 'l1':
            return project_l1_ball(delta, self.eps)
    
    def perturb(self, x, y, x_min=0.0, x_max=1.0):
        """Generate adversarial perturbation using PGD"""
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.randomize:
            if self.norm == 'linf':
                delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            elif self.norm == 'l2':
                delta = torch.randn_like(x)
                delta_flat = delta.view(delta.size(0), -1)
                norm_delta = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-10
                delta = delta / norm_delta.view(delta.size(0), 1, 1, 1)
                r = torch.rand(x.size(0), 1, 1, 1).to(x.device) * self.eps
                delta = delta * r
            elif self.norm == 'l1':
                delta = torch.zeros_like(x)
                mask = torch.rand_like(x) < 0.1
                delta[mask] = torch.empty(mask.sum()).uniform_(-self.eps/10, self.eps/10).to(x.device)
                delta = self.project_perturbation(delta, self.norm)
        else:
            delta = torch.zeros_like(x)
        
        x_adv = torch.clamp(x + delta, x_min, x_max)
        
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            
            x_adv_norm = normalize_cifar10(x_adv)
            outputs = self.model(x_adv_norm)
            loss = nn.CrossEntropyLoss()(outputs, y)
            
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            
            with torch.no_grad():
                if self.norm == 'linf':
                    x_adv = x_adv + self.alpha * grad.sign()
                elif self.norm == 'l2':
                    grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
                    grad_normalized = grad / (grad_norm + 1e-10)
                    x_adv = x_adv + self.alpha * grad_normalized
                else:  # l1
                    x_adv = x_adv + self.alpha * grad.sign()
                
                delta = x_adv - x
                delta = self.project_perturbation(delta, self.norm)
                x_adv = torch.clamp(x + delta, x_min, x_max)
            
            x_adv = x_adv.detach()
        
        return (x_adv - x).detach()

def evaluate_accuracy(model, dataloader, device):
    """Evaluate model accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_norm = normalize_cifar10(inputs)
            outputs = model(inputs_norm)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

def get_cifar10_classes():
    """Return CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
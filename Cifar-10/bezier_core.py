"""
bezier_core.py - Core Bézier curve optimization implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from utils import normalize_cifar10, project_l1_ball

class BezierAdversarialUnconstrained:
    """Bézier curve optimization with unconstrained theta"""
    
    def __init__(self, model, norm='linf', eps=8/255, lr=0.01, num_iter=100):
        self.model = model
        self.norm = norm
        self.eps = eps
        self.lr = lr
        self.num_iter = num_iter
    
    def bezier_curve(self, p0, p1, p2, t):
        """Quadratic Bézier curve: B(t) = (1-t)²p0 + 2(1-t)t·p1 + t²p2"""
        return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    
    def project_norm_ball(self, delta):
        """Project perturbation to specified norm ball"""
        if self.norm == 'linf':
            return torch.clamp(delta, -self.eps, self.eps)
        elif self.norm == 'l2':
            delta_flat = delta.view(delta.size(0), -1)
            norm_delta = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-10
            scale = torch.clamp(norm_delta / self.eps, min=1.0)
            delta_flat = delta_flat / scale
            return delta_flat.view_as(delta)
        elif self.norm == 'l1':
            return project_l1_ball(delta, self.eps)
    
    def optimize_setting_A(self, x, y, delta1, delta2, num_t_samples=20):
        """Setting A: Single image optimization"""
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            successful_points = 0
            
            t_values = torch.rand(num_t_samples).to(x.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                x_adv = torch.clamp(x + delta_t, 0, 1)
                x_adv_norm = normalize_cifar10(x_adv)
                outputs = self.model(x_adv_norm)
                
                loss = -nn.CrossEntropyLoss()(outputs, y)
                total_loss += loss
                
                pred = outputs.argmax(dim=1)
                if pred != y:
                    successful_points += 1
            
            total_loss /= num_t_samples
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append(successful_points / num_t_samples)
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        print(f"Final theta norm: {theta_norms[-1]:.2f} × eps (unconstrained)")
        
        return theta.detach(), losses, success_rates, theta_norms
    
    def optimize_setting_B(self, x1, x2, y, delta1, delta2, num_t_samples=20):
        """Setting B: Two images from same class"""
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            success_x1 = 0
            success_x2 = 0
            success_both = 0
            
            t_values = torch.rand(num_t_samples).to(x1.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                x1_adv = torch.clamp(x1 + delta_t, 0, 1)
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                
                outputs1 = self.model(normalize_cifar10(x1_adv))
                outputs2 = self.model(normalize_cifar10(x2_adv))
                
                loss1 = -nn.CrossEntropyLoss()(outputs1, y)
                loss2 = -nn.CrossEntropyLoss()(outputs2, y)
                total_loss += (loss1 + loss2) / 2
                
                pred1 = outputs1.argmax(dim=1)
                pred2 = outputs2.argmax(dim=1)
                
                if pred1 != y:
                    success_x1 += 1
                if pred2 != y:
                    success_x2 += 1
                if pred1 != y and pred2 != y:
                    success_both += 1
            
            total_loss /= num_t_samples
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append({
                'x1': success_x1 / num_t_samples,
                'x2': success_x2 / num_t_samples,
                'both': success_both / num_t_samples
            })
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        print(f"Final theta norm: {theta_norms[-1]:.2f} × eps (unconstrained)")
        return theta.detach(), losses, success_rates, theta_norms
    
    def optimize_setting_C(self, x1, x2, y1, y2, delta1, delta2, num_t_samples=20):
        """Setting C: Two images from different classes"""
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            success_x1 = 0
            success_x2 = 0
            success_both = 0
            
            t_values = torch.rand(num_t_samples).to(x1.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                x1_adv = torch.clamp(x1 + delta_t, 0, 1)
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                
                outputs1 = self.model(normalize_cifar10(x1_adv))
                outputs2 = self.model(normalize_cifar10(x2_adv))
                
                loss1 = -nn.CrossEntropyLoss()(outputs1, y1)
                loss2 = -nn.CrossEntropyLoss()(outputs2, y2)
                total_loss += (loss1 + loss2) / 2
                
                pred1 = outputs1.argmax(dim=1)
                pred2 = outputs2.argmax(dim=1)
                
                if pred1 != y1:
                    success_x1 += 1
                if pred2 != y2:
                    success_x2 += 1
                if pred1 != y1 and pred2 != y2:
                    success_both += 1
            
            total_loss /= num_t_samples
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append({
                'x1': success_x1 / num_t_samples,
                'x2': success_x2 / num_t_samples,
                'both': success_both / num_t_samples
            })
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        print(f"Final theta norm: {theta_norms[-1]:.2f} × eps (unconstrained)")
        return theta.detach(), losses, success_rates, theta_norms


class BezierAdversarialMultiImage(BezierAdversarialUnconstrained):
    """Extended Bézier optimization with multiple images"""
    
    def optimize_setting_A_multi(self, x_main, y_main, delta1, delta2, 
                                 additional_images=None, additional_labels=None, 
                                 num_t_samples=20):
        """Setting A with multiple auxiliary images"""
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        if additional_images is not None:
            all_images = [x_main] + additional_images
            all_labels = [y_main] + additional_labels
        else:
            all_images = [x_main]
            all_labels = [y_main]
        
        num_images = len(all_images)
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            successful_points_per_image = [0] * num_images
            
            t_values = torch.rand(num_t_samples).to(x_main.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                for img_idx, (x, y) in enumerate(zip(all_images, all_labels)):
                    x_adv = torch.clamp(x + delta_t, 0, 1)
                    x_adv_norm = normalize_cifar10(x_adv)
                    outputs = self.model(x_adv_norm)
                    
                    loss = -nn.CrossEntropyLoss()(outputs, y)
                    
                    # Main image has higher weight
                    weight = 2.0 if img_idx == 0 else 1.0
                    total_loss += weight * loss
                    
                    pred = outputs.argmax(dim=1)
                    if pred != y:
                        successful_points_per_image[img_idx] += 1
            
            total_weights = 2.0 + (num_images - 1) * 1.0
            total_loss /= (num_t_samples * total_weights)
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append({
                'main': successful_points_per_image[0] / num_t_samples,
                'additional': [s / num_t_samples for s in successful_points_per_image[1:]],
                'all_avg': sum(successful_points_per_image) / (num_t_samples * num_images)
            })
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        return theta.detach(), losses, success_rates, theta_norms
    
    def optimize_setting_B_multi(self, x1, x2, y, delta1, delta2,
                                 additional_images=None, additional_labels=None,
                                 num_t_samples=20):
        """Setting B with multiple auxiliary images"""
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        main_images = [x1, x2]
        main_labels = [y, y]
        
        if additional_images is not None:
            all_images = main_images + additional_images
            all_labels = main_labels + additional_labels
        else:
            all_images = main_images
            all_labels = main_labels
        
        num_images = len(all_images)
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            successful_points_per_image = [0] * num_images
            
            t_values = torch.rand(num_t_samples).to(x1.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                for img_idx, (x, y_label) in enumerate(zip(all_images, all_labels)):
                    x_adv = torch.clamp(x + delta_t, 0, 1)
                    outputs = self.model(normalize_cifar10(x_adv))
                    
                    loss = -nn.CrossEntropyLoss()(outputs, y_label)
                    
                    weight = 2.0 if img_idx < 2 else 1.0
                    total_loss += weight * loss
                    
                    pred = outputs.argmax(dim=1)
                    if pred != y_label:
                        successful_points_per_image[img_idx] += 1
            
            total_weights = 2.0 * 2 + max(0, num_images - 2) * 1.0
            total_loss /= (num_t_samples * total_weights)
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append({
                'x1': successful_points_per_image[0] / num_t_samples,
                'x2': successful_points_per_image[1] / num_t_samples,
                'both_main': min(successful_points_per_image[0], successful_points_per_image[1]) / num_t_samples,
                'additional': [s / num_t_samples for s in successful_points_per_image[2:]],
                'all_avg': sum(successful_points_per_image) / (num_t_samples * num_images)
            })
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        return theta.detach(), losses, success_rates, theta_norms
    
    def optimize_setting_C_multi(self, x1, x2, y1, y2, delta1, delta2,
                                 additional_images=None, additional_labels=None,
                                 num_t_samples=20):
        """Setting C with multiple auxiliary images"""
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        main_images = [x1, x2]
        main_labels = [y1, y2]
        
        if additional_images is not None:
            all_images = main_images + additional_images
            all_labels = main_labels + additional_labels
        else:
            all_images = main_images
            all_labels = main_labels
        
        num_images = len(all_images)
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            successful_points_per_image = [0] * num_images
            
            t_values = torch.rand(num_t_samples).to(x1.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                for img_idx, (x, y_label) in enumerate(zip(all_images, all_labels)):
                    x_adv = torch.clamp(x + delta_t, 0, 1)
                    outputs = self.model(normalize_cifar10(x_adv))
                    
                    loss = -nn.CrossEntropyLoss()(outputs, y_label)
                    
                    weight = 2.0 if img_idx < 2 else 1.0
                    total_loss += weight * loss
                    
                    pred = outputs.argmax(dim=1)
                    if pred != y_label:
                        successful_points_per_image[img_idx] += 1
            
            total_weights = 2.0 * 2 + max(0, num_images - 2) * 1.0
            total_loss /= (num_t_samples * total_weights)
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append({
                'x1': successful_points_per_image[0] / num_t_samples,
                'x2': successful_points_per_image[1] / num_t_samples,
                'both_main': min(successful_points_per_image[0], successful_points_per_image[1]) / num_t_samples,
                'additional': [s / num_t_samples for s in successful_points_per_image[2:]],
                'all_avg': sum(successful_points_per_image) / (num_t_samples * num_images)
            })
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        return theta.detach(), losses, success_rates, theta_norms
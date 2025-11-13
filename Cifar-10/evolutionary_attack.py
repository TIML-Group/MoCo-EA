"""
evolutionary_attack.py - Evolutionary algorithm with traditional and Bézier crossover
Fixed: weak initialization to show evolution progress
Modified: Removed adaptive method
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Tuple, Dict
from tqdm import tqdm
import random

from utils import normalize_cifar10, PGDAttack
from bezier_core import BezierAdversarialUnconstrained

class EvolutionaryAttack:
    """Base evolutionary attack with configurable crossover"""
    
    def __init__(self, model, eps=8/255, norm='linf', 
                 population_size=50, elite_size=10,
                 mutation_rate=0.1, mutation_strength=0.02):
        self.model = model
        self.eps = eps
        self.norm = norm
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength * eps
        self.device = next(model.parameters()).device
        self.query_count = 0
        
        # For Bézier operations
        self.bezier = BezierAdversarialUnconstrained(
            model, norm=norm, eps=eps, lr=0.1, num_iter=5
        )
    
    def initialize_population(self, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        """Initialize population with WEAK random perturbations to show evolution"""
        population = []
        
        # Start with weak perturbations (20% of epsilon)
        # This ensures we don't immediately succeed and can see evolution
        initial_strength = 0.2
        
        while len(population) < self.population_size:
            if self.norm == 'linf':
                delta = torch.empty_like(x).uniform_(-self.eps * initial_strength, 
                                                     self.eps * initial_strength)
            elif self.norm == 'l2':
                delta = torch.randn_like(x)
                delta = delta / (torch.norm(delta.flatten()) + 1e-10) * self.eps * initial_strength
            else:  # l1
                delta = torch.randn_like(x) * self.eps * initial_strength * 0.1
                delta = self.bezier.project_norm_ball(delta * 5)  # Project after scaling up
            
            population.append(delta)
        
        return population
    
    def evaluate_fitness(self, population: List[torch.Tensor], 
                        x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """
        Evaluate fitness of population members
        Higher fitness = better attack
        """
        fitness_scores = []
        
        for delta in population:
            x_adv = torch.clamp(x + delta, 0, 1)
            with torch.no_grad():
                outputs = self.model(normalize_cifar10(x_adv))
                pred = outputs.argmax(dim=1)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                correct_prob = probs[0, y].item()
                
                if pred != y:
                    # Successful attack - high fitness
                    # Fitness = 2.0 + (1 - correct_prob) to ensure > 1
                    fitness = 2.0 + (1.0 - correct_prob)
                else:
                    # Failed attack - fitness based on how close we are
                    # Lower correct_prob = closer to success = higher fitness
                    fitness = 1.0 - correct_prob
            
            fitness_scores.append(fitness)
            self.query_count += 1
        
        return np.array(fitness_scores)
    
    def traditional_crossover(self, parent1: torch.Tensor, 
                            parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Traditional uniform crossover"""
        # Uniform crossover
        mask = torch.rand_like(parent1) > 0.5
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
        
        # Project to norm ball
        child1 = self.bezier.project_norm_ball(child1)
        child2 = self.bezier.project_norm_ball(child2)
        
        return child1, child2
    
    def bezier_crossover(self, parent1: torch.Tensor, parent2: torch.Tensor,
                        x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bézier-based crossover"""
        # Quick optimization of control point
        theta = ((parent1 + parent2) / 2).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=0.2)
        
        # Fast optimization (5 iterations)
        for _ in range(5):
            optimizer.zero_grad()
            loss_total = 0
            
            # Sample a few t values
            t_values = torch.tensor([0.25, 0.5, 0.75]).to(self.device)
            for t in t_values:
                delta_t = self.bezier.bezier_curve(parent1, theta, parent2, t)
                delta_t = self.bezier.project_norm_ball(delta_t)
                x_adv = torch.clamp(x + delta_t, 0, 1)
                outputs = self.model(normalize_cifar10(x_adv))
                
                # We want to MAXIMIZE the loss on the true label
                loss = -nn.CrossEntropyLoss()(outputs, y)
                loss_total += loss
                self.query_count += 1
            
            loss_total.backward()
            optimizer.step()
        
        theta = theta.detach()
        
        # Select best points from different regions
        candidates_left = []
        candidates_right = []
        
        # Evaluate left region
        for t in [0.1, 0.25, 0.4]:
            delta_t = self.bezier.bezier_curve(parent1, theta, parent2, t)
            delta_t = self.bezier.project_norm_ball(delta_t)
            fitness = self._evaluate_single_fitness(x + delta_t, y)
            candidates_left.append((delta_t, fitness))
            self.query_count += 1
        
        # Evaluate right region
        for t in [0.6, 0.75, 0.9]:
            delta_t = self.bezier.bezier_curve(parent1, theta, parent2, t)
            delta_t = self.bezier.project_norm_ball(delta_t)
            fitness = self._evaluate_single_fitness(x + delta_t, y)
            candidates_right.append((delta_t, fitness))
            self.query_count += 1
        
        # Select HIGHEST fitness (best attack) from each region
        child1 = max(candidates_left, key=lambda x: x[1])[0]
        child2 = max(candidates_right, key=lambda x: x[1])[0]
        
        return child1, child2
    
    def _evaluate_single_fitness(self, x_adv: torch.Tensor, y: torch.Tensor) -> float:
        """Helper function to evaluate fitness of a single perturbation"""
        x_adv = torch.clamp(x_adv, 0, 1)
        with torch.no_grad():
            outputs = self.model(normalize_cifar10(x_adv))
            pred = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)
            correct_prob = probs[0, y].item()
            
            if pred != y:
                fitness = 2.0 + (1.0 - correct_prob)
            else:
                fitness = 1.0 - correct_prob
        
        return fitness
    
    def mutate(self, individual: torch.Tensor) -> torch.Tensor:
        """Apply mutation to individual"""
        if torch.rand(1).item() < self.mutation_rate:
            noise = torch.randn_like(individual) * self.mutation_strength
            individual = individual + noise
            individual = self.bezier.project_norm_ball(individual)
        return individual
    
    def selection(self, population: List[torch.Tensor], 
                 fitness: np.ndarray) -> List[torch.Tensor]:
        """Tournament selection - select individuals with HIGHER fitness"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness[tournament_idx]
            # Select the one with HIGHEST fitness
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].clone())
        
        return selected
    
    def evolve(self, x: torch.Tensor, y: torch.Tensor, 
              max_generations: int = 100,
              crossover_type: str = 'traditional',
              early_stop_fitness: float = 2.0) -> Dict:
        """
        Main evolution loop
        early_stop_fitness >= 2.0 means successful attack
        """
        start_time = time.time()
        population = self.initialize_population(x, y)
        
        stats = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'success': [],
            'query_counts': [],
            'time_elapsed': []
        }
        
        best_perturbation = None
        best_fitness_ever = -float('inf')
        
        for gen in range(max_generations):
            # Evaluate fitness
            fitness = self.evaluate_fitness(population, x, y)
            
            # Track statistics
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(fitness)
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_perturbation = population[best_idx].clone()
            
            # Check if attack successful (fitness >= 2.0 means misclassification)
            success = best_fitness >= early_stop_fitness
            
            stats['generations'].append(gen)
            stats['best_fitness'].append(float(best_fitness))
            stats['avg_fitness'].append(float(avg_fitness))
            stats['success'].append(success)
            stats['query_counts'].append(self.query_count)
            stats['time_elapsed'].append(time.time() - start_time)
            
            # Early stopping
            if success:
                print(f"  Attack successful at generation {gen} (fitness={best_fitness:.3f})")
                break
            
            # Selection
            parents = self.selection(population, fitness)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents)-1, 2):
                if crossover_type == 'traditional':
                    child1, child2 = self.traditional_crossover(parents[i], parents[i+1])
                elif crossover_type == 'bezier':
                    child1, child2 = self.bezier_crossover(parents[i], parents[i+1], x, y)
                else:
                    raise ValueError(f"Unknown crossover type: {crossover_type}. Use 'traditional' or 'bezier'")
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])
            
            # Elite preservation
            elite_idx = np.argsort(fitness)[-self.elite_size:]
            elite = [population[i].clone() for i in elite_idx]
            
            # New population
            population = elite + offspring[:self.population_size - self.elite_size]
        
        stats['best_perturbation'] = best_perturbation
        stats['final_generation'] = gen
        
        return stats
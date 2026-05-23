## [[ICML 2026] MoCo-EA: Exploiting Adversarial Mode Connectivity for Efficient Evolutionary Attacks ](https://arxiv.org/abs/2605.18919)

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/pdf/2605.18919) [![Paper](https://img.shields.io/badge/Paper-ICML_2026-green)](https://openreview.net/forum?id=RB4V5TkIOz) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### Abstract
>Evolutionary algorithms for adversarial attacks leverage population-based search to discover perturbations without gradient information, but suffer from inefficient crossover operations that destroy adversarial properties through discrete interpolation. We introduce Mode Connectivity Evolutionary Attack (MoCo-EA), which replaces traditional crossover with a novel Bézier crossover operator that optimizes perturbations along a continuous Bézier curve between parent perturbations. Our key insight is that adversarial examples lie on connected manifolds where intermediate points maintain and often enhance attack effectiveness. We demonstrate three findings: (1) Successful adversarial perturbations exhibit mode connectivity; (2) Intermediate points along optimized paths achieve higher transferability than endpoints; (3) Bézier crossover dramatically outperforms discrete genetic operations while reducing convergence time and query requirements. By exploiting the geometric structure of adversarial space through path optimization, MoCo-EA provides an efficient and reliable method. Our work challenges the traditional view of adversarial examples as isolated points and opens new directions for both attack generation and defense research.
  
### Updates
* (2026/05): [Preprint](https://arxiv.org/pdf/2605.18919) has been uploaded.
* (2026/05): Our paper has been accepted at [ICML 2026](https://icml.cc/)🎉🎉🎉🎉
* (2026/04): Our paper has been accepted for Oral Presentation at [ICLR 2026 Workshop on Principled Design for Trustworthy AI](https://trustworthy-ai-workshop.github.io/iclr2026/)🎉🎉🎉🎉

### Setup
```bash
pip install -r requirements.txt
```

### Running Experiments
All experiments are launched from the repository root using `run_experiment.py`:

```bash
# Mode connectivity
python run_experiment.py --experiment connectivity --dataset cifar10 --data-root ./dataset/CIFAR10 --checkpoint resnet18_cifar10_best.pth --output-dir ./results
python run_experiment.py --experiment connectivity --dataset imagenet --data-root ./dataset/imagenet/val --output-dir ./results

# Other experiments
python run_experiment.py --experiment evolutionary --dataset cifar10 --data-root ./dataset/CIFAR10 --checkpoint resnet18_cifar10_best.pth --output-dir ./results
python run_experiment.py --experiment transferability --dataset imagenet --data-root ./dataset/imagenet/val --output-dir ./results
```

CIFAR-10 experiments expect a trained `resnet18_cifar10_best.pth` checkpoint. It can be generated with:

```bash
python cifar10/train_model.py --data-root ./dataset/CIFAR10 --output-dir .
```

### Cite
```
@inproceedings{kim2026mocoea,
  title={MoCo-EA: Exploiting Adversarial Mode Connectivity for Efficient Evolutionary Attacks},
  author={Kim, Hyo Seo and Luo, Gang and Chen, Can and Wang, Binghui and Duan, Yue and Wang, Ren},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

# Global Optimization via Evolution Strategies

[![Domain: Computational Intelligence](https://img.shields.io/badge/Domain-Computational%20Intelligence-blue)](#)
[![Algorithms: ES & L-BFGS-B](https://img.shields.io/badge/Algorithms-ES%20%7C%20L--BFGS--B-ee4c2c)](#)
[![Task: Non-Convex Optimization](https://img.shields.io/badge/Task-Non--Convex%20Optimization-brightgreen)](#)


## Project Overview
Traditional derivative-based methods are highly efficient for convex problems but often fail in complex, non-differentiable, and highly multimodal search space. 

This project explores the implementation, parameter tuning, and statistical benchmarking of a derivative-free **$(\mu,\lambda)$-Evolution Strategy (ES)**. The algorithm is evaluated on the highly multimodal **Rastrigin function**, which is characterized by a vast number of deceptive local optima surrounding a single global minimum.

## Algorithmic Implementation

### 1. Evolution Strategy (ES)
* **Architecture:** Deterministic $(\mu,\lambda)$ truncation selection (non-elitist strategy to promote exploration).
* **Mutation Operator:** Implemented diagonal self-adaptive mutation (uncorrelated step sizes), allowing the algorithm to learn the appropriate mutation strength for each dimension independently.
* **Stopping Criteria:** Maximum iterations, target fitness achievement, and fitness stagnation over generations.

### 2. Gradient-Based Baseline (L-BFGS-B)
* To establish a strong baseline, the ES was compared against **L-BFGS-B**, a quasi-Newton gradient-based method.
* Since L-BFGS-B is a local optimizer, it was implemented within a **multi-start wrapper** (1000 random initializations) to transform it into a stochastic global optimizer.

## 📊 Experimental Phases & Statistical Analysis

The algorithms were rigorously tested across 30 independent trials to ensure statistical reliability[cite: 189]. [cite_start]Significance was determined using **Kruskal-Wallis H-tests**, **Dunn's post-hoc tests**, and **Mann-Whitney U tests**.

1. **Parameter Tuning (Exploration vs. Exploitation):**

2. **Derivative-Free vs. Gradient-Based Benchmarking:**

3. **Scalability and The Curse of Dimensionality:**

## 🛠️ Tech Stack
* **Language:** Python
* **Optimization:** SciPy (`scipy.optimize.minimize` for L-BFGS-B)
* **Statistical Testing:** SciPy Stats (Kruskal-Wallis, Mann-Whitney U)
* **Visualization:** Matplotlib, Seaborn

# Multi-Agent UAV-UGV Cooperative Routing Framework

This repository provides a modular implementation for training, evaluating, and benchmarking multi-agent UAV-UGV cooperative routing strategies using custom reinforcement learning environments and baselines.

---

## 📁 Repository Structure

### 🔧 Environment Models
These define the routing environment logic and agent interaction rules.

- **Custom_environment_model_v1**  
  Basic environment model with standard agent control and task transitions.

- **Custom_environment_model_v3_v2**  
  Environment model supporting **sortie-based agent switching**, where agents alternate by sortie segments.

- **Custom_environment_model_v4_v2**  
  Environment model supporting **per-step agent switching**, where control alternates after every decision step.

---

### 🚀 Training and Evaluation

- **train_modified**  
  Main training script for learning routing policies using the defined environment and DRL framework.

- **eval_modified**  
  Evaluates the trained policy using the **sampling-based decoding** strategy.

- **eval_modified_10240**  
  Similar to `eval_modified` but configured for **larger test set evaluation** (e.g., 10,240 samples).

- **eval_modified_greedy**  
  Evaluates the trained policy using **greedy decoding**, selecting the highest-probability action at each step.

---

### 📦 Data Generation

- **generate_data**  
  Script to generate datasets required for training or evaluation (e.g., mission points, graph instances).

- **scenario_gen_v2**  
  Script to generate full mission scenarios for evaluation (e.g., vehicle positions, mission layouts, constraints).

---

### 🧪 Execution and Config

- **run**  
  Main entry point for running training or evaluation workflows based on selected options.

- **options**  
  Defines configurable parameters and experiment settings (e.g., model type, problem size, decoding method).

---

### 📊 Baselines

- **reinforce_baselines**  
  Includes baseline implementations to assess training stability and compare against DRL-based models.

---

## 🧠 Key Features

- Customizable UAV-UGV cooperative environment with agent switching modes
- Transformer-based encoder-decoder routing policy
- Sortie-wise and per-step agent alternation options
- Sampling and greedy decoding support for evaluation
- Modular scenario and data generation pipeline
- Support for risk-aware and energy-constrained mission planning

---

## 🛠️ Usage

### 1. Train a model
```bash
python train_modified

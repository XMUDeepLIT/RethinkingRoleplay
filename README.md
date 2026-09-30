# 🤖 Rethinking the Necessity of Role-Playing for Large Language Models

> 🧪 This study explores whether role-playing offers universal benefits, analyzes its underlying mechanisms, and further analyzes directions for optimization.

This repository contains the code for the systematic study. The framework is **model-agnostic** — any model supported by vLLM can be used by simply changing `MODEL_PATH` in the configuration block of each script.

---

## 📁 Repository Structure

```
.
├── 📄 requirements.txt                  # Python dependencies
│
├── 📂 preliminary-study/                # 🔰 Phase 1: Baseline Experiments
│   ├── 📜 Preliminary-study_MMLU_test.py         # MMLU (57 subjects) × 8 languages × base/roleplay
│   └── 📜 Preliminary-study_MMLU-redux_test.py   # MMLU-Redux (30 subjects) × 8 languages × base/roleplay
│
├── 📂 hypothesis-verification/          # 🔬 Phase 2: Core Hypothesis Verification
│   ├── 📜 verification_1.py             # Accuracy comparison: Base vs. Roleplay prompts
│   ├── 📜 verification_2.py             # Layer-wise entropy difference
│   └── 📜 verification_3.py             # PCA representation-space deflection analysis
│
├── 📂 optimization-method/              # 🚀 Phase 3: Multilingual Optimization
│   ├── 📜 MLCP.py                       # Multi-Lingual Concatenated Prompt
│   ├── 📜 MLVP.py                       # Multi-Lingual Voting Prompt
│   └── 📜 language_angle_distance.py    # Angular distance between language representations
│
└── 📂 quantitative-analysis/            # 📊 Phase 4: Quantitative Ablation
    └── 📜 Quantitative_Analysis_language.py  # Random k-language combination ablation study
```

---

## 🛠️ Environment Setup

### 📦 Installation

```bash
# Create a virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Experiments

### 🔗 Experiment Pipeline

```
Preliminary Study → Hypothesis Verification → Optimization Methods → Quantitative Analysis
     🔰 (1)                   🔬 (2)                     🚀 (3)                    📊 (4)
```

### 🔰 Phase 1: Preliminary Study

```bash
# MMLU benchmark (57 subjects, 8 languages)
cd preliminary-study
python Preliminary-study_MMLU_test.py

# MMLU-Redux benchmark (30 subjects, 8 languages)
python Preliminary-study_MMLU-redux_test.py
```

### 🔬 Phase 2: Hypothesis Verification

```bash
cd hypothesis-verification

# Experiment 1: Base vs. Roleplay prompt accuracy
python verification_1.py

# Experiment 2: Layer-wise entropy difference
python verification_2.py

# Experiment 3: PCA representation-space deflection
python verification_3.py
```

### 🚀 Phase 3: Multilingual Optimization

```bash
cd optimization-method

# Language-space angular distance analysis
python language_angle_distance.py

# Multi-Language Concatenated Prompt (MLCP)
python MLCP.py

# Multi-Language Voting Prompt (MLVP)
python MLVP.py
```

### 📊 Phase 4: Quantitative Analysis

```bash
cd quantitative-analysis
python Quantitative_Analysis_language.py
```

---

## 🧪 Experimental Design

### 🧠 Models

The framework is model-agnostic. Set `MODEL_PATH` to any HuggingFace model identifier or local path supported by vLLM. All prompting strategies and analysis methods are model-independent.

### 🌐 Test Languages

Experiments cover **8 languages**:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | 🇬🇧 English | `ja` | 🇯🇵 Japanese |
| `zh` | 🇨🇳 Chinese | `de` | 🇩🇪 German |
| `es` | 🇪🇸 Spanish | `km` | 🇰🇭 Khmer |
| `fr` | 🇫🇷 French | `ms` | 🇲🇾 Malay |

### 💬 Prompt Setting

- **🎭 Roleplay**: An expert persona prompt, e.g., *"You are an expert in anatomy. Please answer the following question based on your professional knowledge."*

---

## 📈 Output Files

After running experiments, the following types of output files are generated:

- **📋 `*.json`**: Accuracy statistics, per-subject and global result summaries
- **💾 `*.npz`**: Intermediate data such as hidden states and entropy values
- **🖼️ `*.png`**: Visualizations (300 DPI, suitable for publication)

💡 All scripts support **incremental save / resume**: if an experiment is interrupted, re-running the script will automatically pick up from the last checkpoint (`SAVE_INTERVAL` controls the save frequency).

---

## ⚙️ Configuration

Each script contains a `# ======== Configuration ========` block at the top, where you can adjust:

- 🔧 `MODEL_PATH` — Model path (local directory or HuggingFace identifier); swap this to use any vLLM-compatible model
- 🖥️ `GPU_IDS` / `CUDA_VISIBLE_DEVICES` — Which GPUs to use
- 💾 `SAVE_INTERVAL` — How often to save intermediate results
- 🔗 `tensor_parallel_size` — Tensor parallelism degree
- 📝 Language list, subject list, etc.

---

## 📄 License

This project is intended for academic research purposes only.

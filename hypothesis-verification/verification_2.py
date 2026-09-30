#!/usr/bin/env python3
"""
MMLU-Redux Entropy Difference Visualization - Three Model Comparison Line Chart
Reads from saved npz files and generates comparison line charts for 3 models
Unified y-axis range: -1.25 to 1.25
X-axis extends to maximum layer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ======================== Configuration ========================
MODEL_PATHS = {
    "4B": "./mmlu_redux_entropy_diff_Qwen3-4B",
    "8B": "./mmlu_redux_entropy_diff_Qwen3-8B",
    "14B": "./mmlu_redux_entropy_diff_Qwen3-14B",
}

MODEL_CONFIGS = {
    "4B": {
        "display_name": "Qwen3-4B",
        "color": "#7952A7",  
        "linestyle": "-",
        "marker": "o",
        "markersize": 6,
        "linewidth": 2.5,
    },
    "8B": {
        "display_name": "Qwen3-8B",
        "color": "#3B87B9",  
        "linestyle": "-",
        "marker": "s",
        "markersize": 6,
        "linewidth": 2.5,
    },
    "14B": {
        "display_name": "Qwen3-14B",
        "color": "#48BD7E",  
        "linestyle": "-",
        "marker": "^",
        "markersize": 6,
        "linewidth": 2.5,
    },
}

ALL_SUBJECTS = [
    "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_chemistry", "college_computer_science", "college_mathematics", 
    "college_medicine", "college_physics", "conceptual_physics", 
    "econometrics", "electrical_engineering", "formal_logic", "global_facts",
    "high_school_chemistry", "high_school_geography", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "high_school_us_history", "human_aging", "logical_fallacies", 
    "machine_learning", "miscellaneous", "philosophy", "professional_accounting",
    "professional_law", "public_relations", "virology"
]

Y_AXIS_LIMITS = (-1.25, 1.25)

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'


# ======================== Data Loading ========================
def load_model_data(model_key, subject):
    base_path = MODEL_PATHS[model_key]
    data_dir = os.path.join(base_path, subject)
    
    base_file = os.path.join(data_dir, f"{subject}_base_layer_entropies.npz")
    role_file = os.path.join(data_dir, f"{subject}_roleplay_layer_entropies.npz")
    
    if not os.path.exists(base_file) or not os.path.exists(role_file):
        return None
    
    try:
        base_data = np.load(base_file, allow_pickle=True)
        role_data = np.load(role_file, allow_pickle=True)
        
        layers = []
        for key in base_data.keys():
            if key.startswith('layer_') and key.endswith('_entropy'):
                layer_idx = int(key.split('_')[1])
                layers.append(layer_idx)
        layers = sorted(layers)
        
        diff_means = []
        diff_stds = []
        
        for layer in layers:
            base_vals = base_data[f'layer_{layer}_entropy']
            role_vals = role_data[f'layer_{layer}_entropy']
            
            min_len = min(len(base_vals), len(role_vals))
            diffs = role_vals[:min_len] - base_vals[:min_len]
            
            diff_means.append(np.mean(diffs))
            diff_stds.append(np.std(diffs))
        
        return {
            "layers": layers,
            "diff_means": np.array(diff_means),
            "diff_stds": np.array(diff_stds),
            "n_samples": min_len,
        }
    except Exception as e:
        print(f"  ⚠️ Error loading {model_key}/{subject}: {e}")
        return None


# ======================== Main Function ========================
def main():
    print("="*80)
    print("MMLU-Redux | Entropy Difference - Model Comparison")
    print(f"Models: {', '.join(MODEL_PATHS.keys())}")
    print(f"Y-axis range: {Y_AXIS_LIMITS}")
    print("="*80)
    
    print("\n📂 Loading data from npz files...")
    
    layer_diffs = {model_key: {} for model_key in MODEL_PATHS.keys()}
    
    for subj in ALL_SUBJECTS:
        print(f"  Processing: {subj}")
        for model_key in MODEL_PATHS.keys():
            data = load_model_data(model_key, subj)
            if data is not None:
                for i, layer in enumerate(data["layers"]):
                    if layer not in layer_diffs[model_key]:
                        layer_diffs[model_key][layer] = []
                    layer_diffs[model_key][layer].append(data["diff_means"][i])
    
    max_layer = 0
    for model_key in layer_diffs:
        if layer_diffs[model_key]:
            layers = list(layer_diffs[model_key].keys())
            if layers:
                max_layer = max(max_layer, max(layers))
    
    print(f"\n  Maximum layer index: {max_layer}")
    
    has_data = False
    for model_key in layer_diffs:
        if layer_diffs[model_key]:
            has_data = True
            break
    
    if not has_data:
        print("❌ No data loaded!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model_key, diffs_by_layer in layer_diffs.items():
        if not diffs_by_layer:
            print(f"  ⚠️ No data for {model_key}")
            continue
        
        layers = sorted(diffs_by_layer.keys())
        means = []
        stds = []
        
        for layer in layers:
            diffs = diffs_by_layer[layer]
            means.append(np.mean(diffs))
            stds.append(np.std(diffs))
        
        config = MODEL_CONFIGS[model_key]
        x = layers
        
        min_layer = min(layers)
        max_layer_actual = max(layers)
        
        ax.fill_between(x, 
                        np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds), 
                        alpha=0.15, 
                        color=config["color"])
        
        ax.plot(x, means, 
                marker=config["marker"], 
                markersize=config["markersize"],
                color=config["color"], 
                linestyle=config["linestyle"],
                linewidth=config["linewidth"],
                label=f"{config['display_name']} (layers {min_layer}-{max_layer_actual})")
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Zero')
    
    ax.set_ylim(Y_AXIS_LIMITS)
    
    ax.set_xlim(-1, max_layer + 1)
    
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Entropy Difference (Role-playing - Base)', fontsize=14, fontweight='bold')
    
    step = max(1, max_layer // 10)
    xticks = list(range(0, max_layer + 1, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    
    ax.grid(alpha=0.3, linestyle='--')
    
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    
    out_path = "entropy_diff_three_models.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {out_path}")
    plt.close()
    
    print("="*80)


if __name__ == "__main__":
    main()
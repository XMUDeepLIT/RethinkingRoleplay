#!/usr/bin/env python3
"""
MMLU-Redux 2D Deflection Vector Analysis (L2 Euclidean Norm)
"""

import os
import json
import re
import gc
import argparse
import numpy as np
import torch
import pyarrow as pa
from tqdm import tqdm
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================== Configuration ========================
MODEL_PATH = "Qwen/Qwen3-8B"
DATASET_BASE = "mmlu-redux"
OUTPUT_BASE = "./mmlu_redux_PCA_2D_analysis"

TEST_SUBJECTS = [
    "professional_accounting",
    "college_computer_science",
    "high_school_physics"
]

SUBJECT_DISPLAY = {
    "professional_accounting": "Professional Accounting",
    "college_computer_science": "College Computer Science",
    "high_school_physics": "High School Physics"
}
SUBJECT_COLORS = {
    "professional_accounting": "#7389AE",
    "college_computer_science": "#E8A87C",
    "high_school_physics": "#81B29A"
}

TEST_SAMPLES = 100
SAVE_INTERVAL = 10
DEBUG_SAMPLES = 3

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'


# ======================== Prompt Construction ========================

def build_prompt_base(question, choices):
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    return f"""Answer the following multiple-choice question. Choose the correct option.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text."""


def build_prompt_roleplay(question, choices, subject):
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    subject_display = subject.replace('_', ' ').title()
    return f"""You are an expert in the field of {subject_display}. Please answer the following question based on your professional knowledge.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text."""


# ======================== Answer Extraction ========================

def extract_answer(response: str) -> str:
    if not response:
        return None
    
    original_response = response
    
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'\n+', '\n', response)
    response = response.strip()
    
    if not response:
        response = original_response
    
    match = re.search(r'^([A-D])$', response.strip())
    if match:
        return match.group(1)
    
    match = re.search(r'^([A-D])[\.\)]?\s*$', response.strip())
    if match:
        return match.group(1)
    
    patterns = [
        r'[Aa]nswer:\s*([A-D])',
        r'[Aa]nswer\s+is\s+([A-D])',
        r'[Tt]he\s+correct\s+answer\s+is\s+([A-D])',
        r'[Tt]he\s+answer\s+is\s+([A-D])',
        r'[Oo]ption\s+([A-D])\s+is\s+correct',
        r'[Cc]orrect\s+answer:\s*([A-D])',
        r'[Cc]orrect\s+option\s+is\s+([A-D])',
        r'^([A-D])\s+is\s+correct',
        r'^([A-D])\s+[-:]',
        r'\b([A-D])\b.*?(?:correct|answer)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    words = re.findall(r'\b([A-D])\b', response)
    if words:
        return words[0]
    
    return None


# ======================== Data Loading ========================

def load_mmlu_redux_data(subject, num_samples=None):
    arrow_file = f"{DATASET_BASE}/{subject}/data-00000-of-00001.arrow"
    
    if not os.path.exists(arrow_file):
        print(f"  ❌ File not found: {arrow_file}")
        return None
    
    try:
        with open(arrow_file, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
        
        data = []
        max_samples = min(len(table), num_samples) if num_samples else len(table)
        
        for i in range(max_samples):
            row = {col: table[col][i].as_py() for col in table.column_names}
            
            choices_str = row.get('choices', '[]')
            if isinstance(choices_str, str):
                try:
                    choices = eval(choices_str)
                except:
                    choices = [c.strip().strip('"\'') for c in choices_str.strip('[]').split(',')]
            else:
                choices = choices_str
            
            while len(choices) < 4:
                choices.append("")
            
            answer_idx = row.get('answer', 0)
            answer_letter = chr(65 + answer_idx)
            
            data.append({
                "idx": i,
                "question": row.get('question', ''),
                "choices": choices,
                "answer_letter": answer_letter,
            })
        
        print(f"  ✓ Loaded {len(data)} samples from {subject}")
        return data
    except Exception as e:
        print(f"  ❌ Error loading {subject}: {e}")
        return None


# ======================== Hidden State Extraction ========================

@torch.no_grad()
def extract_hidden_state(model, tokenizer, prompt, device, sample_idx=0, mode="", debug=False):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=16,
        output_hidden_states=True,
        return_dict_in_generate=True,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
    )
    
    hidden_state_avg = None
    
    if outputs.hidden_states is not None:
        last_step_hidden = outputs.hidden_states[-1]
        
        all_layers = []
        for layer_idx, layer_hidden in enumerate(last_step_hidden):
            last_token_hidden = layer_hidden[0, -1, :].cpu().float().numpy()
            all_layers.append(last_token_hidden)
        
        hidden_state_avg = np.mean(all_layers, axis=0)
    
    if hidden_state_avg is None:
        hidden_state_avg = np.zeros(4096)
    
    if debug and sample_idx < DEBUG_SAMPLES:
        print(f"\n  [DEBUG] Sample {sample_idx} - {mode}")
        print(f"    Avg hidden state norm: {np.linalg.norm(hidden_state_avg):.3f}")
    
    return hidden_state_avg


# ======================== Result Saving ========================

def save_hidden_states(output_dir, subject, mode, hidden_states_list):
    os.makedirs(output_dir, exist_ok=True)
    
    if hidden_states_list:
        hidden_file = os.path.join(output_dir, f"{subject}_{mode}_hidden_states.npz")
        valid_hidden = [h for h in hidden_states_list if h is not None and len(h) > 0]
        if valid_hidden:
            hidden_array = np.stack(valid_hidden, axis=0)
            np.savez_compressed(hidden_file, hidden_states=hidden_array)
            print(f"  💾 Saved hidden states: {hidden_array.shape}")


def load_hidden_states(output_dir, subject, mode):
    hidden_file = os.path.join(output_dir, f"{subject}_{mode}_hidden_states.npz")
    
    if os.path.exists(hidden_file):
        try:
            data = np.load(hidden_file, allow_pickle=True)
            hidden_states = [data['hidden_states'][i] for i in range(data['hidden_states'].shape[0])]
            print(f"  🔄 Loaded {len(hidden_states)} existing hidden states for {mode}")
            return hidden_states
        except Exception as e:
            print(f"  ⚠️ Failed to load hidden states: {e}")
    
    return None


# ======================== Subject Processing ========================

def process_subject(subject, model, tokenizer, device, output_base, debug=False):
    print(f"\n{'='*60}")
    print(f"Processing: {subject}")
    print(f"{'='*60}")
    
    samples = load_mmlu_redux_data(subject, TEST_SAMPLES)
    if not samples:
        return None, None
    
    output_dir = os.path.join(output_base, subject)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for mode in ['base', 'roleplay']:
        print(f"\n  📊 {mode.upper()} mode")
        
        existing_hidden = load_hidden_states(output_dir, subject, mode)
        
        if existing_hidden is not None and len(existing_hidden) == len(samples):
            print(f"    ✅ Already processed, skipping")
            results[mode] = {'hidden': existing_hidden}
            continue
        
        hidden_list = []
        
        processed_count = len(existing_hidden) if existing_hidden else 0
        pending_samples = samples[processed_count:]
        
        print(f"    📍 Processing {len(pending_samples)}/{len(samples)} samples")
        
        for i, sample in enumerate(tqdm(pending_samples, desc=f"    Extracting {mode}")):
            if mode == 'base':
                prompt = build_prompt_base(sample['question'], sample['choices'])
            else:
                prompt = build_prompt_roleplay(sample['question'], sample['choices'], subject)
            
            try:
                hidden_state = extract_hidden_state(
                    model, tokenizer, prompt, device, 
                    sample_idx=sample['idx'], mode=mode, debug=debug
                )
                hidden_list.append(hidden_state)
                
            except Exception as e:
                print(f"\n      ⚠️ Error on sample {sample['idx']}: {e}")
                hidden_list.append(None)
            
            if (i + 1) % SAVE_INTERVAL == 0 or (i + 1) == len(pending_samples):
                all_hidden = (existing_hidden or []) + hidden_list
                save_hidden_states(output_dir, subject, mode, all_hidden)
        
        all_hidden = (existing_hidden or []) + hidden_list
        save_hidden_states(output_dir, subject, mode, all_hidden)
        print(f"    ✅ Completed {mode}: {len(all_hidden)} hidden states extracted")
        
        results[mode] = {'hidden': all_hidden}
    
    return results['base'], results['roleplay']


# ======================== 2D Deflection Vector Plot ========================

def plot_deflection_vectors_2d(all_subjects_data, output_base):
    n_subjects = len(all_subjects_data)
    fig, axes = plt.subplots(1, n_subjects, figsize=(5 * n_subjects, 5))
    
    if n_subjects == 1:
        axes = [axes]
    
    for idx, (subject, data) in enumerate(all_subjects_data.items()):
        ax = axes[idx]
        
        hidden_base = data['hidden_base']
        hidden_role = data['hidden_role']
        
        if not hidden_base or not hidden_role:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        
        base_reps = []
        role_reps = []
        
        for hb, hr in zip(hidden_base, hidden_role):
            if hb is not None and hr is not None and len(hb) > 0:
                base_reps.append(hb)
                role_reps.append(hr)
        
        if len(base_reps) == 0:
            ax.text(0.5, 0.5, 'No valid representations', transform=ax.transAxes, ha='center')
            continue
        
        base_reps = np.array(base_reps)
        role_reps = np.array(role_reps)
        
        all_points = np.vstack([base_reps, role_reps])
        pca = PCA(n_components=2)
        pca.fit(all_points)
        
        base_2d = pca.transform(base_reps)
        role_2d = pca.transform(role_reps)
        
        color = SUBJECT_COLORS.get(subject, '#888888')
        display_name = SUBJECT_DISPLAY.get(subject, subject)
        
        n_samples = min(len(base_2d), 100)
        for i in range(n_samples):
            ax.arrow(base_2d[i, 0], base_2d[i, 1],
                    role_2d[i, 0] - base_2d[i, 0],
                    role_2d[i, 1] - base_2d[i, 1],
                    head_width=0.08, head_length=0.08,
                    alpha=0.25, color=color, linewidth=0.5)
        
        ax.scatter(base_2d[:n_samples, 0], base_2d[:n_samples, 1],
                  alpha=0.4, color=color, s=25, label='Base')
        ax.scatter(role_2d[:n_samples, 0], role_2d[:n_samples, 1],
                  alpha=0.4, color=color, marker='^', s=25, label='Roleplay')
        
        base_center = base_2d.mean(axis=0)
        role_center = role_2d.mean(axis=0)
        deflection_norm = np.linalg.norm(role_center - base_center)
        
        ax.arrow(base_center[0], base_center[1],
                role_center[0] - base_center[0],
                role_center[1] - base_center[1],
                head_width=0.2, head_length=0.2,
                color='#C75D4F', linewidth=4.0, label='Centroid Shift')
        
        ax.set_title(f'{display_name}\nDeflection: {deflection_norm:.3f}', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Representation Space Deflection: Base → Roleplay (2D)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(output_base, "deflection_vectors_2d.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  💾 Saved: {output_file}")
    plt.close()


# ======================== Main Function ========================

def main():
    parser = argparse.ArgumentParser(description="MMLU-Redux 2D Deflection Analysis (No accuracy)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference, only generate plots")
    parser.add_argument("--sample_size", type=int, default=100, help="Samples per subject")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    global TEST_SAMPLES
    TEST_SAMPLES = args.sample_size
    
    print("="*80)
    print("MMLU-Redux 2D Deflection Analysis - Pure Visualization")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Subjects: {', '.join(TEST_SUBJECTS)}")
    print(f"Samples per subject: {TEST_SAMPLES}")
    print(f"GPU: {args.gpu_id}")
    print(f"Output: {OUTPUT_BASE}")
    print("="*80)
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    if not args.skip_inference:
        device = f"cuda:{args.gpu_id}"
        print(f"\n🔄 Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        print(f"✓ Model loaded\n")
        
        for subject in TEST_SUBJECTS:
            base_results, role_results = process_subject(subject, model, tokenizer, device, OUTPUT_BASE, debug=args.debug)
            
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n" + "="*60)
    print("Loading data and generating 2D visualization...")
    print("="*60)
    
    viz_data = {}
    
    for subject in TEST_SUBJECTS:
        print(f"\n  Loading {subject}...")
        output_dir = os.path.join(OUTPUT_BASE, subject)
        
        base_hidden_file = os.path.join(output_dir, f"{subject}_base_hidden_states.npz")
        role_hidden_file = os.path.join(output_dir, f"{subject}_roleplay_hidden_states.npz")
        
        hidden_base = []
        hidden_role = []
        
        if os.path.exists(base_hidden_file):
            data = np.load(base_hidden_file, allow_pickle=True)
            hidden_base = [data['hidden_states'][i] for i in range(data['hidden_states'].shape[0])]
            print(f"    Loaded {len(hidden_base)} base hidden states")
        else:
            print(f"    ⚠️ No base hidden states found")
        
        if os.path.exists(role_hidden_file):
            data = np.load(role_hidden_file, allow_pickle=True)
            hidden_role = [data['hidden_states'][i] for i in range(data['hidden_states'].shape[0])]
            print(f"    Loaded {len(hidden_role)} roleplay hidden states")
        else:
            print(f"    ⚠️ No roleplay hidden states found")
        
        viz_data[subject] = {
            'hidden_base': hidden_base,
            'hidden_role': hidden_role,
        }
    
    print(f"\n  📊 Generating 2D deflection visualization...")
    plot_deflection_vectors_2d(viz_data, OUTPUT_BASE)
    
    print(f"\n{'='*80}")
    print(f"✅ ALL DONE!")
    print(f"📁 Output: {OUTPUT_BASE}/deflection_vectors_2d.png")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
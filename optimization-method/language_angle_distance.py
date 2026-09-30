#!/usr/bin/env python3
"""
Spatial Geometry Analysis Script for Four Role-Playing Variants
- Model: Qwen3-14B
- Subject: college_chemistry (from MMLU-Redux)
- Only generates: Pairwise angle heatmap + PCA scatter plot
- No bar charts or related data
"""

import os
import json
import re
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.linalg import svd
from scipy.spatial.distance import cosine
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# ======================== Set Roman Font for Plots ========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

COLORS_HEATMAP = ['#eefbc8', '#fdfd96', '#f8c385', '#f48fb1']
CUSTOM_CMAP = LinearSegmentedColormap.from_list("acl_heatmap", COLORS_HEATMAP)

PCA_COLORS = {
    'en': '#66c2a5', 'zh': '#8da0cb', 'ja': '#e78ac3',
    'ms': '#a6d854', 'de': '#ffd92f', 'fr': '#e5c494',
    'km': '#b3b3b3', 'es': '#fc8d62'
}

# ======================== Configuration ========================
MODEL_PATH = "Qwen/Qwen3-14B"
DATASET_BASE = "mmlu-redux"
SUBJECT = "college_chemistry"
NUM_SAMPLES = 100

LANGUAGES = ["en", "zh", "ja", "ms", "de", "fr", "km", "es"]

LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", 
    "ms": "Malay", "de": "German", "fr": "French",
    "km": "Khmer", "es": "Spanish"
}

LANG_LABELS = {
    "en": "EN", "zh": "ZH", "ja": "JA", 
    "ms": "MS", "de": "DE", "fr": "FR",
    "km": "KM", "es": "ES"
}

ROLEPLAY_PROMPTS = {
    "en": "You are an expert in the field of {subject}. Please answer the following question based on your professional knowledge.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nPlease output only the letter of the correct answer (A, B, C, or D).",
    "zh": "您是{subject}领域的专家。请根据您的专业知识回答以下问题。\n\n问题：{question}\n\n选项：\n{choices_text}\n\n请只输出正确答案的字母（A、B、C或D）。",
    "ja": "あなたは{subject}分野の専門家です。専門知識に基づいて次の質問に答えてください。\n\n質問：{question}\n\n選択肢：\n{choices_text}\n\n正解のアルファベット（A、B、C、D）だけを出力してください。",
    "ms": "Anda seorang pakar dalam bidang {subject}. Sila jawab soalan berikut berdasarkan pengetahuan profesional anda.\n\nSoalan: {question}\n\nPilihan:\n{choices_text}\n\nSila keluarkan hanya huruf jawapan yang betul (A, B, C, atau D).",
    "de": "Sie sind ein Experte auf dem Gebiet {subject}. Bitte beantworten Sie die folgende Frage basierend auf Ihrem Fachwissen.\n\nFrage: {question}\n\nOptionen:\n{choices_text}\n\nBitte geben Sie nur den Buchstaben der richtigen Antwort aus (A, B, C oder D).",
    "fr": "Vous êtes un expert dans le domaine de {subject}. Veuillez répondre à la question suivante en fonction de vos connaissances professionnelles.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nVeuillez indiquer uniquement la lettre de la bonne réponse (A, B, C ou D).",
    "km": "អ្នកគឺជាអ្នកជំនាញក្នុងវិស័យ {subject}។ សូមឆ្លើយសំណួរខាងក្រោមដោយផ្អែកលើចំណេះដឹងជំនាញរបស់អ្នក។\n\nសំណួរ៖ {question}\n\nជម្រើស៖\n{choices_text}\n\nសូមបញ្ចេញតែអក្សរនៃចម្លើយត្រឹមត្រូវប៉ុណ្ណោះ (A, B, C, ឬ D) ។",
    "es": "Usted es un experto en el campo de {subject}. Por favor, responda la siguiente pregunta basándose en su conocimiento profesional.\n\nPregunta: {question}\n\nOpciones:\n{choices_text}\n\nPor favor, genere solo la letra de la respuesta correcta (A, B, C o D)."
}

# ======================== Data Loading ========================
def load_arrow_data(subject, num_samples=None):
    import pyarrow as pa
    arrow_file = f"{DATASET_BASE}/{subject}/data-00000-of-00001.arrow"
    
    if not os.path.exists(arrow_file):
        print(f"ERROR: File not found: {arrow_file}")
        return None
    
    try:
        with open(arrow_file, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
        
        data = []
        for i in range(min(len(table), num_samples) if num_samples else len(table)):
            row = {col: table[col][i].as_py() for col in table.column_names}
            choices_str = row.get('choices', '[]')
            if isinstance(choices_str, str):
                try:
                    choices = eval(choices_str)
                except:
                    choices = [c.strip().strip('"\'' ) for c in choices_str.strip('[]').split(',')]
            else:
                choices = choices_str
            
            answer_idx = row.get('answer', 0)
            answer_letter = chr(65 + answer_idx)
            data.append({
                "idx": i,
                "question": row.get('question', ''),
                "choices": choices,
                "answer_idx": answer_idx,
                "answer_letter": answer_letter,
            })
        
        print(f"Loaded {len(data)} samples from {subject}")
        return data
    except Exception as e:
        print(f"Error loading {subject}: {e}")
        return None

# ======================== Model Loading ========================
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer

# ======================== Prompt Building ========================
def build_roleplay_prompt(question, choices, subject, lang="en"):
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    subject_display = subject.replace('_', ' ').title()
    template = ROLEPLAY_PROMPTS.get(lang, ROLEPLAY_PROMPTS["en"])
    return template.format(
        question=question, 
        choices_text=choices_text, 
        subject=subject_display
    )

# ======================== Hidden State Extraction ========================
@torch.no_grad()
def extract_last_layer_hidden(model, tokenizer, texts, batch_size=1):
    all_hidden_states = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        vocab_size = model.config.vocab_size
        if (inputs["input_ids"] >= vocab_size).any():
            print(f"⚠️ Warning: Token ID exceeds vocab! Max ID: {inputs['input_ids'].max().item()}, Vocab size: {vocab_size}")
            inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)
        
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        attention_mask = inputs["attention_mask"]
        for j in range(hidden_states.shape[0]):
            mask = attention_mask[j].unsqueeze(-1).float()
            mask_sum = mask.sum()
            
            if mask_sum == 0:
                print(f"⚠️ Warning: Sample {j} has all padding")
                pooled = hidden_states[j].mean(dim=0)
            else:
                pooled = (hidden_states[j] * mask).sum(dim=0) / mask_sum
                
            all_hidden_states.append(pooled.cpu().numpy())
    
    return np.array(all_hidden_states)

# ======================== Cache Helpers ========================
def get_cache_path(subject, lang):
    cache_dir = "hidden_state_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{subject}_{lang}.npy")

def load_cached_representations(subject, lang):
    cache_path = get_cache_path(subject, lang)
    if os.path.exists(cache_path):
        try:
            reps = np.load(cache_path)
            print(f"  Loaded cached representations for {LANG_NAMES[lang]} ({lang}), shape: {reps.shape}")
            return reps
        except Exception as e:
            print(f"  Failed to load cache for {lang}: {e}")
            return None
    return None

def save_cached_representations(subject, lang, representations):
    cache_path = get_cache_path(subject, lang)
    try:
        np.save(cache_path, representations)
        print(f"  Cached representations for {LANG_NAMES[lang]} ({lang}), shape: {representations.shape}")
    except Exception as e:
        print(f"  Failed to cache for {lang}: {e}")

# ======================== Spatial Geometry Analysis ========================
def compute_pairwise_angle_matrix(representations_by_lang):
    lang_names = list(representations_by_lang.keys())
    n = len(lang_names)
    angle_matrix = np.zeros((n, n))
    
    centers = {}
    for lang, reps in representations_by_lang.items():
        centers[lang] = np.mean(reps, axis=0)
    
    for i, lang1 in enumerate(lang_names):
        for j, lang2 in enumerate(lang_names):
            if i == j:
                angle_matrix[i, j] = 0
            else:
                cos_sim = np.dot(centers[lang1], centers[lang2]) / (
                    np.linalg.norm(centers[lang1]) * np.linalg.norm(centers[lang2]) + 1e-8
                )
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                angle_matrix[i, j] = np.degrees(np.arccos(cos_sim))
    
    return angle_matrix, lang_names

# ======================== Visualization ========================
def visualize_heatmap_and_pca(representations_by_lang):
    
    # ==========================================
    # Prepare data
    # ==========================================
    
    angle_matrix, lang_names = compute_pairwise_angle_matrix(representations_by_lang)
    lang_labels = [LANG_LABELS.get(lang, lang) for lang in lang_names]
    
    all_reps = []
    for lang, reps in representations_by_lang.items():
        all_reps.extend(reps)
    
    if len(all_reps) >= 2:
        pca = PCA(n_components=2)
        projected = pca.fit_transform(np.array(all_reps))
        
        pca_by_lang = {}
        start_idx = 0
        for lang, reps in representations_by_lang.items():
            end_idx = start_idx + len(reps)
            pca_by_lang[lang] = projected[start_idx:end_idx]
            start_idx = end_idx
        
        pca_centroids = {}
        pca_covs = {}
        for lang, points in pca_by_lang.items():
            pca_centroids[lang] = np.mean(points, axis=0)
            pca_covs[lang] = np.cov(points.T) if len(points) > 1 else np.eye(2) * 0.1
    
    # ==========================================
    # Save data
    # ==========================================
    data_dir = "spatial_analysis_data"
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "angle_matrix.npy"), angle_matrix)
    with open(os.path.join(data_dir, "lang_labels.txt"), 'w') as f:
        f.write("\n".join(lang_labels))
    
    pca_data = {}
    for lang, points in pca_by_lang.items():
        pca_data[lang] = {
            'points': points.tolist(),
            'centroid': pca_centroids[lang].tolist(),
            'covariance': pca_covs[lang].tolist()
        }
    import json
    with open(os.path.join(data_dir, "pca_data.json"), 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    print(f"\nData saved to: {data_dir}/")
    print("  - angle_matrix.npy: Angle matrix")
    print("  - lang_labels.txt: Language labels")
    print("  - pca_data.json: PCA data (points, centroids, covariance)")
    
    # ==========================================
    # Figure 1: Pairwise Angle Heatmap
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    fig1.patch.set_facecolor('white')
    
    im = ax1.imshow(angle_matrix, cmap=CUSTOM_CMAP, aspect='auto')
    
    ax1.set_xticks(np.arange(len(lang_labels)))
    ax1.set_yticks(np.arange(len(lang_labels)))
    ax1.set_xticklabels(lang_labels, fontfamily='serif', fontweight='bold', fontsize=12)
    ax1.set_yticklabels(lang_labels, fontfamily='serif', fontweight='bold', fontsize=12)
    ax1.set_title('Pairwise Angle Heatmap', fontfamily='serif', fontweight='bold', pad=15, fontsize=14)
    
    for i in range(len(lang_labels)):
        for j in range(len(lang_labels)):
            if i != j:
                ax1.text(j, i, f'{angle_matrix[i, j]:.1f}°',
                        ha="center", va="center", color="black", fontsize=10, fontfamily='serif')
    
    cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Angular Distance (degrees)', rotation=270, labelpad=15,
                   fontfamily='serif', fontweight='bold', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    os.makedirs("spatial_analysis_figures", exist_ok=True)
    fig1.savefig('spatial_analysis_figures/heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: spatial_analysis_figures/heatmap.png")
    
    # ==========================================
    # Figure 2: PCA Scatter Plot
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    fig2.patch.set_facecolor('white')
    
    ax2.set_title('PCA Projection of Language Representations', 
                  fontfamily='serif', fontweight='bold', fontsize=14)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                   fontfamily='serif', fontweight='bold', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                   fontfamily='serif', fontweight='bold', fontsize=11)
    ax2.axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax2.axvline(0, color='grey', lw=0.5, alpha=0.5)
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    for lang, points in pca_by_lang.items():
        color = PCA_COLORS.get(lang, '#b3b3b3')
        centroid = pca_centroids[lang]
        cov = pca_covs[lang]
        
        ax2.scatter(points[:, 0], points[:, 1], c=color, s=20, alpha=0.6,
                   edgecolors='grey', linewidth=0.3, label=LANG_NAMES[lang])
        
        if len(points) > 2:
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * np.sqrt(vals) * 1.5
            ell = Ellipse(xy=centroid, width=w, height=h, angle=theta,
                         facecolor='none', edgecolor=color, linestyle='--', 
                         alpha=0.7, linewidth=1.5)
            ax2.add_patch(ell)
        
        short_label = LANG_LABELS.get(lang, lang[:2].upper())
        ax2.text(centroid[0], centroid[1], short_label, fontsize=10,
                fontfamily='serif', fontweight='bold', color='black',
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2',
                                                   facecolor='white', alpha=0.7))
    
    all_centroids = np.array(list(pca_centroids.values()))
    global_centroid = np.mean(all_centroids, axis=0)
    ax2.scatter(global_centroid[0], global_centroid[1], marker='*', s=300,
               c='#d6604d', edgecolor='black', linewidth=1.5, zorder=5, label='Global Centroid')
    
    ax2.legend(loc='upper right', prop={'family': 'serif', 'size': 9})
    ax2.set_aspect('equal')
    
    fig2.savefig('spatial_analysis_figures/pca_projection.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: spatial_analysis_figures/pca_projection.png")

# ======================== Main Experiment ========================
def main():
    print("="*80)
    print("Spatial Geometry Analysis (Lightweight Version)")
    print("Generates: Pairwise Angle Heatmap + PCA Scatter Plot")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: MMLU-Redux - {SUBJECT}")
    print(f"Languages: {LANGUAGES}")
    print("="*80)
    
    print("\nLoading data...")
    samples = load_arrow_data(SUBJECT, NUM_SAMPLES)
    if samples is None:
        return
    print(f"Loaded {len(samples)} samples")
    
    model, tokenizer = load_model()
    
    print("\nExtracting hidden states for each language...")
    representations_by_lang = {}
    
    for lang in LANGUAGES:
        print(f"  Processing {LANG_NAMES[lang]} ({lang})...")
        
        cached_reps = load_cached_representations(SUBJECT, lang)
        if cached_reps is not None:
            representations_by_lang[lang] = cached_reps
            continue
        
        texts = []
        for sample in samples:
            prompt = build_roleplay_prompt(
                sample['question'], 
                sample['choices'], 
                SUBJECT, 
                lang
            )
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            texts.append(formatted)
        
        reps = extract_last_layer_hidden(model, tokenizer, texts, batch_size=1)
        representations_by_lang[lang] = reps
        print(f"    Shape: {reps.shape}")
        
        save_cached_representations(SUBJECT, lang, reps)
    
    print("\nGenerating visualizations...")
    visualize_heatmap_and_pca(representations_by_lang)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
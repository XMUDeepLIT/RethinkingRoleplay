#!/usr/bin/env python3
"""
MMLU Multilingual Test Script - Supports Chinese/Japanese/Malay/German/French/Khmer/Spanish

- 57 topics

- Base/Roleplay modes

- Supports seven languages: Chinese (zh), Japanese (ja), Malay (ms), German (de), French (fr), Khmer (km), Spanish (es)
"""

import os
import json
import re
import time
import sys
import torch
from pathlib import Path
from multiprocessing import Process
from vllm import LLM, SamplingParams
from datasets import load_dataset


TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.55
SAVE_INTERVAL = 10          # Save every N samples
BATCH_SIZE = 128            # Batch inference size

RESULT_ROOT = "mmlu_result"


TEST_SUBJECTS = [
    # STEM
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    "medical_genetics", "virology",
    # Humanities
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "professional_law", "world_religions",
    # Social Sciences
    "econometrics", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
    "human_aging", "human_sexuality", "marketing", "professional_accounting",
    "professional_psychology", "sociology",
    # Other
    "business_ethics", "clinical_knowledge", "global_facts", "management",
    "miscellaneous", "nutrition", "professional_medicine", "public_relations",
    "security_studies", "us_foreign_policy"
]


AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6,7]


MODELS = [
    (os.environ.get("MODEL_1_PATH", "./models/Qwen3-4B"), "qwen3_4b"),
    (os.environ.get("MODEL_2_PATH", "./models/Qwen3-8B"), "qwen3_8b"),
    (os.environ.get("MODEL_3_PATH", "./models/Qwen3-14B"), "qwen3_14b"),
]


LANGUAGES = [
    ("en", "_en", "英文"),
    ("zh", "_zh", "中文"),
    ("ja", "_ja", "日语"),
    ("ms", "_ms", "马来语"),
    ("de", "_de", "德语"),
    ("fr", "_fr", "法语"),
    ("km", "_km", "高棉语"),
    ("es", "_es", "西班牙语"),
]

# Base 
BASE_TEMPLATES = {
    "en": "Answer the following multiple-choice question. Choose the correct option.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nPlease output only the number of the correct answer (0, 1, 2, or 3). Do not output any other text.",
    "zh": "回答以下选择题。选择正确的选项。\n\n问题：{question}\n\n选项：\n{choices_text}\n\n请只输出正确答案的编号（0、1、2或3）。不要输出任何其他文本。",
    "ja": "次の多肢選択問題に答えてください。正しい選択肢を選んでください。\n\n質問：{question}\n\n選択肢：\n{choices_text}\n\n正解の番号（0、1、2、3）だけを出力してください。他のテキストは出力しないでください。",
    "ms": "Jawab soalan pelbagai pilihan berikut. Pilih pilihan yang betul.\n\nSoalan: {question}\n\nPilihan:\n{choices_text}\n\nSila keluarkan hanya nombor jawapan yang betul (0, 1, 2, atau 3). Jangan keluarkan teks lain.",
    "de": "Beantworten Sie die folgende Multiple-Choice-Frage. Wählen Sie die richtige Option.\n\nFrage: {question}\n\nOptionen:\n{choices_text}\n\nBitte geben Sie nur die Nummer der richtigen Antwort aus (0, 1, 2 oder 3). Geben Sie keinen anderen Text aus.",
    "fr": "Répondez à la question à choix multiples suivante. Choisissez la bonne option.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nVeuillez indiquer uniquement le numéro de la bonne réponse (0, 1, 2 ou 3). Ne donnez aucun autre texte.",
    "km": "សូមឆ្លើយសំណួរជ្រើសរើសច្រើនខាងក្រោម។ ជ្រើសរើសជម្រើសត្រឹមត្រូវ។\n\nសំណួរ៖ {question}\n\nជម្រើស៖\n{choices_text}\n\nសូមបញ្ចេញតែលេខនៃចម្លើយត្រឹមត្រូវប៉ុណ្ណោះ (0, 1, 2, ឬ 3) ។ កុំបញ្ចេញអត្ថបទផ្សេងទៀត។",
    "es": "Responda la siguiente pregunta de opción múltiple. Elija la opción correcta.\n\nPregunta: {question}\n\nOpciones:\n{choices_text}\n\nPor favor, genere solo el número de la respuesta correcta (0, 1, 2 o 3). No genere ningún otro texto."
}

# Roleplay 
ROLEPLAY_TEMPLATES = {
    "en": "You are an expert in the field of {subject_display}. Please answer the following question based on your professional knowledge.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nPlease output only the number of the correct answer (0, 1, 2, or 3). Do not output any other text.",
    "zh": "您是{subject_display}领域的专家。请根据您的专业知识回答以下问题。\n\n问题：{question}\n\n选项：\n{choices_text}\n\n请只输出正确答案的编号（0、1、2或3）。不要输出任何其他文本。",
    "ja": "あなたは{subject_display}分野の専門家です。専門知識に基づいて次の質問に答えてください。\n\n質問：{question}\n\n選択肢：\n{choices_text}\n\n正解の番号（0、1、2、3）だけを出力してください。他のテキストは出力しないでください。",
    "ms": "Anda seorang pakar dalam bidang {subject_display}. Sila jawab soalan berikut berdasarkan pengetahuan profesional anda.\n\nSoalan: {question}\n\nPilihan:\n{choices_text}\n\nSila keluarkan hanya nombor jawapan yang betul (0, 1, 2, atau 3). Jangan keluarkan teks lain.",
    "de": "Sie sind ein Experte auf dem Gebiet {subject_display}. Bitte beantworten Sie die folgende Frage basierend auf Ihrem Fachwissen.\n\nFrage: {question}\n\nOptionen:\n{choices_text}\n\nBitte geben Sie nur die Nummer der richtigen Antwort aus (0, 1, 2 oder 3). Geben Sie keinen anderen Text aus.",
    "fr": "Vous êtes un expert dans le domaine de {subject_display}. Veuillez répondre à la question suivante en fonction de vos connaissances professionnelles.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nVeuillez indiquer uniquement le numéro de la bonne réponse (0, 1, 2 ou 3). Ne donnez aucun autre texte.",
    "km": "អ្នកគឺជាអ្នកជំនាញក្នុងវិស័យ {subject_display}។ សូមឆ្លើយសំណួរខាងក្រោមដោយផ្អែកលើចំណេះដឹងជំនាញរបស់អ្នក។\n\nសំណួរ៖ {question}\n\nជម្រើស៖\n{choices_text}\n\nសូមបញ្ចេញតែលេខនៃចម្លើយត្រឹមត្រូវប៉ុណ្ណោះ (0, 1, 2, ឬ 3) ។ កុំបញ្ចេញអត្ថបទផ្សេងទៀត។",
    "es": "Usted es un experto en el campo de {subject_display}. Por favor, responda la siguiente pregunta basándose en su conocimiento profesional.\n\nPregunta: {question}\n\nOpciones:\n{choices_text}\n\nPor favor, genere solo el número de la respuesta correcta (0, 1, 2 o 3). No genere ningún otro texto."
}

def build_prompt(question, choices, mode, subject=None, lang="en"):
    
    choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
    
    if mode == "base":
        template = BASE_TEMPLATES.get(lang, BASE_TEMPLATES["en"])
        return template.format(question=question, choices_text=choices_text)
    else:  # roleplay
        template = ROLEPLAY_TEMPLATES.get(lang, ROLEPLAY_TEMPLATES["en"])
        subject_display = subject.replace('_', ' ').title()
        return template.format(question=question, choices_text=choices_text, subject_display=subject_display)


def load_mmlu_data(subject):
    
    try:
        dataset = load_dataset("./datasets/mmlu", subject, split="test")
        return list(dataset)
    except Exception as e:
        print(f"  ❌ Error loading {subject}: {e}")
        return None


def extract_answer(response):
    if not response:
        return None
    
    
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.strip()
    
    
    patterns = [
        r'\b([0-3])\b',
        r'[Aa]nswer[:\s]*([0-3])',
        r'[Oo]ption\s*([0-3])',
        r'[Cc]hoice\s*([0-3])',
        r'[答答案案][案是:：\s]*([0-3])',
        r'correct answer is ([0-3])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
    
    return None


def batch_generate(llm, tokenizer, prompts, temperature=0.0, max_tokens=2048):
   
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    formatted_list = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(formatted_list, sampling_params)
    return [o.outputs[0].text for o in outputs]


def load_existing_results(detail_file):
    
    existing = {}
    if os.path.exists(detail_file):
        with open(detail_file, "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing[item["idx"]] = item
        print(f"  🔄 Loaded {len(existing)} existing results")
    return existing

def save_result(detail_file, result):
    
    with open(detail_file, "a", encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def save_summary(summary_file, summary):
    
    with open(summary_file, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_on_gpu(gpu_id, subjects_list, model_path, mode, lang, lang_name, output_base):
    """
    mode: "base" or "roleplay"
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    
    print(f"\n🔄 GPU {gpu_id}: Starting {mode} mode, {len(subjects_list)} subjects, lang={lang_name}")
    
    # 加载模型
    print(f"  Loading model on GPU {gpu_id}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    tokenizer = llm.get_tokenizer()
    print(f"  ✓ Model loaded on GPU {gpu_id}")
    
    for subject in subjects_list:
        print(f"\n{'='*60}")
        print(f"GPU {gpu_id} - [{mode.upper()}] Subject: {subject} ({lang_name})")
        print(f"{'='*60}")
        

        test_data = load_mmlu_data(subject)
        if not test_data:
            print(f"  ❌ Failed to load {subject}, skipping...")
            continue
        
        total_samples = len(test_data)
        print(f"  ✓ Loaded {total_samples} questions")
        

        subject_dir = f"{output_base}/{subject}"
        os.makedirs(subject_dir, exist_ok=True)
        summary_file = f"{subject_dir}/{mode}_{lang}.json"
        detail_file = f"{subject_dir}/{mode}_{lang}_details.jsonl"
        

        existing_results = load_existing_results(detail_file)
        processed_indices = set(existing_results.keys())
        pending_indices = [i for i in range(total_samples) if i not in processed_indices]
        
        if not pending_indices:
            print(f"  ✅ All {total_samples} samples already processed")
            continue
        
        print(f"  📍 Resuming: {len(processed_indices)} done, {len(pending_indices)} pending")
        

        correct_count = sum(1 for r in existing_results.values() if r.get("correct", False))
        

        for batch_start in range(0, len(pending_indices), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(pending_indices))
            batch_indices = pending_indices[batch_start:batch_end]
            

            prompts = []
            batch_items = []
            for idx in batch_indices:
                item = test_data[idx]
                prompt = build_prompt(item["question"], item["choices"], mode, subject, lang)
                prompts.append(prompt)
                batch_items.append(item)
            

            responses = batch_generate(llm, tokenizer, prompts, TEMPERATURE)
            

            for idx, item, response in zip(batch_indices, batch_items, responses):
                pred = extract_answer(response)
                true_answer = item["answer"]
                correct = (pred == true_answer) if pred is not None else False
                
                if correct:
                    correct_count += 1
                
                result = {
                    "idx": idx,
                    "correct": correct,
                    "pred": pred,
                    "true": true_answer,
                    "question": item["question"][:200],
                    "response": response[:500]
                }
                save_result(detail_file, result)
            

            processed_count = len(processed_indices) + batch_end
            acc = correct_count / processed_count
            print(f"  Progress: {processed_count}/{total_samples}, Acc: {acc:.2%}")
            

            if processed_count % SAVE_INTERVAL == 0 or processed_count == total_samples:
                interim_summary = {
                    "mode": mode,
                    "subject": subject,
                    "language": lang_name,
                    "gpu": gpu_id,
                    "total": total_samples,
                    "processed": processed_count,
                    "correct": correct_count,
                    "accuracy": acc,
                }
                save_summary(summary_file, interim_summary)
        

        final_accuracy = correct_count / total_samples
        final_summary = {
            "mode": mode,
            "subject": subject,
            "language": lang_name,
            "gpu": gpu_id,
            "total": total_samples,
            "correct": correct_count,
            "accuracy": final_accuracy,
        }
        save_summary(summary_file, final_summary)
        
        print(f"  ✅ {mode.upper()} - {subject} ({lang_name}): {correct_count}/{total_samples} = {final_accuracy:.2%}")


def run_model_language(model_path, model_name, mode, lang, suffix, lang_name):
  
    output_base = os.path.join(RESULT_ROOT, f"{model_name}{suffix}", mode)
    print(f"\n{'='*80}")
    print(f"🚀 Model: {model_name}, Mode: {mode.upper()}, Language: {lang_name}({lang})")
    print(f"📁 Output: {output_base}")
    print(f"{'='*80}")
    
    Path(output_base).mkdir(parents=True, exist_ok=True)
    

    subjects_per_gpu = [[] for _ in range(len(AVAILABLE_GPUS))]
    for idx, subject in enumerate(TEST_SUBJECTS):
        subjects_per_gpu[idx % len(AVAILABLE_GPUS)].append(subject)
    
    processes = []
    for gpu_idx, gpu_id in enumerate(AVAILABLE_GPUS):
        subjects = subjects_per_gpu[gpu_idx]
        if not subjects:
            continue
        
        p = Process(target=run_on_gpu, 
                   args=(gpu_id, subjects, model_path, mode, lang, lang_name, output_base))
        p.start()
        processes.append(p)
        print(f"✅ Started process for GPU {gpu_id} with {len(subjects)} subjects")
        time.sleep(5) 
    
    for p in processes:
        p.join()
    
    print(f"\n✅ Completed: {model_name} ({mode.upper()}) - {lang_name}")


def main():
    print("="*80)
    print("🧪 MMLU Multi-Language Test")
    print("   Modes: Base / Roleplay")
    print("   Languages: English, Chinese, Japanese, Malay, German, French, Khmer, Spanish")
    print("="*80)
    print(f"Total subjects: {len(TEST_SUBJECTS)}")
    print(f"Models: {[name for _, name in MODELS]}")
    print(f"Languages: {[name for _, _, name in LANGUAGES]}")
    print(f"GPUs: {AVAILABLE_GPUS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Result directory: {RESULT_ROOT}")
    print("="*80)
    

    Path(RESULT_ROOT).mkdir(parents=True, exist_ok=True)
    

    for model_path, model_name in MODELS:
        for lang, suffix, lang_name in LANGUAGES:
            for mode in ["base", "roleplay"]:
                run_model_language(model_path, model_name, mode, lang, suffix, lang_name)
                
 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(5)
    
    print("\n" + "="*80)
    print("🎉 All tests completed!")
    print(f"📁 All results saved under: {RESULT_ROOT}/")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

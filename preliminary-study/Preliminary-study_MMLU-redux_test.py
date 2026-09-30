#!/usr/bin/env python3
"""
MMLU-Redux Multilingual Test Script - Supports Chinese/Japanese/Malay/German/French/Khmer/Spanish

- 30 topics, 100 samples per topic

- Base/Roleplay modes

- Supports seven languages: Chinese (zh), Japanese (ja), Malay (ms), German (de), French (fr), Khmer (km), Spanish (es)
"""

import os
import json
import re
import time
import sys
import torch
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue
import pyarrow as pa
from vllm import LLM, SamplingParams

# ======================== Configuration Parameters ========================
DATASET_BASE = os.environ.get("MMLU_REDUX_DATA", "./datasets/mmlu-redux")
OUTPUT_BASE = os.environ.get("MMLU_REDUX_OUTPUT", "./mmlu_redux_results")
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.35
SAVE_INTERVAL = 10          # Save every N samples
BATCH_SIZE = 128            # Batch inference size
TEST_SAMPLES_PER_SUBJECT = 100

# List of test subjects
TEST_SUBJECTS = [
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

# Multi-GPU configuration (automatically detect available GPUs)
AVAILABLE_GPUS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]

# List of models to test (path, identifier)

MODELS = [
    (os.environ.get("MODEL_1_PATH", "./models/Qwen3-4B"), "qwen3_4b"),
    (os.environ.get("MODEL_2_PATH", "./models/Qwen3-8B"), "qwen3_8b"),
    (os.environ.get("MODEL_3_PATH", "./models/Qwen3-14B"), "qwen3_14b"),
]

# Languages to test (code, output suffix, language name)
LANGUAGES = [
    ("en", "", "英语"), 
    ("zh", "_zh", "中文"),
    ("ja", "_ja", "日语"),
    ("ms", "_ms", "马来语"),
    ("de", "_de", "德语"),
    ("fr", "_fr", "法语"),
    ("km", "_km", "高棉语"),
    ("es", "_es", "西班牙语"),
]

PROMPT_TEMPLATES_FILE = "prompt_templates.json"

# ======================== Multilingual Prompt Templates ========================
def generate_prompt_templates():
    """Multilingual Prompt Templates"""
    templates = {
        "base": {
            "en": "Answer the following multiple-choice question. Choose the correct option.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nPlease output only the letter of the correct answer (A, B, C, or D). Do not output any other text.",
            "zh": "回答以下选择题。选择正确的选项。\n\n问题：{question}\n\n选项：\n{choices_text}\n\n请只输出正确答案的字母（A、B、C或D）。不要输出任何其他文本。",
            "ja": "次の多肢選択問題に答えてください。正しい選択肢を選んでください。\n\n質問：{question}\n\n選択肢：\n{choices_text}\n\n正解のアルファベット（A、B、C、D）だけを出力してください。他のテキストは出力しないでください。",
            "ms": "Jawab soalan pelbagai pilihan berikut. Pilih pilihan yang betul.\n\nSoalan: {question}\n\nPilihan:\n{choices_text}\n\nSila keluarkan hanya huruf jawapan yang betul (A, B, C, atau D). Jangan keluarkan teks lain.",
            "de": "Beantworten Sie die folgende Multiple-Choice-Frage. Wählen Sie die richtige Option.\n\nFrage: {question}\n\nOptionen:\n{choices_text}\n\nBitte geben Sie nur den Buchstaben der richtigen Antwort aus (A, B, C oder D). Geben Sie keinen anderen Text aus.",
            "fr": "Répondez à la question à choix multiples suivante. Choisissez la bonne option.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nVeuillez indiquer uniquement la lettre de la bonne réponse (A, B, C ou D). Ne donnez aucun autre texte.",
            "km": "សូមឆ្លើយសំណួរជ្រើសរើសច្រើនខាងក្រោម។ ជ្រើសរើសជម្រើសត្រឹមត្រូវ។\n\nសំណួរ៖ {question}\n\nជម្រើស៖\n{choices_text}\n\nសូមបញ្ចេញតែអក្សរនៃចម្លើយត្រឹមត្រូវប៉ុណ្ណោះ (A, B, C, ឬ D) ។ កុំបញ្ចេញអត្ថបទផ្សេងទៀត។",
            "es": "Responda la siguiente pregunta de opción múltiple. Elija la opción correcta.\n\nPregunta: {question}\n\nOpciones:\n{choices_text}\n\nPor favor, genere solo la letra de la respuesta correcta (A, B, C o D). No genere ningún otro texto."
        },
        "roleplay": {
            "en": "You are an expert in the field of {subject_display}. Please answer the following question based on your professional knowledge.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nPlease output only the letter of the correct answer (A, B, C, or D). Do not output any other text.",
            "zh": "您是{subject_display}领域的专家。请根据您的专业知识回答以下问题。\n\n问题：{question}\n\n选项：\n{choices_text}\n\n请只输出正确答案的字母（A、B、C或D）。不要输出任何其他文本。",
            "ja": "あなたは{subject_display}分野の専門家です。専門知識に基づいて次の質問に答えてください。\n\n質問：{question}\n\n選択肢：\n{choices_text}\n\n正解のアルファベット（A、B、C、D）だけを出力してください。他のテキストは出力しないでください。",
            "ms": "Anda seorang pakar dalam bidang {subject_display}. Sila jawab soalan berikut berdasarkan pengetahuan profesional anda.\n\nSoalan: {question}\n\nPilihan:\n{choices_text}\n\nSila keluarkan hanya huruf jawapan yang betul (A, B, C, atau D). Jangan keluarkan teks lain.",
            "de": "Sie sind ein Experte auf dem Gebiet {subject_display}. Bitte beantworten Sie die folgende Frage basierend auf Ihrem Fachwissen.\n\nFrage: {question}\n\nOptionen:\n{choices_text}\n\nBitte geben Sie nur den Buchstaben der richtigen Antwort aus (A, B, C oder D). Geben Sie keinen anderen Text aus.",
            "fr": "Vous êtes un expert dans le domaine de {subject_display}. Veuillez répondre à la question suivante en fonction de vos connaissances professionnelles.\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nVeuillez indiquer uniquement la lettre de la bonne réponse (A, B, C ou D). Ne donnez aucun autre texte.",
            "km": "អ្នកគឺជាអ្នកជំនាញក្នុងវិស័យ {subject_display}។ សូមឆ្លើយសំណួរខាងក្រោមដោយផ្អែកលើចំណេះដឹងជំនាញរបស់អ្នក។\n\nសំណួរ៖ {question}\n\nជម្រើស៖\n{choices_text}\n\nសូមបញ្ចេញតែអក្សរនៃចម្លើយត្រឹមត្រូវប៉ុណ្ណោះ (A, B, C, ឬ D) ។ កុំបញ្ចេញអត្ថបទផ្សេងទៀត។",
            "es": "Usted es un experto en el campo de {subject_display}. Por favor, responda la siguiente pregunta basándose en su conocimiento profesional.\n\nPregunta: {question}\n\nOpciones:\n{choices_text}\n\nPor favor, genere solo la letra de la respuesta correcta (A, B, C o D). No genere ningún otro texto."
        }
    }
    with open(PROMPT_TEMPLATES_FILE, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)
    return templates

def load_prompt_templates():
    """Load prompt template file, generate if it doesn't exist"""
    if not os.path.exists(PROMPT_TEMPLATES_FILE):
        print(f"⚠️ Template file does not exist, generating {PROMPT_TEMPLATES_FILE} ...")
        return generate_prompt_templates()
    with open(PROMPT_TEMPLATES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

PROMPTS = load_prompt_templates()

def build_prompt_base(question, choices, lang="en"):
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    template = PROMPTS["base"].get(lang, PROMPTS["base"]["en"])
    return template.format(question=question, choices_text=choices_text)

def build_prompt_roleplay(question, choices, subject, lang="en"):
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    subject_display = subject.replace('_', ' ').title()
    template = PROMPTS["roleplay"].get(lang, PROMPTS["roleplay"]["en"])
    return template.format(question=question, choices_text=choices_text, subject_display=subject_display)

# ======================== Data Loading ========================
def load_mmlu_redux_arrow(subject, num_samples=None):
    arrow_file = f"{DATASET_BASE}/{subject}/data-00000-of-00001.arrow"
    if not os.path.exists(arrow_file):
        print(f"  ❌ File not found: {arrow_file}")
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
                    choices = [c.strip().strip('"\'') for c in choices_str.strip('[]').split(',')]
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
                "error_type": row.get('error_type', 'unknown'),
                "source": row.get('source', subject),
            })
        print(f"  ✓ Loaded {len(data)} samples from {subject}")
        return data
    except Exception as e:
        print(f"  ❌ Error loading {subject}: {e}")
        return None

def extract_answer(response):
    if not response:
        return None
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.strip()
    patterns = [
        r'\b([A-D])\b',
        r'[Aa]nswer:\s*([A-D])',
        r'[Oo]ption\s*([A-D])',
        r'([A-D])\.',
        r'[答答案案][案是:：\s]*([A-D])',
        r'correct answer is ([A-D])',
        r'is ([A-D])\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None

# ======================== Incremental Save and Resume ========================
def load_existing_results(result_file):
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results = data.get('results', [])
                completed_indices = {r['idx'] for r in results}
                print(f"  🔄 Found existing results: {len(results)} samples completed")
                return results, completed_indices
        except Exception as e:
            print(f"  ⚠️ Failed to load existing results: {e}")
            return [], set()
    return [], set()

def save_results_incremental(output_file, results, mode, subject, is_partial=True):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct_count = sum(1 for r in results if r.get('correct', False))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "mode": mode,
            "subject": subject,
            "timestamp": datetime.now().isoformat(),
            "is_partial": is_partial,
            "total_samples": len(results),
            "correct_count": correct_count,
            "accuracy": correct_count / len(results) if results else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved {len(results)} results to {output_file}")

# ======================== Batch Processing Functions ========================
def process_batch_base(items, subject, llm, tokenizer, lang):
    prompts = []
    for item in items:
        prompt = build_prompt_base(item['question'], item['choices'], lang)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=2048)
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for item, output in zip(items, outputs):
        response = output.outputs[0].text
        predicted = extract_answer(response)
        is_correct = (predicted == item['answer_letter']) if predicted else False
        result_item = {
            "idx": item['idx'],
            "question": item['question'],
            "choices": item['choices'],
            "ground_truth": item['answer_letter'],
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:500],
        }
        results.append(result_item)
    return results

def process_batch_roleplay(items, subject, llm, tokenizer, lang):
    prompts = []
    for item in items:
        prompt = build_prompt_roleplay(item['question'], item['choices'], subject, lang)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=2048)
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for item, output in zip(items, outputs):
        response = output.outputs[0].text
        predicted = extract_answer(response)
        is_correct = (predicted == item['answer_letter']) if predicted else False
        result_item = {
            "idx": item['idx'],
            "question": item['question'],
            "choices": item['choices'],
            "ground_truth": item['answer_letter'],
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:500],
        }
        results.append(result_item)
    return results

def run_mode_with_resume_batch(test_data, subject, mode_name, batch_process_func, llm, tokenizer, lang, output_base, save_interval, batch_size):
    result_file = f"{output_base}/{subject}_{mode_name}.json"
    existing_results, completed_indices = load_existing_results(result_file)
    results = existing_results.copy()
    pending_items = [item for item in test_data if item['idx'] not in completed_indices]
    if not pending_items:
        correct_count = sum(1 for r in results if r.get('correct', False))
        print(f"  ✅ All {len(test_data)} samples already completed for {mode_name}")
        return results, correct_count
    print(f"  📍 Resuming {mode_name}: {len(pending_items)}/{len(test_data)} samples pending")
    correct_count = sum(1 for r in results if r.get('correct', False))
    total_processed = len(results)
    for batch_start in range(0, len(pending_items), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_items))
        batch_items = pending_items[batch_start:batch_end]
        try:
            batch_results = batch_process_func(batch_items, subject, llm, tokenizer, lang)
            for result in batch_results:
                if result['correct']:
                    correct_count += 1
                results.append(result)
                total_processed += 1
            print(f"  Processed batch {batch_start//batch_size + 1}/{(len(pending_items)-1)//batch_size + 1}, "
                  f"total: {total_processed}/{len(test_data)} samples, accuracy: {correct_count/total_processed:.2%}")
            if total_processed % save_interval == 0 or total_processed == len(test_data):
                save_results_incremental(result_file, results, mode_name, subject, is_partial=(total_processed < len(test_data)))
        except Exception as e:
            print(f"  ❌ Error processing batch starting at {batch_start}: {e}")
            for item in batch_items:
                error_item = {"idx": item['idx'], "error": str(e), "correct": False,
                              "question": item['question'], "ground_truth": item['answer_letter']}
                results.append(error_item)
                total_processed += 1
    save_results_incremental(result_file, results, mode_name, subject, is_partial=False)
    final_accuracy = correct_count / len(test_data) if test_data else 0
    print(f"✅ {mode_name.upper()} Mode - {subject}: {correct_count}/{len(test_data)} = {final_accuracy:.2%}")
    return results, correct_count

def update_global_summary(output_base, subject, subject_results):
    global_summary_file = f"{output_base}/global_summary.json"
    lock_file = f"{output_base}/.lock"
    for attempt in range(10):
        try:
            if os.path.exists(lock_file):
                time.sleep(1)
                continue
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            if os.path.exists(global_summary_file):
                with open(global_summary_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            else:
                all_results = {
                    "config": {"samples_per_subject": TEST_SAMPLES_PER_SUBJECT,
                               "temperature": TEMPERATURE, "batch_size": BATCH_SIZE},
                    "subjects": {},
                    "summary": {"base": {"total_correct": 0, "total_samples": 0, "accuracy": 0},
                                "roleplay": {"total_correct": 0, "total_samples": 0, "accuracy": 0}}
                }
            all_results["subjects"][subject] = subject_results
            for mode in ["base", "roleplay"]:
                total_correct = sum(subj_data[mode]["correct"] for subj_name, subj_data in all_results["subjects"].items() if mode in subj_data)
                total_samples = sum(subj_data[mode]["total"] for subj_name, subj_data in all_results["subjects"].items() if mode in subj_data)
                all_results["summary"][mode]["total_correct"] = total_correct
                all_results["summary"][mode]["total_samples"] = total_samples
                all_results["summary"][mode]["accuracy"] = total_correct / total_samples if total_samples else 0
            with open(global_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            os.remove(lock_file)
            break
        except Exception as e:
            print(f"⚠️ Error updating global summary: {e}")
            if os.path.exists(lock_file):
                try: os.remove(lock_file)
                except: pass
            time.sleep(2)

def run_model_language(model_path, model_name, lang, suffix, lang_name):
    
    output_base = f"{OUTPUT_BASE}/{model_name}{suffix}"
    print(f"\n{'='*80}")
    print(f"🚀 Starting: Model={model_name}, Language={lang_name}({lang}), Output={output_base}")
    print(f"{'='*80}")
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    
    subjects_per_gpu = [[] for _ in range(len(AVAILABLE_GPUS))]
    for idx, subject in enumerate(TEST_SUBJECTS):
        subjects_per_gpu[idx % len(AVAILABLE_GPUS)].append(subject)
    
    processes = []
    result_queue = Queue()
    
    
    for gpu_idx, gpu_id in enumerate(AVAILABLE_GPUS):
        subjects = subjects_per_gpu[gpu_idx]
        if not subjects:
            continue
        p = Process(target=run_mode_on_gpu, args=(gpu_id, subjects, model_path, lang, output_base, result_queue))
        p.start()
        processes.append(p)
        print(f"✅ Started process for GPU {gpu_id} with {len(subjects)} subjects")
        time.sleep(5)  
    
    for p in processes:
        p.join()
    
    print(f"\n✅ Completed: Model={model_name}, Language={lang_name}({lang})")
    
    
    summary_file = f"{output_base}/global_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            data = json.load(f)
        print(f"  Base accuracy: {data['summary']['base']['accuracy']:.2%}")
        print(f"  Roleplay accuracy: {data['summary']['roleplay']['accuracy']:.2%}")

def run_mode_on_gpu(gpu_id, subjects_list, model_path, lang, output_base, result_queue):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    from vllm import LLM, SamplingParams
    
    print(f"\n🔄 GPU {gpu_id}: Loading model {model_path}...")
    llm = LLM(model=model_path, tensor_parallel_size=1, enforce_eager=True,
              trust_remote_code=True, gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
    tokenizer = llm.get_tokenizer()
    print(f"✓ GPU {gpu_id}: Model loaded")
    
    for subject in subjects_list:
        print(f"\nGPU {gpu_id} - Processing subject: {subject} (lang={lang})")
        test_data = load_mmlu_redux_arrow(subject, TEST_SAMPLES_PER_SUBJECT)
        if not test_data:
            continue
        
        # Base mode
        base_results, base_correct = run_mode_with_resume_batch(
            test_data, subject, "base", process_batch_base,
            llm, tokenizer, lang, output_base, SAVE_INTERVAL, BATCH_SIZE)
        
        # Roleplay mode
        roleplay_results, roleplay_correct = run_mode_with_resume_batch(
            test_data, subject, "roleplay", process_batch_roleplay,
            llm, tokenizer, lang, output_base, SAVE_INTERVAL, BATCH_SIZE)
        
        # Update global summary
        subject_results = {
            "base": {"correct": base_correct, "total": len(test_data), "accuracy": base_correct/len(test_data) if test_data else 0},
            "roleplay": {"correct": roleplay_correct, "total": len(test_data), "accuracy": roleplay_correct/len(test_data) if test_data else 0}
        }
        update_global_summary(output_base, subject, subject_results)
    
    result_queue.put({"gpu_id": gpu_id, "status": "completed"})

# ======================== main ====================
def main():
    print("="*80)
    print("🧪 MMLU-Redux Multi-Language Test (Chinese, Japanese, Malay, German, French, Khmer, Spanish)")
    print("="*80)
    print(f"Total subjects: {len(TEST_SUBJECTS)}")
    print(f"Samples per subject: {TEST_SAMPLES_PER_SUBJECT}")
    print(f"Models to test: {[name for _, name in MODELS]}")
    print(f"Languages: {[lang_name for _, _, lang_name in LANGUAGES]}")
    print(f"GPUs: {AVAILABLE_GPUS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Dataset path: {DATASET_BASE}")
    print(f"Output path: {OUTPUT_BASE}")
    print("="*80)
    
    
    for model_path, model_name in MODELS:
        
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model path not found: {model_path}")
            print(f"   Please set environment variable or check path")
            continue
        
        for lang, suffix, lang_name in LANGUAGES:
            run_model_language(model_path, model_name, lang, suffix, lang_name)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(10)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
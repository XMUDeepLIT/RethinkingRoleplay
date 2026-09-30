#!/usr/bin/env python3
"""
MMLU-Redux Multilingual Prompt Concatenation Test Script 
- Concatenates full prompts from 8 languages directly
- No additional prompt text
- Tests multiple models sequentially
"""

import os
import json
import re
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue
import pyarrow as pa
from vllm import LLM, SamplingParams

# ======================== Suppress NCCL logs ========================
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# ======================== Configuration ========================
MODELS = [
    ("Qwen/Qwen3-14B", "qwen3_14b"),

]

# ======================== GPU Configuration ========================
GPU_PAIRS = [
    [0, 1], [2, 3], [4, 5], [6, 7]
]

# ======================== Global Configuration ========================
DATASET_BASE = "mmlu-redux"
BASE_OUTPUT_DIR = "./mmlu-redux_results_MLCP"
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.3
SAVE_INTERVAL = 10
BATCH_SIZE = 128

SUPPORTED_LANGUAGES = ["en", "zh", "es", "fr", "ja", "de", "km", "ms"]

ALL_SUBJECTS = [
    "electrical_engineering","anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_chemistry", "college_computer_science", "college_mathematics", 
    "college_medicine", "college_physics",  "formal_logic", "global_facts",
     "high_school_geography", "high_school_statistics",
    "high_school_us_history", "human_aging", "logical_fallacies", 
    "econometrics", 
    "machine_learning", "miscellaneous", "philosophy", "professional_accounting",
    "professional_law", "public_relations","high_school_macroeconomics",
    "high_school_chemistry","high_school_mathematics", "high_school_physics",  "virology", "conceptual_physics"
]

def split_subjects_into_groups(subjects, num_groups=4):
    group_size = len(subjects) // num_groups
    remainder = len(subjects) % num_groups
    groups = []
    start = 0
    for i in range(num_groups):
        end = start + group_size + (1 if i < remainder else 0)
        groups.append(subjects[start:end])
        start = end
    return groups

SUBJECTS_GROUPS = split_subjects_into_groups(ALL_SUBJECTS, num_groups=4)
TEST_SAMPLES_PER_SUBJECT = 100

# ======================== Build 8-Language Full Prompts ========================

def build_prompt_multilingual(question, choices, subject, lang):
    subject_display = subject.replace('_', ' ').title()
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    prompts = {
        "en": f"""You are an expert in the field of {subject_display}. Please answer the following question based on your professional knowledge.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

        "zh": f"""你是{subject_display}领域的专家。请基于你的专业知识回答以下问题。

问题：{question}

选项：
{choices_text}

请只输出正确答案的字母（A、B、C或D），不要输出任何其他文本。""",

        "es": f"""Eres un experto en el campo de {subject_display}. Responde la siguiente pregunta basándote en tus conocimientos profesionales.

Pregunta: {question}

Opciones:
{choices_text}

Por favor, genera solo la letra de la respuesta correcta (A, B, C o D). No generes ningún otro texto.""",

        "fr": f"""Vous êtes un expert dans le domaine de {subject_display}. Veuillez répondre à la question suivante en vous basant sur vos connaissances professionnelles.

Question: {question}

Options:
{choices_text}

Veuillez générer uniquement la lettre de la bonne réponse (A, B, C ou D). Ne générez aucun autre texte.""",

        "ja": f"""あなたは{subject_display}分野の専門家です。専門知識に基づいて次の質問に答えてください。

質問：{question}

選択肢：
{choices_text}

正解の文字（A、B、C、D）のみを出力してください。他のテキストは出力しないでください。""",

        "de": f"""Sie sind ein Experte auf dem Gebiet der {subject_display}. Bitte beantworten Sie die folgende Frage basierend auf Ihrem Fachwissen.

Frage: {question}

Optionen:
{choices_text}

Bitte geben Sie nur den Buchstaben der richtigen Antwort aus (A, B, C oder D). Geben Sie keinen anderen Text aus.""",

        "km": f"""អ្នកគឺជាអ្នកជំនាញក្នុងវិស័យ {subject_display}។ សូមឆ្លើយសំណួរខាងក្រោមដោយផ្អែកលើចំណេះដឹងជំនាញរបស់អ្នក។

សំណួរ៖ {question}

ជម្រើស៖
{choices_text}

សូមបញ្ចេញតែអក្សរនៃចម្លើយត្រឹមត្រូវប៉ុណ្ណោះ (A, B, C, ឬ D)។ កុំបញ្ចេញអត្ថបទផ្សេងទៀត។""",

        "ms": f"""Anda adalah pakar dalam bidang {subject_display}. Sila jawab soalan berikut berdasarkan pengetahuan profesional anda.

Soalan: {question}

Pilihan:
{choices_text}

Sila keluarkan hanya huruf jawapan yang betul (A, B, C, atau D). Jangan keluarkan teks lain."""
    }
    
    return prompts.get(lang, prompts["en"])


def build_concat_multilingual_prompt(question, choices, subject):
    lang_labels = {
        "en": "\n" + "="*50 + "\n[ENGLISH VERSION]\n" + "="*50 + "\n",
        "zh": "\n" + "="*50 + "\n[中文版本]\n" + "="*50 + "\n",
        "es": "\n" + "="*50 + "\n[VERSIÓN EN ESPAÑOL]\n" + "="*50 + "\n",
        "fr": "\n" + "="*50 + "\n[VERSION FRANÇAISE]\n" + "="*50 + "\n",
        "ja": "\n" + "="*50 + "\n[日本語バージョン]\n" + "="*50 + "\n",
        "de": "\n" + "="*50 + "\n[DEUTSCHE VERSION]\n" + "="*50 + "\n",
        "km": "\n" + "="*50 + "\n[កំណែភាសាខ្មែរ]\n" + "="*50 + "\n",
        "ms": "\n" + "="*50 + "\n[VERSI BAHASA MELAYU]\n" + "="*50 + "\n"
    }
    
    prompt = ""
    for lang in SUPPORTED_LANGUAGES:
        lang_prompt = build_prompt_multilingual(question, choices, subject, lang)
        prompt += lang_labels[lang] + lang_prompt
    
    return prompt


# ======================== Helper Functions ========================

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
        r'^\s*([A-D])\s*$',
        r'final answer[:\s]*([A-D])',
        r'answer[:\s]*([A-D])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    if len(response.strip()) == 1 and response.strip().upper() in 'ABCD':
        return response.strip().upper()
    
    return None


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


def save_results_incremental(output_file, results, subject, model_name, is_partial=True):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    correct_count = sum(1 for r in results if r.get('final_correct', False))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "subject": subject,
            "prompt_type": "concat_multilingual_full_prompts",
            "languages": SUPPORTED_LANGUAGES,
            "timestamp": datetime.now().isoformat(),
            "is_partial": is_partial,
            "total_samples": len(results),
            "correct_count": correct_count,
            "accuracy": correct_count / len(results) if results else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 Saved {len(results)} results to {output_file}")


def process_concat_batch(items, subject, llm, tokenizer, batch_size, instance_name):
    all_prompts = []
    
    for item in items:
        prompt = build_concat_multilingual_prompt(
            item['question'], item['choices'], subject
        )
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(formatted)
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        max_tokens=4096,
    )
    
    all_responses = []
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_responses.extend([output.outputs[0].text for output in outputs])
        
        progress = min(i + batch_size, len(all_prompts)) / len(all_prompts)
        print(f"  {instance_name} - Progress: {progress:.1%}", end='\r')
    
    print()
    
    results = []
    for item, response in zip(items, all_responses):
        predicted = extract_answer(response)
        is_correct = (predicted == item['answer_letter']) if predicted else False
        
        result_item = {
            "idx": item['idx'],
            "question": item['question'],
            "choices": item['choices'],
            "ground_truth": item['answer_letter'],
            "final_answer": predicted,
            "final_correct": is_correct,
            "model_response": response[:500],
        }
        results.append(result_item)
    
    return results


def run_subject(subject, llm, tokenizer, output_dir, model_name, instance_name):
    print(f"\n{'#'*60}")
    print(f"{instance_name} - Processing subject: {subject}")
    print(f"{'#'*60}")
    
    test_data = load_mmlu_redux_arrow(subject, TEST_SAMPLES_PER_SUBJECT)
    if not test_data or len(test_data) == 0:
        print(f"{instance_name} - ❌ Failed to load data for {subject}, skipping...")
        return None, None
    
    result_file = f"{output_dir}/{subject}_concat_multilingual.json"
    existing_results, completed_indices = load_existing_results(result_file)
    
    results = existing_results.copy()
    pending_items = []
    
    for item in test_data:
        if item['idx'] not in completed_indices:
            pending_items.append(item)
    
    if len(pending_items) == 0:
        print(f"  {instance_name} - ✅ All {len(test_data)} samples already completed")
        correct_count = sum(1 for r in results if r.get('final_correct', False))
        return results, correct_count
    
    print(f"  {instance_name} - 📍 Resuming: {len(pending_items)}/{len(test_data)} samples pending")
    
    correct_count = sum(1 for r in results if r.get('final_correct', False))
    total_processed = len(results)
    
    for batch_start in range(0, len(pending_items), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(pending_items))
        batch_items = pending_items[batch_start:batch_end]
        
        try:
            batch_results = process_concat_batch(
                batch_items, subject, llm, tokenizer, BATCH_SIZE, instance_name
            )
            
            for result in batch_results:
                if result['final_correct']:
                    correct_count += 1
                results.append(result)
                total_processed += 1
            
            current_acc = correct_count / total_processed if total_processed > 0 else 0
            print(f"  {instance_name} - Processed {total_processed}/{len(test_data)} samples, accuracy: {current_acc:.2%}")
            
            if total_processed % SAVE_INTERVAL == 0 or total_processed == len(test_data):
                save_results_incremental(result_file, results, subject, model_name,
                                        is_partial=(total_processed < len(test_data)))
                
        except Exception as e:
            print(f"  {instance_name} - ❌ Error processing batch: {e}")
            for item in batch_items:
                error_item = {
                    "idx": item['idx'],
                    "error": str(e),
                    "final_correct": False,
                    "question": item['question'],
                    "ground_truth": item['answer_letter'],
                }
                results.append(error_item)
                total_processed += 1
    
    save_results_incremental(result_file, results, subject, model_name, is_partial=False)
    
    final_accuracy = correct_count / len(test_data) if test_data else 0
    print(f"\n✅ {instance_name} - {subject}: {correct_count}/{len(test_data)} = {final_accuracy:.2%}")
    
    return results, correct_count


def run_instance(gpu_ids, subjects, output_dir, model_path, model_name, instance_id, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    
    tensor_parallel_size = len(gpu_ids)
    instance_name = f"Proc{instance_id}"
    
    print("\n" + "="*80)
    print(f"🚀 Starting {instance_name}")
    print(f"   Model: {model_name}")
    print(f"   Model Path: {model_path}")
    print(f"   GPUs: {gpu_ids}")
    print(f"   Tensor Parallel Size: {tensor_parallel_size}")
    print(f"   Subjects: {len(subjects)}")
    print(f"   Output: {output_dir}")
    print("="*80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 {instance_name}: Loading model on GPUs {gpu_ids}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    tokenizer = llm.get_tokenizer()
    print(f"✅ {instance_name}: Model loaded successfully!")
    
    subject_results = {}
    for subject in subjects:
        results, correct_count = run_subject(subject, llm, tokenizer, output_dir, model_name, instance_name)
        if results:
            subject_results[subject] = {
                "total": len(results),
                "correct": correct_count,
                "accuracy": correct_count / len(results) if results else 0
            }
        time.sleep(2)
    
    result_queue.put({
        "instance_id": instance_id,
        "gpu_ids": gpu_ids,
        "model_name": model_name,
        "model_path": model_path,
        "subjects": subject_results,
        "subjects_list": subjects
    })
    
    print(f"\n✅ {instance_name} completed!")
    
    del llm
    import torch
    torch.cuda.empty_cache()


def save_global_summary(all_results, output_dir):
    model_results = {}
    
    for result in all_results:
        model_name = result["model_name"]
        if model_name not in model_results:
            model_results[model_name] = {
                "model_path": result["model_path"],
                "subjects": {},
                "total_correct": 0,
                "total_samples": 0
            }
        
        for subject, data in result["subjects"].items():
            model_results[model_name]["subjects"][subject] = data
            model_results[model_name]["total_correct"] += data["correct"]
            model_results[model_name]["total_samples"] += data["total"]
    
    for model_name in model_results:
        total_correct = model_results[model_name]["total_correct"]
        total_samples = model_results[model_name]["total_samples"]
        model_results[model_name]["accuracy"] = total_correct / total_samples if total_samples > 0 else 0
    
    sorted_models = sorted(
        model_results.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )
    
    model_comparison = []
    for model_name, data in sorted_models:
        model_comparison.append({
            "model_name": model_name,
            "model_path": data["model_path"],
            "total_correct": data["total_correct"],
            "total_samples": data["total_samples"],
            "accuracy": data["accuracy"]
        })
    
    global_summary = {
        "timestamp": datetime.now().isoformat(),
        "prompt_type": "concat_multilingual_full_prompts",
        "languages": SUPPORTED_LANGUAGES,
        "samples_per_subject": TEST_SAMPLES_PER_SUBJECT,
        "models_tested": [m[1] for m in MODELS],
        "model_results": model_results,
        "model_comparison": model_comparison,
        "best_model": model_comparison[0] if model_comparison else None
    }
    
    output_file = f"{output_dir}/global_summary.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("📊 GLOBAL SUMMARY SAVED")
    print("="*80)
    print(f"File: {output_file}")
    print(f"Models tested: {len(model_results)}")
    print("\n📈 Model Performance Comparison:")
    for i, comp in enumerate(model_comparison, 1):
        print(f"  {i}. {comp['model_name']}: {comp['accuracy']:.2%} ({comp['total_correct']}/{comp['total_samples']})")
    print("="*80)
    
    return global_summary


def test_model(model_path, model_name, model_index):
    print("\n" + "="*80)
    print(f"🔬 Testing Model {model_index+1}/{len(MODELS)}: {model_name}")
    print(f"   Path: {model_path}")
    print("="*80)
    
    output_dir = f"{BASE_OUTPUT_DIR}/{model_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    gpu_pairs = GPU_PAIRS.copy()
    
    result_queue = Queue()
    processes = []
    
    for i, gpu_ids in enumerate(gpu_pairs, 1):
        subjects = SUBJECTS_GROUPS[i-1] if i-1 < len(SUBJECTS_GROUPS) else []
        if not subjects:
            continue
            
        p = Process(target=run_instance, args=(
            gpu_ids, subjects, output_dir, model_path, model_name, i, result_queue
        ))
        p.start()
        processes.append(p)
        print(f"✅ Started Process {i} on GPUs {gpu_ids} (PID: {p.pid})")
        time.sleep(3)
    
    print(f"\n⏳ Waiting for all processes to complete for {model_name}...")
    for p in processes:
        p.join()
        print(f"   Process {p.pid} finished")
    
    all_results = []
    while not result_queue.empty():
        all_results.append(result_queue.get())
    
    print(f"\n✅ All processes completed for {model_name}")
    return all_results, output_dir


def main():
    print(f"Total models to test: {len(MODELS)}")
    for i, (path, name) in enumerate(MODELS, 1):
        print(f"  {i}. {name}: {path}")
    print(f"Prompt Type: Concatenated 8-language full prompts")
    print(f"Total subjects: {len(ALL_SUBJECTS)}")
    print(f"Samples per subject: {TEST_SAMPLES_PER_SUBJECT}")
    print(f"Languages: {SUPPORTED_LANGUAGES}")
    print("="*80)
    
    all_model_results = []
    
    for model_idx, (model_path, model_name) in enumerate(MODELS):
        if not os.path.exists(model_path):
            print(f"\n⚠️ Model path not found: {model_path}")
            print(f"   Skipping {model_name}...")
            continue
        
        try:
            results, output_dir = test_model(model_path, model_name, model_idx)
            all_model_results.extend(results)
            
            if results:
                save_global_summary(all_model_results, BASE_OUTPUT_DIR)
            
            print(f"\n✅ Model {model_name} testing completed!")
            
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            
            time.sleep(5)
            
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("🎉 All models testing completed!")
    print(f"Results saved to: {BASE_OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
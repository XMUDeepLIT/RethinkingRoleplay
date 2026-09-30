#!/usr/bin/env python3
"""
MMLU-Redux Multi-Model Batch Test Script (Auto Mode)
- Automatically tests multiple models with multilingual role-playing
- Each model's results saved in separate directory
- Generates comparison report for all models
"""

import os
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue
import pyarrow as pa
from vllm import LLM, SamplingParams

# ======================== Model Configuration ========================
MODELS_CONFIG = [
    {
        "name": "Qwen3-14B",
        "path": "Qwen/Qwen3-14B",
        "completed": True,
    },

]

# ======================== Global Configuration ========================
DATASET_BASE = "mmlu-redux"
BASE_OUTPUT_DIR = "./mmlu-redux_results_MLVP"
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.3
SAVE_INTERVAL = 10
BATCH_SIZE = 128

SUPPORTED_LANGUAGES = ["en", "zh", "es", "fr", "ja", "de", "km", "ms"]
LANG_NAMES = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "de": "German", "km": "Khmer", "ms": "Malay"
}

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
TEST_SAMPLES_PER_SUBJECT = 100

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

# ======================== Multilingual Prompts ========================

def build_prompt_roleplay_multilingual(question, choices, subject, lang):
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

# ======================== Helper Functions ========================

def vote_answers(predictions):
    if isinstance(predictions, dict):
        answers = [ans for ans in predictions.values() if ans is not None]
    else:
        answers = [ans for ans in predictions if ans is not None]
    
    if not answers:
        return None, {}
    
    vote_counter = Counter(answers)
    most_common = vote_counter.most_common(1)[0]
    return most_common[0], dict(vote_counter)

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

def save_results_incremental(output_file, results, subject, is_partial=True):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    correct_count = sum(1 for r in results if r.get('final_correct', False))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "subject": subject,
            "languages": SUPPORTED_LANGUAGES,
            "timestamp": datetime.now().isoformat(),
            "is_partial": is_partial,
            "total_samples": len(results),
            "correct_count": correct_count,
            "accuracy": correct_count / len(results) if results else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 Saved {len(results)} results to {output_file}")

# ======================== Batch Processing ========================

def process_multilingual_batch(items, subject, llm, tokenizer, gpu_id, batch_size):
    all_prompts = []
    prompt_mapping = []
    
    for item in items:
        for lang in SUPPORTED_LANGUAGES:
            prompt = build_prompt_roleplay_multilingual(
                item['question'], item['choices'], subject, lang
            )
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(formatted)
            prompt_mapping.append({
                "item_idx": item['idx'],
                "lang": lang,
                "item": item
            })
    
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=2048)
    
    all_responses = []
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_responses.extend([output.outputs[0].text for output in outputs])
        
        progress = min(i + batch_size, len(all_prompts)) / len(all_prompts)
        print(f"  GPU {gpu_id} - Language inference progress: {progress:.1%}", end='\r')
    
    print()
    
    sample_predictions = {}
    for mapping, response in zip(prompt_mapping, all_responses):
        item_idx = mapping["item_idx"]
        lang = mapping["lang"]
        predicted = extract_answer(response)
        
        if item_idx not in sample_predictions:
            sample_predictions[item_idx] = {}
        
        sample_predictions[item_idx][lang] = {
            "predicted": predicted,
            "response": response[:300]
        }
    
    results = []
    for item in items:
        idx = item['idx']
        lang_predictions = sample_predictions.get(idx, {})
        
        predictions_dict = {}
        for lang in SUPPORTED_LANGUAGES:
            if lang in lang_predictions:
                predictions_dict[lang] = lang_predictions[lang]['predicted']
            else:
                predictions_dict[lang] = None
        
        final_answer, vote_details = vote_answers(predictions_dict)
        is_correct = (final_answer == item['answer_letter']) if final_answer else False
        
        result_item = {
            "idx": item['idx'],
            "question": item['question'],
            "choices": item['choices'],
            "ground_truth": item['answer_letter'],
            "final_answer": final_answer,
            "final_correct": is_correct,
            "vote_details": vote_details,
            "per_language": {
                lang: {
                    "predicted": predictions_dict.get(lang),
                    "response": lang_predictions.get(lang, {}).get('response', '') if lang in lang_predictions else ''
                }
                for lang in SUPPORTED_LANGUAGES
            }
        }
        results.append(result_item)
    
    return results

def run_subject_on_gpu(subject, gpu_id, llm, tokenizer, output_base):
    print(f"\n{'#'*60}")
    print(f"GPU {gpu_id} - Processing subject: {subject}")
    print(f"{'#'*60}")
    
    test_data = load_mmlu_redux_arrow(subject, TEST_SAMPLES_PER_SUBJECT)
    if not test_data or len(test_data) == 0:
        print(f"GPU {gpu_id} - ❌ Failed to load data for {subject}, skipping...")
        return None, None
    
    result_file = f"{output_base}/{subject}_multilingual_roleplay.json"
    existing_results, completed_indices = load_existing_results(result_file)
    
    results = existing_results.copy()
    pending_items = []
    
    for item in test_data:
        if item['idx'] not in completed_indices:
            pending_items.append(item)
    
    if len(pending_items) == 0:
        print(f"  GPU {gpu_id} - ✅ All {len(test_data)} samples already completed")
        correct_count = sum(1 for r in results if r.get('final_correct', False))
        return results, correct_count
    
    print(f"  GPU {gpu_id} - 📍 Resuming: {len(pending_items)}/{len(test_data)} samples pending")
    
    correct_count = sum(1 for r in results if r.get('final_correct', False))
    total_processed = len(results)
    
    for batch_start in range(0, len(pending_items), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(pending_items))
        batch_items = pending_items[batch_start:batch_end]
        
        try:
            batch_results = process_multilingual_batch(
                batch_items, subject, llm, tokenizer, gpu_id, BATCH_SIZE
            )
            
            for result in batch_results:
                if result['final_correct']:
                    correct_count += 1
                results.append(result)
                total_processed += 1
            
            current_acc = correct_count / total_processed if total_processed > 0 else 0
            print(f"  GPU {gpu_id} - Processed {total_processed}/{len(test_data)} samples, accuracy: {current_acc:.2%}")
            
            if total_processed % SAVE_INTERVAL == 0 or total_processed == len(test_data):
                save_results_incremental(result_file, results, subject, 
                                        is_partial=(total_processed < len(test_data)))
                
        except Exception as e:
            print(f"  GPU {gpu_id} - ❌ Error processing batch: {e}")
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
    
    save_results_incremental(result_file, results, subject, is_partial=False)
    
    final_accuracy = correct_count / len(test_data) if test_data else 0
    print(f"\n✅ GPU {gpu_id} - {subject}: {correct_count}/{len(test_data)} = {final_accuracy:.2%}")
    
    return results, correct_count

def run_gpu_worker(gpu_id, subjects_list, result_queue, model_path, output_base, model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    
    print(f"\n{'='*60}")
    print(f"🚀 GPU {gpu_id} - Model: {model_name}")
    print(f"   Subjects: {len(subjects_list)}")
    print(f"{'='*60}")
    
    print(f"\n🔄 GPU {gpu_id}: Loading model...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    tokenizer = llm.get_tokenizer()
    print(f"✓ GPU {gpu_id}: Model loaded successfully!")
    
    for subject in subjects_list:
        results, correct_count = run_subject_on_gpu(subject, gpu_id, llm, tokenizer, output_base)
        if results:
            update_global_summary(subject, {
                "total": len(results),
                "correct": correct_count,
                "accuracy": correct_count / len(results) if results else 0
            }, gpu_id, output_base, model_name, model_path)
    
    print(f"\n✅ GPU {gpu_id} completed!")
    result_queue.put({"gpu_id": gpu_id, "status": "completed"})

def update_global_summary(subject, subject_result, gpu_id, output_base, model_name, model_path):
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
                    "config": {
                        "model_name": model_name,
                        "model_path": model_path,
                        "languages": SUPPORTED_LANGUAGES,
                        "samples_per_subject": TEST_SAMPLES_PER_SUBJECT,
                        "voting_method": "majority_vote",
                        "gpus": AVAILABLE_GPUS,
                        "total_subjects": len(TEST_SUBJECTS),
                        "batch_size": BATCH_SIZE
                    },
                    "subjects": {},
                    "summary": {"total_correct": 0, "total_samples": 0, "accuracy": 0},
                    "gpu_progress": {}
                }
            
            if subject_result:
                all_results["subjects"][subject] = subject_result
            
            if f"gpu_{gpu_id}" not in all_results["gpu_progress"]:
                all_results["gpu_progress"][f"gpu_{gpu_id}"] = {"subjects_completed": []}
            if subject not in all_results["gpu_progress"][f"gpu_{gpu_id}"]["subjects_completed"]:
                all_results["gpu_progress"][f"gpu_{gpu_id}"]["subjects_completed"].append(subject)
            
            total_correct = 0
            total_samples = 0
            for subj_data in all_results["subjects"].values():
                if subj_data:
                    total_correct += subj_data.get("correct", 0)
                    total_samples += subj_data.get("total", 0)
            
            all_results["summary"]["total_correct"] = total_correct
            all_results["summary"]["total_samples"] = total_samples
            all_results["summary"]["accuracy"] = total_correct / total_samples if total_samples > 0 else 0
            
            with open(global_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            os.remove(lock_file)
            break
            
        except Exception as e:
            print(f"⚠️ Error updating global summary: {e}")
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except:
                    pass
            time.sleep(2)

# ======================== Main Test Function ========================

def test_single_model(model_config):
    model_name = model_config["name"]
    model_path = model_config["path"]
    output_base = f"{BASE_OUTPUT_DIR}/{model_name}"
    
    print("\n" + "="*80)
    print(f"🧪 Testing model: {model_name}")
    print(f"   Path: {model_path}")
    print(f"   Output: {output_base}")
    print("="*80)
    
    summary_file = f"{output_base}/global_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            existing = json.load(f)
            summary = existing.get("summary", {})
            if summary.get("total_samples", 0) >= len(TEST_SUBJECTS) * TEST_SAMPLES_PER_SUBJECT:
                print(f"✅ Model {model_name} already has complete results!")
                print(f"   Accuracy: {summary.get('accuracy', 0):.2%}")
                print(f"   Total: {summary.get('total_correct', 0)}/{summary.get('total_samples', 0)}")
                return True
    
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    subjects_per_gpu = [[] for _ in range(len(AVAILABLE_GPUS))]
    for idx, subject in enumerate(TEST_SUBJECTS):
        gpu_idx = idx % len(AVAILABLE_GPUS)
        subjects_per_gpu[gpu_idx].append(subject)
    
    processes = []
    result_queue = Queue()
    
    for gpu, subjects in zip(AVAILABLE_GPUS, subjects_per_gpu):
        if subjects:
            p = Process(target=run_gpu_worker, 
                       args=(gpu, subjects, result_queue, model_path, output_base, model_name))
            p.start()
            processes.append(p)
            print(f"✅ Started process for GPU {gpu}")
            time.sleep(5)
    
    print("\n⏳ Waiting for all GPU processes to complete...")
    for p in processes:
        p.join()
    
    print(f"\n✅ Model {model_name} test completed!")
    return True

# ======================== Generate Comparison Report ========================

def generate_comparison_report():
    print("\n" + "="*80)
    print("📊 Generating comparison report...")
    print("="*80)
    
    report_dir = f"{BASE_OUTPUT_DIR}/comparison"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_config in MODELS_CONFIG:
        model_name = model_config["name"]
        result_file = f"{BASE_OUTPUT_DIR}/{model_name}/global_summary.json"
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                summary = data.get("summary", {})
                all_results[model_name] = {
                    "accuracy": summary.get("accuracy", 0),
                    "total_correct": summary.get("total_correct", 0),
                    "total_samples": summary.get("total_samples", 0),
                    "config": data.get("config", {}),
                    "completed": model_config.get("completed", False)
                }
        else:
            all_results[model_name] = {
                "error": "No results found",
                "completed": model_config.get("completed", False)
            }
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("📊 MULTI-MODEL COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"{'Model':<30} {'Accuracy':<12} {'Correct':<12} {'Total':<12} {'Status'}")
    report_lines.append("-"*80)
    
    sorted_models = sorted(all_results.items(), 
                          key=lambda x: x[1].get("accuracy", 0) if "accuracy" in x[1] else 0, 
                          reverse=True)
    
    for model_name, data in sorted_models:
        if "accuracy" in data:
            acc = data["accuracy"]
            correct = data.get("total_correct", 0)
            total = data.get("total_samples", 0)
            status = "✅" if data.get("completed") else "🔄"
            report_lines.append(f"{model_name:<30} {acc:<12.2%} {correct:<12} {total:<12} {status}")
        else:
            status = "✅" if data.get("completed") else "🔄"
            report_lines.append(f"{model_name:<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {status} ({data.get('error', 'No data')})")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("📈 DETAILED PER-MODEL PERFORMANCE")
    report_lines.append("="*80)
    
    for model_name, data in sorted_models:
        if "accuracy" in data:
            report_lines.append(f"\n🔹 {model_name}")
            report_lines.append(f"   Overall Accuracy: {data['accuracy']:.2%}")
            report_lines.append(f"   Total Correct: {data['total_correct']}/{data['total_samples']}")
            
            result_file = f"{BASE_OUTPUT_DIR}/{model_name}/global_summary.json"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    full_results = json.load(f)
                
                subjects = full_results.get("subjects", {})
                if subjects:
                    subject_accs = [(name, d.get("accuracy", 0)) 
                                   for name, d in subjects.items() if d]
                    subject_accs.sort(key=lambda x: x[1], reverse=True)
                    
                    report_lines.append(f"\n   Top 5 Subjects:")
                    for i, (subj, acc) in enumerate(subject_accs[:5], 1):
                        report_lines.append(f"      {i}. {subj:<35} {acc:.2%}")
                    
                    report_lines.append(f"\n   Bottom 5 Subjects:")
                    for i, (subj, acc) in enumerate(subject_accs[-5:], 1):
                        report_lines.append(f"      {i}. {subj:<35} {acc:.2%}")
    
    report_text = "\n".join(report_lines)
    report_file = f"{report_dir}/model_comparison.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n📄 Text report saved to: {report_file}")
    
    comparison_json = {
        "generated_at": datetime.now().isoformat(),
        "models": all_results
    }
    json_file = f"{report_dir}/model_comparison.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_json, f, indent=2)
    print(f"📄 JSON report saved to: {json_file}")
    
    print("\n" + report_text)
    
    return all_results

# ======================== Main Function (Auto Mode) ========================

def main():
    print("="*80)
    print("🤖 MMLU-Redux Multi-Model Batch Test (Auto Mode)")
    print("="*80)
    print(f"Total models: {len(MODELS_CONFIG)}")
    print(f"Languages: {SUPPORTED_LANGUAGES}")
    print(f"Subjects: {len(TEST_SUBJECTS)}")
    print(f"Samples per subject: {TEST_SAMPLES_PER_SUBJECT}")
    print(f"GPUs: {AVAILABLE_GPUS}")
    print("="*80)
    
    print("\n📋 Models:")
    for i, model in enumerate(MODELS_CONFIG, 1):
        status = "✅ Completed" if model["completed"] else "⏳ Pending"
        print(f"   {i}. {model['name']:<30} {status}")
    
    print("\n" + "-"*80)
    print("🚀 Starting automatic testing for pending models...")
    print("-"*80)
    
    for model_config in MODELS_CONFIG:
        if not model_config["completed"]:
            print(f"\n{'='*80}")
            print(f"Testing: {model_config['name']}")
            print(f"{'='*80}")
            
            success = test_single_model(model_config)
            if success:
                model_config["completed"] = True
                print(f"\n✅ {model_config['name']} completed successfully!")
            else:
                print(f"\n❌ {model_config['name']} test failed!")
    
    print("\n" + "="*80)
    print("Generating final comparison report...")
    print("="*80)
    generate_comparison_report()
    
    print("\n" + "="*80)
    print("🎉 All tasks completed!")
    print(f"Results saved to: {BASE_OUTPUT_DIR}/")
    print(f"Comparison report: {BASE_OUTPUT_DIR}/comparison/")
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
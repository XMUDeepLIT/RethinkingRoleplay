#!/usr/bin/env python3


import os
import json
import re
import time
import sys
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue
import pyarrow as pa
from vllm import LLM, SamplingParams


MODEL_PATH = "./models/Qwen/Qwen3-4B"
OUTPUT_BASE = "./mmlu_redux_results_qwen3_4b_information_richness_analysis"

DATASET_BASE = "./datasets/mmlu-redux"

TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.5
SAVE_INTERVAL = 10
BATCH_SIZE = 128


TEST_SUBJECTS = [
    "anatomy",
    "college_physics",
    "high_school_geography",
    "business_ethics",
    "human_aging",
    "philosophy",
    "professional_accounting"
]
TEST_SAMPLES_PER_SUBJECT = 100


AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6,7]

# ======================== moderate version  ========================
ROLEPLAY_PROMPTS = {
    "anatomy": """You are a professional expert in human anatomy. You have solid knowledge of major organ systems, including skeleton, muscles, nerves, circulation, respiration, digestion, and reproduction. You are familiar with standard anatomical terms and the structural relationships of tissues and organs. Please select the correct answer based on authoritative anatomical knowledge.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

    "college_physics": """You are a professional physics expert in college-level physics. Your expertise includes classical mechanics, thermodynamics, electromagnetism, waves, optics, and modern physics. You understand fundamental physical laws and core definitions. Please choose the answer based on rigorous physical principles.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

    "high_school_geography": """You are a professional geography teacher. You are proficient in physical geography (terrain, climate, hydrology) and human geography (population, urban development, industry, agriculture). You understand how natural environments and human activities interact. Please select the answer according to geographical principles.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

    "business_ethics": """You are an expert in business ethics and corporate responsibility. You are familiar with ethical theories, compliance requirements, fair competition, labor rights, and corporate governance. You evaluate business practices from moral and professional standards. Please choose the answer that aligns with mainstream business ethics.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",


    "human_aging": """You are an expert in gerontology and human aging. You have solid knowledge of biological aging processes, psychological changes in later life, age-related diseases, and social aspects of aging. You understand theories of aging, longevity factors, and interventions for healthy aging. Please select the answer based on scientific evidence in gerontology.

    Question: {question}

    Options:
    {choices_text}

    Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

    "philosophy": """You are a professional philosopher with knowledge across major philosophical traditions. Your expertise includes metaphysics, epistemology, ethics, logic, and political philosophy. You are familiar with key thinkers and philosophical arguments. Please reason through the question using rigorous philosophical analysis.

    Question: {question}

    Options:
    {choices_text}

    Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

    "professional_accounting": """You are a professional accountant with practical experience. You have knowledge of financial accounting standards, managerial accounting, auditing, tax regulations, and financial reporting. Your expertise covers GAAP, IFRS, cost accounting, and professional ethics. Please select the answer that follows professional accounting standards.

    Question: {question}

    Options:
    {choices_text}

    Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text."""
}
# ======================== rich version ========================
# ROLEPLAY_PROMPTS = {
#     "anatomy": """You are a professional expert in human anatomy. You have comprehensive knowledge of human body structures, covering major organ systems such as skeleton, muscles, nerves, blood circulation, respiration, digestion, urinary and reproductive systems. You are familiar with standard anatomical terms, the positional relationship, structural features and physiological functions of various tissues and organs. You can tell apart common structural features and typical anatomical knowledge points. Please analyze the question carefully and make your choice strictly based on authoritative anatomical knowledge.

# Question: {question}

# Options:
# {choices_text}

# Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

#     "college_physics": """You are a professional physics expert who masters the full range of college physics knowledge. Your expertise includes classical mechanics, fluid mechanics, thermodynamics, electromagnetism, mechanical waves, optics and basic modern physics. You have a solid understanding of fundamental physical laws, core definitions, physical models and calculation principles. You are able to identify confusing concepts and common mistakes in physics learning, and judge answers according to rigorous physical theories and objective rules.

# Question: {question}

# Options:
# {choices_text}

# Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

#     "high_school_geography": """You are a professional geography teacher with solid theoretical knowledge. You are proficient in both physical geography and human geography. Your knowledge includes global terrain features, climate distribution, hydrological circulation, soil and vegetation, as well as population distribution, urban development, industry, agriculture, transportation and regional differences. You understand the mutual influence between human society and natural environment, and analyze problems from the perspective of geographical laws and regional characteristics.

# Question: {question}

# Options:
# {choices_text}

# Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

#     "business_ethics": """You are an expert focusing on business ethics and corporate social responsibility. You are familiar with basic ethical theories, commercial behavior norms and industry compliance requirements. Your work covers conflicts of interest, fair competition, labor rights, advertising credibility, environmental protection, data ethics and corporate governance. You can evaluate business behaviors from moral, legal and social perspectives, and select answers that conform to mainstream business ethics and professional standards.

# Question: {question}

# Options:
# {choices_text}

# Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

#     "human_aging": """You are a professional expert in gerontology and human aging. You have comprehensive knowledge of the biological, psychological, and social aspects of aging. Your expertise covers cellular senescence, age-related physiological changes, cognitive decline, neurodegenerative diseases, and the impact of lifestyle factors on longevity. You understand theories of aging, demographic trends in population aging, and interventions for healthy aging. Please analyze the question carefully based on scientific research and clinical evidence in gerontology.

# Question: {question}

# Options:
# {choices_text}

# Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

#     "philosophy": """You are a professional philosopher with broad knowledge across major philosophical traditions. Your expertise includes metaphysics, epistemology, ethics, logic, philosophy of mind, philosophy of science, and political philosophy. You are familiar with key thinkers from ancient Greek philosophy to contemporary analytic and continental traditions. You understand philosophical argumentation, conceptual analysis, and can distinguish between different philosophical positions and their implications. Please reason through the question using rigorous philosophical analysis.

# Question: {question}

# Options:
# {choices_text}

# Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text.""",

#     "professional_accounting": """You are a professional certified accountant with extensive practical experience. You have comprehensive knowledge of financial accounting standards, managerial accounting, auditing procedures, tax regulations, and financial reporting requirements. Your expertise covers GAAP, IFRS, cost accounting, budgeting, internal controls, and professional ethics in accounting. You understand how to apply accounting principles to real business scenarios and can identify correct treatment of various financial transactions. Please select the answer that follows professional accounting standards.

# Question: {question}

# Options:
# {choices_text}

# Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text."""
# }

# simple version
GENERIC_ROLEPLAY_TEMPLATE = """You are an expert in the field of {subject_display}. Please answer the following question based on your professional knowledge.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text."""

def build_prompt_base(question, choices):

    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    return f"""Answer the following multiple-choice question. Choose the correct option.

Question: {question}

Options:
{choices_text}

Please output only the letter of the correct answer (A, B, C, or D). Do not output any other text."""

def build_prompt_roleplay(question, choices, subject):
    
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    
    if subject in ROLEPLAY_PROMPTS:
        return ROLEPLAY_PROMPTS[subject].format(
            question=question,
            choices_text=choices_text,
            subject=subject
        )
    else:
        
        subject_display = subject.replace('_', ' ').title()
        return GENERIC_ROLEPLAY_TEMPLATE.format(
            subject_display=subject_display,
            question=question,
            choices_text=choices_text
        )


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


def extract_answer_base(response):
    
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


def process_batch_base(items, subject, llm, tokenizer, gpu_id):
    
    prompts = []
    for item in items:
        prompt = build_prompt_base(item['question'], item['choices'])
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
    
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=2048)
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for item, output in zip(items, outputs):
        response = output.outputs[0].text
        predicted = extract_answer_base(response)
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

def process_batch_roleplay(items, subject, llm, tokenizer, gpu_id):
    
    prompts = []
    for item in items:
        prompt = build_prompt_roleplay(item['question'], item['choices'], subject)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
    
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=2048)
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for item, output in zip(items, outputs):
        response = output.outputs[0].text
        predicted = extract_answer_base(response)
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


def run_mode_with_resume_batch(test_data, subject, mode_name, batch_process_func, llm, tokenizer, gpu_id, save_interval, batch_size):

    result_file = f"{OUTPUT_BASE}/{subject}_{mode_name}.json"
    
    existing_results, completed_indices = load_existing_results(result_file)
    
    results = existing_results.copy()
    pending_items = []
    
    for item in test_data:
        if item['idx'] not in completed_indices:
            pending_items.append(item)
    
    if len(pending_items) == 0:
        print(f"  GPU {gpu_id} - ✅ All {len(test_data)} samples already completed for {mode_name}")
        correct_count = sum(1 for r in results if r.get('correct', False))
        return results, correct_count
    
    print(f"  GPU {gpu_id} - 📍 Resuming {mode_name}: {len(pending_items)}/{len(test_data)} samples pending")
    print(f"  GPU {gpu_id} - Using batch size: {batch_size}")
    
    correct_count = sum(1 for r in results if r.get('correct', False))
    total_processed = len(results)
    
    for batch_start in range(0, len(pending_items), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_items))
        batch_items = pending_items[batch_start:batch_end]
        
        try:
            batch_results = batch_process_func(batch_items, subject, llm, tokenizer, gpu_id)
            
            for result in batch_results:
                if result['correct']:
                    correct_count += 1
                results.append(result)
                total_processed += 1
            
            print(f"  GPU {gpu_id} - Processed batch {batch_start//batch_size + 1}/{(len(pending_items)-1)//batch_size + 1}, "
                  f"total: {total_processed}/{len(test_data)} samples, accuracy: {correct_count/total_processed:.2%}")
            
            if total_processed % save_interval == 0 or total_processed == len(test_data):
                save_results_incremental(result_file, results, mode_name, subject, 
                                        is_partial=(total_processed < len(test_data)))
                
        except Exception as e:
            print(f"  GPU {gpu_id} - ❌ Error processing batch starting at {batch_start}: {e}")
            for item in batch_items:
                error_item = {
                    "idx": item['idx'],
                    "error": str(e),
                    "correct": False,
                    "question": item['question'],
                    "ground_truth": item['answer_letter'],
                }
                results.append(error_item)
                total_processed += 1
    
    save_results_incremental(result_file, results, mode_name, subject, is_partial=False)
    
    final_accuracy = correct_count / len(test_data) if test_data else 0
    print(f"\n✅ GPU {gpu_id} - {mode_name.upper()} Mode - {subject}: {correct_count}/{len(test_data)} = {final_accuracy:.2%}")
    
    return results, correct_count


def update_global_summary(subject, subject_results, gpu_id):
    
    global_summary_file = f"{OUTPUT_BASE}/global_summary.json"
    lock_file = f"{OUTPUT_BASE}/.lock"
    
    max_attempts = 10
    for attempt in range(max_attempts):
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
                        "model": MODEL_PATH,
                        "samples_per_subject": TEST_SAMPLES_PER_SUBJECT,
                        "gpus": AVAILABLE_GPUS,
                        "temperature": TEMPERATURE,
                        "total_subjects": len(TEST_SUBJECTS),
                        "save_interval": SAVE_INTERVAL,
                        "batch_size": BATCH_SIZE
                    },
                    "subjects": {},
                    "summary": {
                        "base": {"total_correct": 0, "total_samples": 0, "accuracy": 0},
                        "roleplay": {"total_correct": 0, "total_samples": 0, "accuracy": 0}
                    },
                    "gpu_progress": {}
                }
            
            all_results["subjects"][subject] = subject_results
            
            if f"gpu_{gpu_id}" not in all_results["gpu_progress"]:
                all_results["gpu_progress"][f"gpu_{gpu_id}"] = {"subjects_completed": []}
            if subject not in all_results["gpu_progress"][f"gpu_{gpu_id}"]["subjects_completed"]:
                all_results["gpu_progress"][f"gpu_{gpu_id}"]["subjects_completed"].append(subject)
            
            for mode in ["base", "roleplay"]:
                total_correct = 0
                total_samples = 0
                for subj_name, subj_data in all_results["subjects"].items():
                    if mode in subj_data:
                        total_correct += subj_data[mode]["correct"]
                        total_samples += subj_data[mode]["total"]
                all_results["summary"][mode]["total_correct"] = total_correct
                all_results["summary"][mode]["total_samples"] = total_samples
                all_results["summary"][mode]["accuracy"] = total_correct / total_samples if total_samples > 0 else 0
            
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


def run_mode_on_gpu(gpu_id, subjects_list, result_queue):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    
    print(f"\n{'='*60}")
    print(f"🚀 GPU {gpu_id} started with {len(subjects_list)} subjects: {subjects_list}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"{'='*60}")
    
    from vllm import LLM, SamplingParams
    
    print(f"\n🔄 GPU {gpu_id}: Loading model...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    tokenizer = llm.get_tokenizer()
    print(f"✓ GPU {gpu_id}: Model loaded successfully!")
    
    for subject in subjects_list:
        print(f"\n{'#'*60}")
        print(f"GPU {gpu_id} - Processing subject: {subject}")
        print(f"{'#'*60}")
        
        test_data = load_mmlu_redux_arrow(subject, TEST_SAMPLES_PER_SUBJECT)
        if not test_data or len(test_data) == 0:
            print(f"GPU {gpu_id} - ❌ Failed to load data for {subject}, skipping...")
            continue
        
        print(f"\nGPU {gpu_id} - Base Mode for {subject} (batch_size={BATCH_SIZE})")
        base_results, base_correct = run_mode_with_resume_batch(
            test_data, subject, "base", process_batch_base, 
            llm, tokenizer, gpu_id, SAVE_INTERVAL, BATCH_SIZE
        )
        
        print(f"\nGPU {gpu_id} - Roleplay Mode for {subject} (batch_size={BATCH_SIZE})")
        roleplay_results, roleplay_correct = run_mode_with_resume_batch(
            test_data, subject, "roleplay", process_batch_roleplay, 
            llm, tokenizer, gpu_id, SAVE_INTERVAL, BATCH_SIZE
        )
        
        update_global_summary(subject, {
            "base": {"correct": base_correct, "total": len(test_data), "accuracy": base_correct/len(test_data)},
            "roleplay": {"correct": roleplay_correct, "total": len(test_data), "accuracy": roleplay_correct/len(test_data)}
        }, gpu_id)
    
    print(f"\n✅ GPU {gpu_id} completed all {len(subjects_list)} subjects!")
    result_queue.put({"gpu_id": gpu_id, "status": "completed"})


def main():
    print("="*80)
    print("🧪 MMLU-Redux Two-Mode Test (Base + Rich Roleplay) - Four Specific Subjects")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_BASE}")
    print(f"Total subjects: {len(TEST_SUBJECTS)}")
    print(f"Samples per subject: {TEST_SAMPLES_PER_SUBJECT}")
    print(f"Available GPUs: {AVAILABLE_GPUS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Save interval: Every {SAVE_INTERVAL} samples")
    print("="*80)
    
    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    
    
    subjects_per_gpu = [[] for _ in range(len(AVAILABLE_GPUS))]
    for idx, subject in enumerate(TEST_SUBJECTS):
        gpu_idx = idx % len(AVAILABLE_GPUS)
        subjects_per_gpu[gpu_idx].append(subject)
    
    for i, (gpu, subjects) in enumerate(zip(AVAILABLE_GPUS, subjects_per_gpu)):
        print(f"\nGPU {gpu} will process {len(subjects)} subjects: {subjects}")
    
    processes = []
    result_queue = Queue()
    
    for gpu, subjects in zip(AVAILABLE_GPUS, subjects_per_gpu):
        if subjects:
            p = Process(target=run_mode_on_gpu, args=(gpu, subjects, result_queue))
            p.start()
            processes.append(p)
            print(f"✅ Started process for GPU {gpu}")
            time.sleep(5)
    
    print("\n⏳ Waiting for all GPU processes to complete...")
    for p in processes:
        p.join()
    
    print(f"\n{'='*80}")
    print("🎉 FINAL SUMMARY - ALL GPUS COMPLETED!")
    print(f"{'='*80}")
    
    global_summary_file = f"{OUTPUT_BASE}/global_summary.json"
    if os.path.exists(global_summary_file):
        with open(global_summary_file, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        
        summary = final_results["summary"]
        print(f"\n📊 Base Mode:")
        print(f"  Total: {summary['base']['total_correct']}/{summary['base']['total_samples']} = {summary['base']['accuracy']:.2%}")
        print(f"\n📊 Roleplay Mode (with rich subject-specific prompts):")
        print(f"  Total: {summary['roleplay']['total_correct']}/{summary['roleplay']['total_samples']} = {summary['roleplay']['accuracy']:.2%}")
        
        print(f"\n📊 Per-GPU Progress:")
        for gpu_name, progress in final_results["gpu_progress"].items():
            print(f"  {gpu_name}: {len(progress['subjects_completed'])} subjects completed")
        
        print(f"\n📊 Comparison:")
        base_acc = summary['base']['accuracy']
        roleplay_acc = summary['roleplay']['accuracy']
        diff = roleplay_acc - base_acc
        print(f"  Base vs Roleplay: {base_acc:.2%} vs {roleplay_acc:.2%}")
        print(f"  Difference: {diff:+.2%}")
    
    print(f"\n{'='*80}")
    print(f"✅ Tests completed! Results saved to: {OUTPUT_BASE}/")
    print(f"Global summary: {global_summary_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
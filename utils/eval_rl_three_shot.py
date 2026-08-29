import vllm_utils as vllm
import drgrpo_grader as grader
import json
from pathlib import Path
import re
from tqdm import tqdm

def extract_ground_truth(answer: str) -> str|None:
    match = re.search(r"####\s*(.*)", answer)
    if match is None:
        return None
    return match.group(1).strip()

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "r1_zero_three_shot_gsm8k.prompt"
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "gsm8k" / "train.jsonl"
RESULT_PATH = Path(__file__).resolve().parent / "results" / "r1_zero_three_shot_gsm8k_base_eval.jsonl"
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

batch_size = 16
server = vllm.VLLMServer(model_id="allenai/OLMo-2-0425-1B")
server.start()
sampling_params = {}
sampling_params['stop'] = ["</answer>"] 
sampling_params['include_stop_str_in_output'] = True
sampling_params["temperature"] = 1.0
sampling_params["max_tokens"] = 512
sampling_params["n"] = 1
sampling_params["seed"] = 42

questions = []
prompts = []
groundtruths = []
completions = []
category_count = {
    "1":0,
    "2":0,
    "3":0
}


with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        questions.append(data["question"])
        prompt = prompt_template.format(question = data["question"])
        prompts.append(prompt)
        groundtruths.append(extract_ground_truth(data["answer"]))

with RESULT_PATH.open("w", encoding="utf-8") as f:
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Generating",
    ):
        batch_prompts = prompts[start:start + batch_size]

        batch_completions = server.generate_completions(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )

        for j, completion in enumerate(batch_completions):
            sample_index = start + j

            result = grader.r1_zero_reward_fn(
                completion.text,
                ground_truth=groundtruths[sample_index],
            )

            if result["format_reward"] == 1.0 and result["answer_reward"] == 1.0:
                category = "1"
            elif result["format_reward"] == 1.0 and result["answer_reward"] == 0.0:
                category = "2"
            else:
                category = "3"

            record = {
                "id": sample_index,
                "question": questions[sample_index],
                "ground_truth": groundtruths[sample_index],
                "completion": completion.text,
                "format_reward": result["format_reward"],
                "answer_reward": result["answer_reward"],
                "reward": result["reward"],
                "category": category,
            }

            category_count[category] += 1

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(category_count)

import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "models/Qwen3-8B",
    local_files_only=True
)

lengths = []

with Path("outputs/sft_data/test.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)

        # 当前SFT格式的assistant译文
        completion = x.get("completion")
        if isinstance(completion, list):
            target = completion[0]["content"]
        else:
            target = x.get("target_text", "")

        ids = tokenizer(
            target,
            add_special_tokens=False
        )["input_ids"]

        lengths.append(len(ids))

lengths.sort()

def pct(p):
    i = min(int(len(lengths) * p), len(lengths) - 1)
    return lengths[i]

print("样本数:", len(lengths))
print("平均:", round(sum(lengths) / len(lengths), 1))
print("P50:", pct(0.50))
print("P90:", pct(0.90))
print("P95:", pct(0.95))
print("P99:", pct(0.99))
print("最大:", max(lengths))
print(">512:", sum(x > 512 for x in lengths))
print(">768:", sum(x > 768 for x in lengths))
print(">1024:", sum(x > 1024 for x in lengths))
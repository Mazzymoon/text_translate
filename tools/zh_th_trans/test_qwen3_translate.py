import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# 1. 从 ModelScope 下载模型
# =========================

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

print("Downloading / locating model from ModelScope...")

model_dir = snapshot_download(
    MODEL_ID,
    cache_dir=r"F:\project\LLM\text_translate\models"
)

print("Model directory:")
print(model_dir)


# =========================
# 2. 从本地目录加载 tokenizer
# =========================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True
)


# =========================
# 3. 从本地目录加载模型
# =========================

print("Loading model...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=quantization_config,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

model.eval()


# =========================
# 4. 测试中文 → 泰语翻译
# =========================

source_text = """
人工智能技术近年来快速发展，并逐渐应用于制造业、医疗、教育和交通等领域。
随着大模型技术不断成熟，企业正在探索利用人工智能提高生产效率和降低运营成本。
"""

prompt = f"""请将下面的中文准确翻译成自然、流畅的泰语。

要求：
1. 忠实保留原文含义。
2. 不得遗漏信息。
3. 不得增加原文不存在的信息。
4. 数字、日期、专有名词应尽可能准确保留。
5. 只输出泰语译文，不要解释，不要添加“翻译如下”等内容。

中文原文：
{source_text}
"""

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    [text],
    return_tensors="pt",
).to(model.device)


# =========================
# 5. 模型生成
# =========================

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )


# =========================
# 6. 提取模型新生成的内容
# =========================

# 删除输入提示词
generated_ids = outputs[:, inputs.input_ids.shape[1]:]

translation = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]


# =========================
# 7. 输出结果
# =========================

print("\n===== 中文 =====")
print(source_text)

print("\n===== 泰语 =====")
print(translation)

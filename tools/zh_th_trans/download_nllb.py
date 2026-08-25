from modelscope import snapshot_download

model_dir = snapshot_download(
    "facebook/nllb-200-distilled-600M",
    local_dir=r"F:\project\LLM\text_translate\models\models\nllb-200-distilled-600M"
)

print(model_dir)
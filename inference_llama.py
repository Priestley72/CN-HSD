import re
import torch
import json
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_PATH       = "./models/Qwen3-8B-llama-epoch8"
DATA_PATH        = "./data/enhanced_test1.json"
FORMAT_DATA_PATH = "./data/format_enhanced_test1.json"
SAVE_PATH        = "./data/result_epoch8_enhanced.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模型和Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # 使用 bfloat16 数据类型
    device_map=device,          # 自动分配GPU
    # load_in_8bit=True           # 使用 8-bit 量化
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,     # 信任代码 适用Qwen
    padding_side='left',        # 左侧填充
    enable_thinking=False,      # 禁用思考模式
    use_fast=False              # 慢速tokenizer 提高中文精度
)
format_test_dataset = json.load(open(FORMAT_DATA_PATH, "r", encoding="utf-8"))
test_dataset = json.load(open(DATA_PATH, "r", encoding="utf-8"))
def generate_multiple_outputs(input_texts, num_outputs):
    inputs = tokenizer(input_texts, return_tensors="pt",
                       padding=True, truncation=True).to(device)
    # 使用num_return_sequences生成多个输出
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.0001,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=num_outputs
    )
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # 将结果重新组织为每个输入对应num_outputs个输出
    organized_outputs = []
    for i in range(len(input_texts)):
        start_idx = i * num_outputs
        end_idx = start_idx + num_outputs
        organized_outputs.append(generated_texts[start_idx:end_idx])
    return organized_outputs
def extract_output(generated_text):
    # 正则提取输出内容
    pattern = r"###输出:\s*(.*?)(\[END\])"
    match = re.search(pattern, generated_text, re.DOTALL)
    return match.group(1).strip() + " [END]" if match else ""
batch_size = 4      # 批处理大小
num_infer = 5       # 平行推理次数
results = []
for i in tqdm(range(0, len(format_test_dataset), batch_size), desc="Inference", unit="batch"):
    batch_texts = [item["input"] for item in format_test_dataset[i: i + batch_size]]
    # 平行推理 为每个样本生成num_inferences个输出
    batch_outputs = generate_multiple_outputs(batch_texts, num_infer)
    for j, outputs in enumerate(batch_outputs):
        # 提取结果 统计分布 选择出现次数最多的结果
        extracted_outputs = [extract_output(text) for text in outputs]
        output_counter = Counter(extracted_outputs)
        most_common = output_counter.most_common(1)
        final_output, count = most_common[0] if most_common else ("", 0)
        # 计算一致性比例
        consist_ratio = count / num_infer
        results.append({
            "id": test_dataset[i + j]["id"],
            "content": test_dataset[i + j]["content"],
            "output": final_output,
            "consist_ratio": consist_ratio,
            "all_outputs": extracted_outputs  # 保存所有推理结果用于分析原因
        })
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"测试完成 结果已保存到{SAVE_PATH}")

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
batch_size = 4      # 批处理大小
M = 5               # 平行推理次数
K = 3               # K-shot
max_retries = 3     # 最大重试次数
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
results = []
for i in tqdm(range(0, len(format_test_dataset), batch_size), desc="Inference", unit="batch"):
    batch_texts = [item["input"] for item in format_test_dataset[i: i + batch_size]]
    for j in range(len(batch_texts)):
        retry_count = 0
        final_output = ""
        all_outputs = []
        consist_ratio = 0.0
        current_text = batch_texts[j]
        while retry_count <= max_retries:
            # 生成M个输出
            current_outputs = generate_multiple_outputs([current_text], M)[0]
            extracted_outputs = [extract_output(text) for text in current_outputs]
            all_outputs.extend(extracted_outputs)
            # 统计当前轮次的输出分布
            output_counter = Counter(extracted_outputs)
            # 检查是否有输出达到阈值K
            for output, count in output_counter.most_common():
                if count >= K:
                    final_output = output
                    consist_ratio = count / M
                    break
            # 如果找到满足条件的输出，退出循环
            if final_output: break
            retry_count += 1
        # 如果所有重试后仍无满足条件的输出，则选择出现次数最多的
        if not final_output and all_outputs:
            output_counter = Counter(all_outputs)
            most_common = output_counter.most_common(1)
            if most_common:
                final_output, count = most_common[0]
                consist_ratio = count / (M * (retry_count + 1))
        # 记录结果
        results.append({
            "id": format_test_dataset[i + j]["id"],
            "content": format_test_dataset[i + j]["content"],
            "output": final_output,
            "consist_ratio": consist_ratio,
            "all_outputs": all_outputs,
        })
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"测试完成 结果已保存到{SAVE_PATH}")

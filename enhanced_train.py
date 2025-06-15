import json
import torch
import random
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
random.seed(3407)
# 配置参数
batch_size = 32     # 批处理大小
TOP_K = 1           # 相似样本数量
split_ratio = 0.5   # 训练集分割比例
train_path = "./data/train.json"
save_path = "./data/enhanced_train.json"
model_path = "./models/chinese-lert-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载数据
train_data = json.load(open(train_path, 'r', encoding='utf-8'))
# 随机打乱并分割训练集
random.shuffle(train_data)
split_idx = int(len(train_data) * split_ratio)
train_part1 = train_data[:split_idx]
train_part2 = train_data[split_idx:]
# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
model.eval()
# 手动计算文本embedding
def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="计算embedding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)
# 均值池化获取句子embedding
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# 找出非仇恨样本索引
def find_non_hate_indices(outputs):
    non_hate_indices = []
    for idx, output in enumerate(outputs):
        if "[SEP]" in output:
            hate_flags = [x.split("|")[-1].strip() for x in output.split("[SEP]")]
        else:
            hate_flags = [output.split("|")[-1].strip().replace(" [END]", "")]
        is_all_non_hate = all("否" in flag or "non-hate" in flag.lower() for flag in hate_flags)
        if is_all_non_hate:
            non_hate_indices.append(idx)
    return non_hate_indices
# 准备两个部分的数据
contents1 = [item["content"] for item in train_part1]
outputs1 = [item["output"] for item in train_part1]
contents2 = [item["content"] for item in train_part2]
outputs2 = [item["output"] for item in train_part2]
# 计算embedding
print("计算第一部分embedding...")
embeddings1 = get_embeddings(contents1, model, tokenizer, device, batch_size)
print("计算第二部分embedding...")
embeddings2 = get_embeddings(contents2, model, tokenizer, device, batch_size)
# 找出两部分的non-hate样本
non_hate_indices1 = find_non_hate_indices(outputs1)
non_hate_indices2 = find_non_hate_indices(outputs2)
def enhance_dataset(base_data, base_contents, base_embeddings,
                    ref_contents, ref_outputs, ref_embeddings, ref_non_hate_indices):
    enhanced_data = []
    for i, content in enumerate(tqdm(base_contents, desc="增强处理")):
        # 取Top-K最相似的样本
        cos_scores = util.cos_sim(base_embeddings[i].unsqueeze(0), ref_embeddings)[0]
        top_k_indices = torch.topk(cos_scores, k=TOP_K).indices.tolist()
        # 随机取一个non-hate样本
        non_hate_idx = random.choice(ref_non_hate_indices)
        # 构建增强数据
        similar_data = []
        for idx in top_k_indices:
            similar_data.append({
                "content": ref_contents[idx],
                "output": ref_outputs[idx],
            })
        non_hate_data = [{
            "content": ref_contents[non_hate_idx],
            "output": ref_outputs[non_hate_idx],
        }]
        enhanced_item = {
            "id": base_data[i]["id"],
            "content": base_data[i]["content"],
            "output": base_data[i]["output"],
            "similar_data": similar_data,
            "non_hate_data": non_hate_data
        }
        enhanced_data.append(enhanced_item)
    return enhanced_data
print("第一部分数据增强...")
enhanced_part1 = enhance_dataset(train_part1, contents1, embeddings1,
                                 contents2, outputs2, embeddings2, non_hate_indices2)
print("第二部分数据增强...")
enhanced_part2 = enhance_dataset(train_part2, contents2, embeddings2,
                                 contents1, outputs1, embeddings1, non_hate_indices1)
final_enhanced_data = enhanced_part1 + enhanced_part2
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(final_enhanced_data, f, ensure_ascii=False, indent=4)
print(f"增强训练集构建完成 保存到{save_path}")
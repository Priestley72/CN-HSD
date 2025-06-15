import json
import torch
import random
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
# 合法目标群体
batch_size = 32
top_k = 1
train_path = "./data/train.json"
test_path = "./data/test1.json"
save_path = "./data/enhanced_test1.json"
model_path = "./models/chinese-lert-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载数据 模型 Tokenizer
train_data = json.load(open(train_path, 'r', encoding='utf-8'))
test_data = json.load(open(test_path, 'r', encoding='utf-8'))
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
model.eval()
def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    """手动计算文本embedding"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="计算embedding"):
        batch = texts[i:i + batch_size]
        # Tokenize并移动到设备
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)
        # 计算embedding
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用均值池化获取句子embedding
        batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)
def mean_pooling(model_output, attention_mask):
    """均值池化获取句子embedding"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# 提取文本和output
train_contents = [item["content"] for item in train_data]
train_outputs = [item["output"] for item in train_data]
test_contents = [item["content"] for item in test_data]
# 找出纯粹非仇恨样本的索引
non_hate_indices = []
for idx, output in enumerate(train_outputs):
    if "[SEP]" in output:
        hate_flags = [x.split("|")[-1].strip() for x in output.split("[SEP]")]
    else:
        hate_flags = [output.split("|")[-1].strip().replace(" [END]", "")]
    is_all_non_hate = all("否" in flag or "non-hate" in flag.lower() for flag in hate_flags)
    if is_all_non_hate:
        non_hate_indices.append(idx)
# 计算训练集和测试集的embedding
train_embeddings = get_embeddings(train_contents, model, tokenizer, device, batch_size)
test_embeddings = get_embeddings(test_contents, model, tokenizer, device, batch_size)
# 样本增强
TOP_K = top_k   # 取Top-K最相似的训练样本
enhanced_data = []
for i, test_content in tqdm(enumerate(test_contents),
                            total=len(test_contents),
                            desc="Processing test contents"):
    test_embedding = test_embeddings[i]
    # 计算与所有训练样本的相似度
    cos_scores = util.cos_sim(test_embedding, train_embeddings)[0]
    # 取Top-K最相似的训练样本
    top_k_indices = torch.topk(cos_scores, k=TOP_K).indices.tolist()
    # 随机取一个非仇恨样本
    non_hate_idx = random.choice(non_hate_indices)
    # 构建增强数据
    similar_data = []
    for idx in top_k_indices:
        similar_data.append({
            "content": train_contents[idx],
            "output": train_outputs[idx],
        })
    non_hate_data = [{
        "content": train_contents[non_hate_idx],
        "output": train_outputs[non_hate_idx],
    }]
    enhanced_item = {
        "id": test_data[i]['id'],
        "content": test_content,
        "similar_data": similar_data,
        "non_hate_data": non_hate_data,
    }
    enhanced_data.append(enhanced_item)
# 保存增强后的数据
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(enhanced_data, f, ensure_ascii=False, indent=4)
print(f"数据增强完成 已保存到{save_path}")
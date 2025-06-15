import json
from utils import text2tuples

def count_hate_quads(file_path):
    """统计文件中仇恨四元组的数量"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total_quads = 0
    hate_quads = 0
    for item in data:
        if 'output' not in item:
            continue
        quads = text2tuples(item['output'])
        total_quads += len(quads)
        for quad in quads:
            # 判断是否为仇恨内容 (根据Hateful字段)
            if quad['Hateful'].lower().strip() == 'hate':
                hate_quads += 1
    return total_quads, hate_quads
# 文件路径
test_file = "./data/positive_test1.json"
train_file = "./data/train.json"
test_total, test_hate = count_hate_quads(test_file)
train_total, train_hate = count_hate_quads(train_file)
print(f"测试集统计结果:")
print(f"总四元组数量: {test_total}")
print(f"仇恨四元组数量: {test_hate}")
print(f"仇恨比例: {test_hate / test_total:.2%}\n")

print(f"训练集统计结果:")
print(f"总四元组数量: {train_total}")
print(f"仇恨四元组数量: {train_hate}")
print(f"仇恨比例: {train_hate / train_total:.2%}")
# 输出更详细的分组统计
def count_by_group(file_path):
    """统计各仇恨群体的分布"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    group_counter = {}
    for item in data:
        if 'output' not in item:
            continue
        quads = text2tuples(item['output'])
        for quad in quads:
            if quad['Hateful'].lower().strip() == 'hate':
                groups = [g.strip() for g in quad['Targeted_Group'].split(',')]
                for group in groups:
                    group_counter[group] = group_counter.get(group, 0) + 1
    return group_counter
print("\n测试集仇恨群体分布:")
test_groups = count_by_group(test_file)
for group, count in sorted(test_groups.items(), key=lambda x: x[1], reverse=True):
    print(f"{group}: {count} ({count / test_hate:.1%})")
print("\n训练集仇恨群体分布:")
train_groups = count_by_group(train_file)
for group, count in sorted(train_groups.items(), key=lambda x: x[1], reverse=True):
    print(f"{group}: {count} ({count / train_hate:.1%})")
import json
from utils import calculate_f1
# PRED_FILE = "./data/result_epoch10.json"
PRED_FILE = "./data/result_epoch8_enhanced.json"
# PRED_FILE = "./data/result_best.json"
TRUE_FILE = "./data/positive_test1.json"
pred_data = json.load(open(PRED_FILE, 'r', encoding='utf-8'))
true_data = json.load(open(TRUE_FILE, 'r', encoding='utf-8'))
# 评估结果
results = calculate_f1(pred_data, true_data)
# 输出结果
print(f"Hard Score\n"
      f"P={results['hard_precision']:.4f}\t"
      f"R={results['hard_recall']:.4f}\t"
      f"F1={results['hard_f1']:.4f}")
print(f"Soft Score\n"
      f"P={results['soft_precision']:.4f}\t"
      f"R={results['soft_recall']:.4f}\t"
      f"F1={results['soft_f1']:.4f}")
print(f"Avg Score\n"
      f"F1={results['avg_f1']:.4f}")

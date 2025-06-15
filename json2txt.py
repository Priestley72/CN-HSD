import re
import json
from utils import text2tuples, tuples2text
DATA_PATH = "./data/result_epoch8_enhanced.json"
SAVE_PATH = "./data/result_epoch8_enhanced.txt"
data = json.load(open(DATA_PATH, "r", encoding="utf-8"))
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for item in data:
        text = item["output"]
        # 去除所有空格（包括空格、制表符等）
        # text = text.replace(" ", "").replace("\t", "")
        # text = text.replace("\n", "[SEP]")
        # 匹配hate后紧邻汉字的情况（hate汉字）
        # pattern1 = re.compile(r'hate([\u4e00-\u9fff])')
        # text = pattern1.sub(r'hate[SEP]\1', text)
        # 匹配###输出:错误切分的情况
        # text = re.sub(r'^###ouput:', '', text)
        # text = re.sub(r'^###输出:', '', text)
        # text = re.sub(r'(?<=\S)###输出:|###输出:(?=\S)', '[SEP]', text)
        # text = re.sub(r'(?<=\S)###output:|###output:(?=\S)', '[SEP]', text)
        # output = tuples2text(text2tuples(text))
        output = text
        f.write(f"{output}\n")
print(f"转换完成 已保存到{SAVE_PATH}")

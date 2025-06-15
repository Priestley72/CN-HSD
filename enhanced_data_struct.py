import json
data_path = "./data/enhanced_test1.json"
save_path = "./data/format_enhanced_test1.json"
data = json.load(open(data_path, 'r', encoding='utf-8'))
sys_prompt = "你是一名内容审查专家，能够分析中文语句并从中提取一个或多个四元组, 每个四元组按以下顺序组成“评论对象 | 对象观点 | 仇恨群体 | 是否仇恨”\n"
instruction = """请严格执行以下要求：
- 尽量使用中文或句中原有片段，不使用其他语言。
- “评论对象”可以是具体人名、称呼、群体或抽象概念，如果没有明确对象，则填写“NULL”，其他结果需保留原词形式。
- “对象观点”针对“评论对象”所发表的核心观点、评价、情绪或行为表述，应为关键片段，尽量保持原句表述，简明扼要。
- “仇恨群体”只包括(LGBTQ、Region、Sexism、Racism、others、non-hate)，其中Region表示地域歧视，Racism表示种族歧视，Sexism表示性别歧视，LGBTQ表示歧视性少数群体，others表示歧视其他特定群体，non-hate表示非仇恨或中性群体，同一四元组可涉及多个仇恨群体
- “仇恨标签”标注该评论对象-论点-群体组合是否构成对群体的仇恨表达，hate表示具有明显的仇恨或攻击性，non-hate表示为中性或不构成群体仇恨
- 元素之间用" | "分隔，多个四元组之间只允许用" [SEP] "分隔，最后都以" [END]"结尾，注意空格
- 禁止含有换行符或制表符等无关符号，禁止添加解释性文字等无关内容，禁止重复同一四元组\n"""
format_data = []
for item in data:
    format_data.append(
        {
            "instruction": "",
            "input": sys_prompt + instruction + \
                     f"样例一：\n"
                     f"输入“{item['similar_data'][0]['content']}”, 输出“{item['similar_data'][0]['output']}”\n"
                     f"样例二：\n"
                     f"输入“{item['non_hate_data'][0]['content']}”, 输出“{item['non_hate_data'][0]['output']}”\n"
                     f"现在请分析以下句子并提取四元组，单纯回复四元组结果，禁止回复思考内容，禁止回复无关内容，禁止重复同一四元组:\n" + \
                     item["content"] + " /no_think\n###输出:",
            # "output": item["output"],
            "output": "",
        }
    )
with open(save_path, 'w', encoding='utf-8') as file:
    json.dump(format_data, file, ensure_ascii=False, indent=4)
print(f"数据格式化完成 已保存到{save_path}")
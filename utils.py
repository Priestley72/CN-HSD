import re
import difflib
# region 格式转换
def text2tuples(text):
    """
    文本解析为四元组列表
    from    "Target | Argument | Targeted Group | Hateful [SEP] ... [END]"
    to      [{'Target': ..., 'Argument': ..., 'Targeted_Group': ..., 'Hateful': ...}, ...]
    """
    tuples = []
    for quad_text in re.split(r'\s*\[SEP\]\s*', text):
        quad_text = re.sub(r'\s*\[END\]\s*$', '', quad_text.strip())
        # 文本为空则跳过
        if not quad_text: continue
        # 按|分割各元素
        elements = [elem.strip() for elem in quad_text.split('|')]
        # 确保有4个元素
        if len(elements) == 4:
            tuples.append({
                'Target': elements[0],
                'Argument': elements[1],
                'Targeted_Group': elements[2],
                'Hateful': elements[3]
            })
    return tuples
def tuples2text(tuples):
    """
    四元组列表转换为文本
    from    [{'Target': ..., 'Argument': ..., 'Targeted_Group': ..., 'Hateful': ...}, ...]
    to      "Target | Argument | Targeted Group | Hateful [SEP] ... [END]"
    """
    text = ''
    for i, quad in enumerate(tuples):
        quad_text = f"{quad['Target']} | {quad['Argument']} | {quad['Targeted_Group']} | {quad['Hateful']}"
        text += quad_text
        if i < len(tuples) - 1:
            text += ' [SEP] '
    text += ' [END]'
    return text
def json2txt(data):
    """
    将JSON文件输出为结果txt文件
    """
    text = ''
    for item in data:
        item_text = f"{item['output']}\n"
        text += item_text
    return text
# endregion
# region F1评估计算
def calculate_similarity(text1, text2):
    """
    计算字符串相似度
    要求使用difflib.SequenceMatcher
    """
    seq_matcher = difflib.SequenceMatcher(None, text1, text2)
    similarity = seq_matcher.ratio()
    return similarity
def if_match_hard(tuple1, tuple2):
    """
    硬匹配
    四元组元素需完全一致
    """
    return (tuple1['Target'] == tuple2['Target'] and
            tuple1['Argument'] == tuple2['Argument'] and
            tuple1['Targeted_Group'] == tuple2['Targeted_Group'] and
            tuple1['Hateful'] == tuple2['Hateful'])
def if_match_soft(tuple1, tuple2):
    """
    软匹配
    Targeted_Group  Hateful     要求一致
    Target          Argument    相似度需大于0.5
    """
    if (tuple1['Targeted_Group'] != tuple2['Targeted_Group'] or
        tuple1['Hateful'] != tuple2['Hateful']):
        return False
    target_similarity = calculate_similarity(tuple1['Target'], tuple2['Target'])
    argument_similarity = calculate_similarity(tuple1['Argument'], tuple2['Argument'])
    # 相似度都大于0.5则匹配成功
    return target_similarity > 0.5 and argument_similarity > 0.5
def calculate_f1(pred_data, true_data):
    """
    计算F1分数
    硬匹配和软匹配都计算一次
    """
    # ID->四元组列表
    id2tuples_true, id2tuples_pred = {}, {}
    # 解析标准答案
    for item1, item2 in zip(pred_data, true_data):
        id1, id2 = item1['id'], item2['id']
        tuples1 = text2tuples(item1['output'])
        tuples2 = text2tuples(item2['output'])
        id2tuples_pred[id2] = tuples1
        id2tuples_true[id1] = tuples2
    # 收集每个示例的硬匹配和软匹配结果，用于计算总体F1
    all_hard_tp, all_hard_fp, all_hard_fn = 0, 0, 0
    all_soft_tp, all_soft_fp, all_soft_fn = 0, 0, 0
    # 对每个示例进行评估
    for item_id in id2tuples_true:
        tuples_pred = id2tuples_pred.get(item_id, [])
        tuples_true = id2tuples_true.get(item_id, [])
        # 硬匹配评估
        hard_matched_pred = set()
        hard_matched_true = set()
        for i, pred_tuple in enumerate(tuples_pred):
            for j, true_tuple in enumerate(tuples_true):
                if j in hard_matched_true:
                    continue
                if if_match_hard(pred_tuple, true_tuple):
                    hard_matched_pred.add(i)
                    hard_matched_true.add(j)
                    break
        hard_tp = len(hard_matched_pred)
        hard_fp = len(tuples_pred) - hard_tp
        hard_fn = len(tuples_true) - len(hard_matched_true)
        all_hard_tp += hard_tp
        all_hard_fp += hard_fp
        all_hard_fn += hard_fn
        # 软匹配评估
        soft_matched_pred = set()
        soft_matched_true = set()
        for i, pred_tuple in enumerate(tuples_pred):
            for j, true_tuple in enumerate(tuples_true):
                if j in soft_matched_true:
                    continue
                if if_match_soft(pred_tuple, true_tuple):
                    soft_matched_pred.add(i)
                    soft_matched_true.add(j)
                    break
        soft_tp = len(soft_matched_pred)
        soft_fp = len(tuples_pred) - soft_tp
        soft_fn = len(tuples_true) - len(soft_matched_true)
        all_soft_tp += soft_tp
        all_soft_fp += soft_fp
        all_soft_fn += soft_fn
    # 计算硬匹配的精确率、召回率和F1分数
    hard_precision = all_hard_tp / (all_hard_tp + all_hard_fp) if (all_hard_tp + all_hard_fp) > 0 else 0
    hard_recall = all_hard_tp / (all_hard_tp + all_hard_fn) if (all_hard_tp + all_hard_fn) > 0 else 0
    hard_f1 = 2 * hard_precision * hard_recall / (hard_precision + hard_recall) if (hard_precision + hard_recall) > 0 else 0
    # 计算软匹配的精确率、召回率和F1分数
    soft_precision = all_soft_tp / (all_soft_tp + all_soft_fp) if (all_soft_tp + all_soft_fp) > 0 else 0
    soft_recall = all_soft_tp / (all_soft_tp + all_soft_fn) if (all_soft_tp + all_soft_fn) > 0 else 0
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0
    # 计算平均F1分数
    avg_f1 = (hard_f1 + soft_f1) / 2
    return {
        'hard_precision': hard_precision,
        'hard_recall': hard_recall,
        'hard_f1': hard_f1,
        'soft_precision': soft_precision,
        'soft_recall': soft_recall,
        'soft_f1': soft_f1,
        'avg_f1': avg_f1
    }
# endregion


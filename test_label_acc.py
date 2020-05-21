import argparse
import pickle
# <<<<<<< unsupervised-labeling
import json
# =======
from data import QAFullInputFeatures, SQuADFullInputFeatures
from typing import List

# >>>>>>> tri-training

parser = argparse.ArgumentParser()
parser.add_argument('--predict_labels', type=str, required=True)
parser.add_argument('--feature_file', type=str, required=True)

args = parser.parse_args()

# <<<<<<< unsupervised-labeling
# prediction = json.load(open(args.predict_labels, 'r'))
# label = pickle.load(open(args.feature_file, 'rb'))
#
#
# num_right = 0
# num_total = 0
# for key, value in prediction.items():
#     pid = key
#     prediction_id = value['sentence_id']
#     doc_id = value['doc_span_index']
#     if label[pid] == -1 or prediction_id == -1:
#         continue
#     if abs(prediction_id - label[pid]['sentence_id']) <= 0 and doc_id == label[pid]['doc_span']:
#         num_right += 1
#     num_total += 1
#
# print('right %d, total %d, proportion %f' % (num_right, num_total, float(num_right) / num_total))

# =======
with open(args.predict_labels, 'r') as reader:
    pred_labels = json.load(reader)

with open(args.feature_file, 'rb') as reader:
    features: List[QAFullInputFeatures] = pickle.load(reader)

total = 0
correct = 0
for feature in features:
    # key = (feature.qas_id, feature.doc_span_index)
    # if key in pred_labels and pred_labels[key]['sentence_id'] != -1:
    #     total += 1
    #     if pred_labels[key]['sentence_id'] == feature.sentence_id:
    #         correct += 1
    if feature.qas_id in pred_labels and feature.doc_span_index == pred_labels[feature.qas_id]['doc_span_index'] and \
            pred_labels[feature.qas_id]['sentence_id'] != -1:
        total += 1
        if pred_labels[feature.qas_id]['sentence_id'] == feature.sentence_id:
            correct += 1

print(f'Label accuracy: {correct} / {total} = {correct * 1.0 / total}')
# >>>>>>> tri-training

# running
# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py
# --feature_file /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent4_view0/sentence_id_file.json
# Label accuracy: 5027 / 19463 = 0.25828495093253867
#
# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py
# --feature_file /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent4_view1/sentence_id_file.json
# Label accuracy: 5107 / 19286 = 0.2648034843928238

# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py
# --feature_file /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent0_view1/sentence_id_file_recurrent4.json.merge
# Label accuracy: 1467 / 3992 = 0.3674849699398798
# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py --feature_file
# /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent0_view0/sentence_id_file_recurrent4.json.merge
# Label accuracy: 1310 / 3793 = 0.3453730556287899

# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py
# --feature_file /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent0_view1/sentence_id_file.json
# Label accuracy: 125 / 1675 = 0.07462686567164178

# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py
# --feature_file /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent0_view0/sentence_id_file.json
# Label accuracy: 2332 / 6814 = 0.3422365717640153

# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py
# --feature_file /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent0_view0/sentence_id_file_recurrent1.json.merge
# Label accuracy: 64 / 795 = 0.08050314465408805

# (torch-transformers) jiaofangkai@gpu-sigma:~/CoQA-Challenge/BERT_RC$ python test_label_acc.py
# --feature_file /home/jiaofangkai/CoQA-Challenge/BERT/max_f1/coqa-train-v1.0.json_bert-base-uncased_512_128_385_2_coqa
# --predict_labels
# /home/jiaofangkai/CoQA-Challenge/BERT_RC/experiments/coqa/co-training/v7.0_1000/recurrent0_view1/sentence_id_file_recurrent1.json.merge
# Label accuracy: 434 / 996 = 0.4357429718875502
# =========================================================================================================================


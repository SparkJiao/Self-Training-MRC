import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True, help='Old sentence label predictions file.')
parser.add_argument('--predict2', type=str, required=True, help='New sentence label predictions file.')
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--label_threshold', type=float, required=True)
parser.add_argument('--multi_evidence', default=False, action='store_true', help='If true, sentence probability will be merged.'
                                                                                 'Otherwise, sentence ids will be merged.')
parser.add_argument('--k', type=int, default=1000, help='Select only the top `k` new predictions to merge.')

args = parser.parse_args()

label_threshold = args.label_threshold
k = args.k

predictions1 = json.load(open(args.predict1, 'r'))
predictions2 = json.load(open(args.predict2, 'r'))

# Version 1
# predictions2_map = {}
# if args.multi_evidence:
#     for qa_id, pred in predictions2.item():
#         sentence_prob = pred['sentence_prob']
#         label_prob = pred['choice_prob']
#         for choice_index, choice_prob in enumerate(sentence_prob):
#             for sentence_index, sentence_prob in enumerate(choice_prob):
#                 predictions2_map[(qa_id, choice_index, sentence_index)] = (label_prob, sentence_prob)
#     sorted_predictions2 = sorted(predictions2_map.items(), key=lambda x: x[1][0] * x[1][1], reverse=True)
#
#     cnt = 0
#     for (qa_id, choice_index, sentence_index), (label_prob, sentence_prob) in predictions2:
#         if qa_id in predictions1:
#             choice_prob = predictions1[qa_id]['sentence_prob'][choice_index]
#             if any(sentence != -1 for sentence in choice_prob):
#                 continue
#             else:
#                 predictions1[qa_id]['sentence_prob'][choice_index][sentence_index] = sentence_prob
#                 cnt += 1
#         else:
#             predictions1[qa_id] = predictions2[qa_id]
#             choice_num = len(predictions2[qa_id]['sentence_prob'])
#             max_sentence_num = len(predictions2[qa_id]['sentence_prob'][0])
#             predictions1[qa_id]['sentence_prob'] = [[-1] * max_sentence_num] * choice_num
#             predictions1[qa_id]['sentence_label'] = [[-1] * max_sentence_num] * choice_num
#             predictions1[qa_id]['max_weight_index'] = [-1] * choice_num
#
#             predictions1[qa_id]['sentence_prob'][choice_index][sentence_index] = sentence_prob
#             cnt += 1
#         if cnt > args.k:
#             break

filtered_predictions2 = filter(lambda x: x[1]['choice_prob'] > label_threshold and x[1]['choice_prediction'] == x[1]['answer'],
                               predictions2.items())

if args.multi_evidence:
    pred_set = []
    for qa_id, pred in filtered_predictions2:
        options_sentence_prob_sum = pred['sentence_prob_sum']
        choice_prob = pred['choice_prob']
        for option_index, sentence_prob_sum in enumerate(options_sentence_prob_sum):
            pred_set.append(((qa_id, option_index), (choice_prob, sentence_prob_sum)))
    sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
    selected_ids = defaultdict(list)
    for (qa_id, option_index), (_, _) in sorted_predictions[:k]:
        selected_ids[qa_id].append(option_index)
else:
    pred_set = []
    for qa_id, pred in filtered_predictions2:
        max_weights = pred['max_weight']
        choice_prob = pred['choice_prob']
        for option_index, max_weight in enumerate(max_weights):
            pred_set.append(((qa_id, option_index), (choice_prob, max_weight)))
    sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
    selected_ids = defaultdict(list)
    for (qa_id, option_index), (_, _) in sorted_predictions[:k]:
        selected_ids[qa_id].append(option_index)


labeled = 0
for qa_id, option_list in selected_ids.items():
    if qa_id in predictions1:
        for option_index in option_list:
            flag = False
            if predictions1[qa_id]['sentence_label'][option_index] is None:
                predictions1[qa_id]['sentence_label'][option_index] = predictions2[qa_id]['sentence_label'][option_index]
                flag = True
            else:
                pass

            if predictions1[qa_id]['sentence_prob'][option_index] is None:
                predictions1[qa_id]['sentence_prob'][option_index] = predictions2[qa_id]['sentence_prob'][option_index]
                flag = True
            else:
                pass

            if predictions1[qa_id]['max_weight_index'][option_index] is None:
                predictions1[qa_id]['max_weight_index'][option_index] = predictions2[qa_id]['max_weight_index'][option_index]
                flag = True
            else:
                pass
            if flag:
                labeled += 1
    else:
        predictions1[qa_id] = predictions2[qa_id]
        option_num = len(predictions2[qa_id]['raw_sentence_prob'])
        for idx in range(option_num):
            flag = False
            if idx not in option_list:
                predictions1[qa_id]['sentence_prob'][idx] = None
                predictions1[qa_id]['sentence_label'][idx] = None
                predictions1[qa_id]['max_weight_index'][idx] = None
            else:
                flag = True
            if flag:
                labeled += 1

print(f'Add new labels {labeled}')
with open(args.output_file, 'w') as f:
    json.dump(predictions1, f, indent=2)

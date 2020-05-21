import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--predict', type=str, required=True)
parser.add_argument('--k', type=int, default=2000)
parser.add_argument('--label_threshold', type=float, required=True)
parser.add_argument('--multi_evidence', default=False, action='store_true')

args = parser.parse_args()
k = args.k
label_threshold = args.label_threshold  # This parameter is the same as that of predict_sentence_ids method.

# Version 1.0
# sentence_labels = json.load(open(args.predict_file, 'r'))
#
# # Choose predictions above the choice prediction threshold
# sorted_predictions = sorted(sentence_labels.items(), key=lambda x: x[1]["choice_prob"], reverse=True)
# selected_output = {x[0]: x[1] for x in sorted_predictions if x[1]["choice_prob"] > label_threshold}
# # if len(selected_output) > k:
# #     selected_output = selected_output[:k]
#
# if args.multi_evidence:
#     prob_pool = []
#     for qa_id, pred in selected_output.items():
#         sentence_prob = pred['sentence_prob']
#         for choice_index, choice_prob in enumerate(sentence_prob):
#             prob_pool.extend(
#                 [((qa_id, choice_index, sentence_index), sentence) for sentence_index, sentence in enumerate(choice_prob) if
#                  sentence != -1])
#
#     prob_pool = sorted(prob_pool, key=lambda x: x[1], reverse=True)
#     prob_pool = list(map(lambda x: x[0], prob_pool[:k]))
#
#     for qa_id, pred in selected_output.items():
#         sentence_prob = pred['sentence_prob']
#         for choice_index, choice_prob in enumerate(sentence_prob):
#             for sentence_index, sentence in enumerate(choice_prob):
#                 if sentence == -1:
#                     continue
#                 if (qa_id, choice_index, sentence_index) not in prob_pool:
#                     selected_output[qa_id]['sentence_prob'][choice_index][sentence_index] = -1
#                     # Currently it doesn't matter if you don't mask following values as -1
#                     selected_output[qa_id]['sentence_label'][choice_index][sentence_index] = -1
#                     if selected_output[qa_id]['max_weight_index'][choice_index] == sentence_index:
#                         selected_output[qa_id]['max_weight_index'][choice_index] = -1
#
# else:
#     sentence_id_pool = []
#     for qa_id, pred in selected_output.items():
#         max_weight = pred['max_weight']
#         max_weight_index = pred['max_weight_index']
#         sentence_id_pool.extend(
#             [((qa_id, choice_index), max_weight[choice_index]) for choice_index, weight_index in enumerate(max_weight_index) if
#              weight_index != -1])
#     sentence_id_pool = sorted(sentence_id_pool, key=lambda x: x[1], reverse=True)
#     sentence_id_pool = list(map(lambda x: x[0], sentence_id_pool[:k]))
#
#     for qa_id, pred in selected_output.items():
#         max_weight_index = pred['max_weight_index']
#         for choice_index, weight_index in enumerate(max_weight_index):
#             if weight_index == -1:
#                 continue
#             if (qa_id, choice_index) not in sentence_id_pool:
#                 selected_output[qa_id]['max_weight_index'][choice_index] = -1
#
# with open(f"{args.predict_file}-top{k}-{label_threshold}-{args.multi_evidence}", 'w') as f:
#     json.dump(selected_output, f, indent=2)

# Version 2
predictions = json.load(open(args.predict, 'r'))

# We need this filter because we output all predictions, including wrong predictions and above restriction.
filtered_predictions = filter(lambda x: x[1]['choice_prob'] > label_threshold and x[1]['choice_prediction'] == x[1]['answer'],
                              predictions.items())
if args.multi_evidence:
    pred_set = []
    for qa_id, pred in filtered_predictions:
        options_sentence_prob_sum = pred['sentence_prob_sum']
        choice_prob = pred['choice_prob']
        for option_index, sentence_prob_sum in enumerate(options_sentence_prob_sum):
            pred_set.append(((qa_id, option_index), (choice_prob, sentence_prob_sum)))

    sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
    selected_ids = defaultdict(list)
    for ((qa_id, option_index), (_, _)) in sorted_predictions[:k]:
        selected_ids[qa_id].append(option_index)

else:
    pred_set = []
    for qa_id, pred in filtered_predictions:
        max_weights = pred['max_weight']
        choice_prob = pred['choice_prob']
        for option_index, max_weight in enumerate(max_weights):
            pred_set.append(((qa_id, option_index), (choice_prob, max_weight)))
    sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
    selected_ids = defaultdict(list)
    for (qa_id, option_index), (_, _) in sorted_predictions[:k]:
        selected_ids[qa_id].append(option_index)

total_labels = 0

initial_predictions = predictions
output = {}
for qa_id, option_list in selected_ids.items():
    output[qa_id] = initial_predictions[qa_id]
    option_num = len(output[qa_id]['raw_sentence_prob'])
    assert option_num == 4
    for idx in range(option_num):
        if idx not in option_list:
            output[qa_id]['max_weight_index'][idx] = None

            output[qa_id]['sentence_prob'][idx] = None
            output[qa_id]['sentence_label'][idx] = None
        else:
            if output[qa_id]['max_weight_index'][idx] == -1:
                raise RuntimeError(output[qa_id]['max_weight'][idx])
            # total_labels += 1

# print(total_labels)

for qa_id, pred in output.items():
    for option_index in range(4):
        if pred['max_weight_index'][option_index] is not None and pred['max_weight_index'][option_index] != -1:
            total_labels += 1
print(total_labels)

with open(f'{args.predict}-top{k}-{label_threshold}-{args.multi_evidence}', 'w') as f:
    json.dump(output, f, indent=2)

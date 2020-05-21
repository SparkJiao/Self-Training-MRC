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

# Version 2
predictions = json.load(open(args.predict, 'r'))

# We need this filter because we output all predictions, including wrong predictions and above restriction.
filtered_predictions = filter(
    lambda x: x[1]['choice_prob'] > label_threshold and x[1]['choice_prediction'] == x[1]['answer'],
    predictions.items())
if args.multi_evidence:
    pred_set = []
    for qa_id, pred in filtered_predictions:
        options_sentence_prob_sum = pred['sentence_prob_sum']
        choice_prob = pred['choice_prob']
        for option_index, sentence_prob_sum in enumerate(options_sentence_prob_sum):
            # attention weight threshold
            if pred['max_weight'][option_index] == 0:
                continue
            pred_set.append(((qa_id, option_index), (choice_prob, sentence_prob_sum)))

    sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
    selected_ids = defaultdict(list)
    for ((qa_id, option_index), (_, _)) in sorted_predictions:
        selected_ids[qa_id].append(option_index)

else:
    pred_set = []
    for qa_id, pred in filtered_predictions:
        max_weights = pred['max_weight']
        choice_prob = pred['choice_prob']
        for option_index, max_weight in enumerate(max_weights):
            # attention weight threshold
            if max_weight == 0:
                continue
            pred_set.append(((qa_id, option_index), (choice_prob, max_weight)))
    sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
    selected_ids = defaultdict(list)
    for (qa_id, option_index), (_, _) in sorted_predictions:
        selected_ids[qa_id].append(option_index)

initial_predictions = predictions
output = {}
total_labels = 0
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
            total_labels += 1
        if total_labels >= k:
            break
    if total_labels >= k:
        break

print(f'Labeled {total_labels} data in total at the first time.')

with open(f'{args.predict}-top{k}-{label_threshold}-{args.multi_evidence}', 'w') as f:
    json.dump(output, f, indent=2)

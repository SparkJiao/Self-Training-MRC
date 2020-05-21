import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True, help='Old sentence label predictions file.')
parser.add_argument('--predict2', type=str, required=True, help='New sentence label predictions file.')
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--label_threshold', type=float, required=True)
parser.add_argument('--k', type=int, default=1000, help='Select only the top `k` new predictions to merge.')

args = parser.parse_args()

label_threshold = args.label_threshold
k = args.k

predictions1 = json.load(open(args.predict1, 'r'))
predictions2 = json.load(open(args.predict2, 'r'))

filtered_predictions2 = filter(
    lambda x: x[1]['choice_prob'] > label_threshold and x[1]['choice_prediction'] == x[1]['answer'],
    predictions2.items())

pred_set = []
for qa_id, pred in filtered_predictions2:
    weights = pred['weights']
    choice_prob = pred['choice_prob']
    sentence_ids = pred['sentence_ids']
    for option_index, weight in enumerate(weights):
        # attention weight threshold
        if not sentence_ids[option_index]:
            continue
        pred_set.append(((qa_id, option_index), (choice_prob, weight)))
sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
selected_ids = defaultdict(list)
for (qa_id, option_index), (_, _) in sorted_predictions:
    selected_ids[qa_id].append(option_index)

print(f'Filtered predictions number: {len(pred_set)}')

labeled = 0
for qa_id, option_list in selected_ids.items():
    if qa_id in predictions1:
        for option_index in option_list:
            if not predictions1[qa_id]['sentence_ids'][option_index]:
                predictions1[qa_id]['sentence_ids'][option_index] = predictions2[qa_id]['sentence_ids'][option_index]
                labeled += 1
            if labeled >= k:
                break
    else:
        predictions1[qa_id] = predictions2[qa_id]
        option_num = len(predictions2[qa_id]['weights'])
        for idx in range(option_num):
            if idx not in option_list:
                predictions1[qa_id]['sentence_ids'][idx] = []
            else:
                labeled += 1
            if labeled >= k:
                break
    if labeled >= k:
        break

print(f'Add new labels {labeled}')
with open(args.output_file, 'w') as f:
    json.dump(predictions1, f, indent=2)

import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--predict', type=str, required=True)
parser.add_argument('--k', type=int, default=2000)
parser.add_argument('--label_threshold', type=float, required=True)

args = parser.parse_args()
k = args.k
label_threshold = args.label_threshold  # This parameter is the same as that of predict_sentence_ids method.

predictions = json.load(open(args.predict, 'r'))

# We need this filter because we output all predictions, including wrong predictions and above restriction.
filtered_predictions = filter(lambda x: x[1]['choice_prob'] > label_threshold and x[1]['choice_prediction'] == x[1]['answer'],
                              predictions.items())
pred_set = []
for qa_id, pred in filtered_predictions:
    choice_prob = pred['choice_prob']
    weights = pred['weights']
    sentence_ids = pred['sentence_ids']
    for option_index, weight in enumerate(weights):
        if not sentence_ids[option_index]:
            continue
        pred_set.append(((qa_id, option_index), (choice_prob, weight)))

sorted_predictions = sorted(pred_set, key=lambda x: x[1][0] * x[1][1], reverse=True)
selected_ids = defaultdict(list)
for ((qa_id, option_index), (_, _)) in sorted_predictions:
    selected_ids[qa_id].append(option_index)

initial_predictions = predictions
output = {}
total_labels = 0
for qa_id, option_list in selected_ids.items():
    output[qa_id] = initial_predictions[qa_id]
    option_num = len(output[qa_id]['weights'])
    # assert option_num == 4
    for idx in range(option_num):
        if idx not in option_list:
            output[qa_id]["sentence_ids"][idx] = []
        else:
            # if output[qa_id]["weights"][idx] <= 0:
            #     raise RuntimeError
            if not output[qa_id]["sentence_ids"][idx]:
                raise RuntimeError
            total_labels += 1
        if total_labels >= k:
            break
    if total_labels >= k:
        break

print(f'Labeled {total_labels} data in total at the first time.')

with open(f'{args.predict}-top{k}-{label_threshold}', 'w') as f:
    json.dump(output, f, indent=2)

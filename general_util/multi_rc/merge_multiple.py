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
    (lambda x: x[1]['choice_prob'] > label_threshold and x[1]['choice_prediction'] == x[1]['answer'] and x[1]['sentence_ids']),
    predictions2.items())

sorted_predictions2 = sorted(filtered_predictions2, key=lambda x: x[1]['choice_prob'] * x[1]['weight'], reverse=True)

labeled = 0
for qas_id, pred in sorted_predictions2:
    if qas_id in predictions1:
        continue
    else:
        assert pred['sentence_ids'] != []
        predictions1[qas_id] = pred
        labeled += 1
    if labeled >= k:
        break

print(f'Add new labels {labeled}')
with open(args.output_file, 'w') as f:
    json.dump(predictions1, f, indent=2)

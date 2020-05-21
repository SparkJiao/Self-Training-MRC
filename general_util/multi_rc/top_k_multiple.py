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
filtered_predictions = filter(lambda x: (x[1]['choice_prob'] > label_threshold and x[1]['choice_prediction'] == x[1]['answer'] and
                                         x[1]['sentence_ids']), predictions.items())

sorted_predictions = sorted(filtered_predictions, key=lambda x: x[1]['choice_prob'] * x[1]['weight'], reverse=True)[:k]
output = {x[0]: x[1] for x in sorted_predictions}


print(f'Labeled {len(output)} data in total at the first time.')

with open(f'{args.predict}-top{k}-{label_threshold}', 'w') as f:
    json.dump(output, f, indent=2)

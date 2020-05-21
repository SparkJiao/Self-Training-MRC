import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--predict', type=str, required=True)
parser.add_argument('--label_threshold', type=float, required=True)
parser.add_argument('--weight_threshold', type=float, required=True)
parser.add_argument('--k', type=int, required=True)

args = parser.parse_args()

prediction = json.load(open(args.predict, 'r'))
filter_predictions = filter(
    lambda x: x[1]['prediction_prob'] > args.label_threshold and x[1]['weight'] > args.weight_threshold and (
        x[1]['sentence_id'] and x[1]['sentence_id'] != -1), prediction.items())
sorted_prediction = sorted(filter_predictions, key=lambda x: x[1]['weight'] * x[1]['prediction_prob'], reverse=True)
k = min(args.k, len(sorted_prediction))
new_prediction = {x[0]: x[1] for x in sorted_prediction[:k]}

print(f"Select {len(new_prediction)} labels")

with open('%s_top_%d' % (args.predict, args.k), 'w') as f:
    json.dump(new_prediction, f)

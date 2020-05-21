import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True)
parser.add_argument('--predict2', type=str, required=True)
parser.add_argument('--label_threshold', type=float, required=True)
parser.add_argument('--weight_threshold', type=float, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--k', type=int, required=True)

args = parser.parse_args()
k = args.k

prediction1 = json.load(open(args.predict1, 'r'))
prediction2 = json.load(open(args.predict2, 'r'))

prediction2_filtered = filter(
    lambda x: x[1]['prediction_prob'] > args.label_threshold and x[1]['weight'] > args.weight_threshold and (
        x[1]['sentence_id'] and x[1]['sentence_id'] != -1),
    prediction2.items())
prediction2 = sorted(prediction2_filtered, key=lambda x: x[1]['weight'] * x[1]['prediction_prob'], reverse=True)

count = 0
for key, value2 in prediction2:
    if key in prediction1:
        continue
    prediction1[key] = value2
    # if value2['prediction_prob'] > 0.9:
    #     prediction1[key] = value2
    count += 1
    if count >= k:
        break

with open(args.output_dir, 'w') as f:
    json.dump(prediction1, f)

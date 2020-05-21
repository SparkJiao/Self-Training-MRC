import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--predict', type=str, required=True)
parser.add_argument('--k', type=int, required=True)

args = parser.parse_args()

prediction = json.load(open(args.predict, 'r'))
sorted_prediction = sorted(prediction.items(), key=lambda x: x[1]['weight'] * x[1]['prediction_prob'], reverse=True)
new_prediction = {x[0]:x[1] for x in sorted_prediction[:args.k] if x[1]['prediction_prob'] > 0.9}

with open('%s_top_%d' % (args.predict, args.k), 'w') as f:
    json.dump(new_prediction, f)


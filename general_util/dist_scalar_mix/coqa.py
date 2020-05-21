import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--prediction1', type=str, required=True)
parser.add_argument('--prediction2', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--w', type=float, default=0.5, help='Scalar weight for distributions of first predictions')
args = parser.parse_args()

predictions1 = json.load(open(args.prediction1, 'r'))
predictions2 = json.load(open(args.prediction2, 'r'))

output = []
correct = 0
total = 0
for v1, v2 in zip(predictions1, predictions2):
    total += 1
    assert v1['id'] == v2['id']
    assert v1['turn_id'] == v2['turn_id']

    raw_dist_1 = v1['raw_choice_logits']
    raw_dist_2 = v2['raw_choice_logits']

    mix_dist = list(map(lambda x, y: x * args.w + y * (1 - args.w), raw_dist_1, raw_dist_2))
    final_choice = -1
    if mix_dist[1] > mix_dist[2]:
        final_choice = 0
    elif mix_dist[2] > mix_dist[1]:
        final_choice = 1
    else:
        final_choice = 0
    output.append({
        'mix_logits': mix_dist,
        'mix_answer': final_choice,
        'pred1': v1,
        'pred2': v2
    })
    assert v1['gold_answer'] == v2['gold_answer']
    if final_choice == v1['gold_answer']:
        correct += 1

acc = correct * 1.0 / total
print(f"Accuracy: {correct} / {total} = {acc}")

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'accuracy.json'), 'w') as f:
    json.dump({'accuracy': acc}, f, indent=2)
with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
    json.dump(output, f, indent=2)

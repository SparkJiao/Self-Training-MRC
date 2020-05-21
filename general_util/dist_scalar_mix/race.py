import argparse
import os
import json
import numpy as np
import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--prediction1', type=str, required=True)
parser.add_argument('--prediction2', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--w', type=float, default=0.5, help='Scalar weight for distributions of first predictions')
args = parser.parse_args()

predictions1 = json.load(open(args.prediction1, 'r'))
predictions2 = json.load(open(args.prediction2, 'r'))

output = {}
total = 0
correct = 0
for k1, v1 in predictions1.items():
    total += 1
    assert k1 in predictions2
    v2 = predictions2[k1]
    assert v1['answer'] == v2['answer']
    dist1 = v1['raw_choice_logits']
    dist2 = v2['raw_choice_logits']

    mixed_dist = list(map(lambda x, y: x * args.w + y * (1 - args.w), dist1, dist2))
    # mixed_prediction = np.array(mixed_dist).argmax().tolist()
    mixed_prediction = F.softmax(torch.Tensor(mixed_dist), dim=-1).max(dim=-1)[1].item()

    output[k1] = {
        'pred1': v1,
        'pred2': v2,
        'mixed_choice_logits': mixed_dist,
        'mixed_prediction': mixed_prediction
    }

    if mixed_prediction == v1['answer']:
        correct += 1

acc = correct * 1.0 / total
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'accuracy.json'), 'w') as f:
    json.dump({'accuracy': acc}, f, indent=2)
with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
    json.dump(output, f, indent=2)

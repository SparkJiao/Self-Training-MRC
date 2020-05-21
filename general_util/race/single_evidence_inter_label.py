import json
import argparse
import copy
import os

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True)
parser.add_argument('--predict2', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

predictions1 = json.load(open(args.predict1, 'r'))
predictions2 = json.load(open(args.predict2, 'r'))

inter_predictions = {}
total_labels = 0
for key, value1 in predictions1.items():
    if key not in predictions2:
        continue
    if value1['choice_prediction'] != predictions2[key]['choice_prediction']:
        continue
    value2 = predictions2[key]
    option_num = len(value1['raw_sentence_prob'])
    inter_value = copy.deepcopy(value1)
    cnt = 0
    for option_index in range(option_num):
        if value1['max_weight_index'][option_index] == value2['max_weight_index'][option_index]:
            cnt += 1
        else:
            inter_value['max_weight_index'][option_index] = None
            inter_value['sentence_prob'][option_index] = None
            inter_value['sentence_label'][option_index] = None
    if cnt > 0:
        inter_predictions[key] = inter_value
    total_labels += cnt

print(f'After inter-merge, the total number of labels is {total_labels}')
predict1_name = args.predict1.split('/')[-1]
predict2_name = args.predict2.split('/')[-1]
with open(os.path.join(args.output_dir, f'{predict1_name}-{predict2_name}-inter-merge.json'), 'w') as f:
    json.dump(inter_predictions, f, indent=2)

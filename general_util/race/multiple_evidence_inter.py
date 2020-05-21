import json
import argparse
import copy
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True)
parser.add_argument('--predict2', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--drop_greater', default=False, action='store_true')

args = parser.parse_args()

predictions1 = json.load(open(args.predict1, 'r'))
predictions2 = json.load(open(args.predict2, 'r'))

inter_predictions = {}
# total_labels = 0
all_the_same = 0
greater = 0
for key, value1 in predictions1.items():
    if key not in predictions2:
        continue
    if value1['choice_prediction'] != predictions2[key]['choice_prediction']:
        continue
    value2 = predictions2[key]
    option_num = len(value1['sentence_ids'])
    inter_value = copy.deepcopy(value1)
    for option_index in range(option_num):
        if value1['sentence_ids'][option_index] == [] or value2['sentence_ids'][option_index] == []:
            # FIXME:
            #  2019.9.6
            #  Delete the label in inter_value
            # FIXME:
            #  2019.9.19
            #  value2['sentence_ids'] == [] -> value2['sentence_ids][option_index] == []
            inter_value['sentence_ids'][option_index] = []
            continue
        if value1['sentence_ids'][option_index] == value2['sentence_ids'][option_index]:
            all_the_same += 1
        else:
            if value2['weights'][option_index] > value1['weights'][option_index]:
                inter_value['sentence_ids'][option_index] = value2['sentence_ids'][option_index]
                inter_value['weights'][option_index] = value2['weights'][option_index]
                inter_value['raw_sentences'][option_index] = value2['raw_sentences'][option_index]
            greater += 1
            if args.drop_greater:
                inter_value['sentence_ids'][option_index] = []
        # if value1['max_weight_index'][option_index] == value2['max_weight_index'][option_index]:
        #     cnt += 1
        # else:
        #     inter_value['max_weight_index'][option_index] = None
        #     inter_value['sentence_prob'][option_index] = None
        #     inter_value['sentence_label'][option_index] = None
    inter_predictions[key] = inter_value

final_label = 0
for value in inter_predictions.values():
    for sentence_ids in value['sentence_ids']:
        if sentence_ids:
            final_label += 1

# print(f'After inter-merge, the total number of labels is {total_labels}')
print(f'After inter-inter-merge, number of all the same labels is {all_the_same} and number of greater labels is {greater}')
print(f'The number of final labels is {final_label}')

recurrent_pattern = re.compile(r'recurrent\d+')
view_pattern = re.compile(r'view\d+')
# predict1_name = args.predict1[re.search(args.predict1, args.predict1)] + '_' + args.predict1[re.search(view_pattern, args.predict1)]
predict1_name = recurrent_pattern.findall(args.predict1)[-1] + '_' + view_pattern.findall(args.predict1)[-1]
predict2_name = recurrent_pattern.findall(args.predict2)[-1] + '_' + view_pattern.findall(args.predict2)[-1]

with open(os.path.join(args.output_dir,
                       f'{predict1_name}-{predict2_name}-{args.drop_greater}-inter-merge-{all_the_same}-{greater}-{final_label}.json'),
          'w') as f:
    json.dump(inter_predictions, f, indent=2)

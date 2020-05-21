import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True)
parser.add_argument('--predict2', type=str, required=True)
parser.add_argument('--predict3', type=str, required=True)

args = parser.parse_args()

prediction1 = json.load(open(args.predict1, 'r'))
prediction2 = json.load(open(args.predict2, 'r'))
prediction3 = json.load(open(args.predict3, 'r'))
label = json.load(open('../BERT/max_f1/coqa-dev-v1.0.json', 'r'))['data']

total = 0
right = 0
item = label[0]
for item1, item2, item3 in zip(prediction1, prediction2, prediction3):
    if item1['id'] != item2['id'] or item1['id'] != item3['id']:
        print('id mismatched!')
        break
    if item1['turn_id'] != item2['turn_id'] or item1['turn_id'] != item3['turn_id']:
        print('turn_id mismatched!')
        break
    yes_num = 0
    if item1['answer'] == 'yes':
        yes_num += 1
    if item2['answer'] == 'yes':
        yes_num += 1
    if item3['answer'] == 'yes':
        yes_num += 1
    answer = 'no'
    if yes_num > 1:
        answer = 'yes'
    pid = item1['id']
    while item['id'] != pid:
        item = label[0]
        label = label[1:]
    total += 1
    if answer in item['answers'][item1['turn_id'] - 1]['input_text'].lower():
        right += 1
print('right: %d, total: %d, accuracy: %f' % (right, total, float(right)/ total))


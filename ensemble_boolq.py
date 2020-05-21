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
r_label = open('./max_f1/dev.jsonl', 'r')

total = 0
right = 0
for pid, (line, item1, item2, item3) in enumerate(zip(r_label, prediction1, prediction2, prediction3)):
    item = json.loads(line)
    yes_num = 0
    if item1['answer'] == 'yes':
        yes_num += 1
    if item2['answer'] == 'yes':
        yes_num += 1
    if item3['answer'] == 'yes':
        yes_num += 1
    answer = False
    if yes_num > 1:
        answer = True
    total += 1
    if answer == item['answer']:
        right += 1
print('right: %d, total: %d, accuracy: %f' % (right, total, float(right)/ total))


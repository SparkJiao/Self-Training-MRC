import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True)
parser.add_argument('--predict2', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

prediction1 = json.load(open(args.predict1, 'r'))
prediction2 = json.load(open(args.predict2, 'r'))

new_prediction = {}
for key, value1 in prediction2.items():
    if key not in prediction1:
        new_prediction[key] = {'sentence_id': value1['sentence_id'],
                               'doc_span_index': value1['doc_span_index'],
                               'weight': value1['weight'],
                               'choice_prediction': value1['choice_prediction'],
                               'prediction_prob': value1['prediction_prob']}

for key, value1 in prediction1.items():
    if key not in prediction2:
        new_prediction[key] = {'sentence_id': value1['sentence_id'],
                               'doc_span_index': value1['doc_span_index'],
                               'weight': value1['weight'],
                               'choice_prediction': value1['choice_prediction'],
                               'prediction_prob': value1['prediction_prob']}
        continue
    value2 = prediction2[key]
    if value1['sentence_id'] != value2['sentence_id']:
        continue
    if value1['doc_span_index'] != value2['doc_span_index']:
        continue
    if value1['choice_prediction'] != value2['choice_prediction']:
        continue
    weight = (value1['weight'] + value2['weight']) * 0.5
    prob = (value1['prediction_prob'] + value2['prediction_prob']) * 0.5
    new_prediction[key] = {'sentence_id': value1['sentence_id'],
                           'doc_span_index': value1['doc_span_index'],
                           'weight': weight,
                           'choice_prediction': value1['choice_prediction'],
                           'prediction_prob': prob}

print('evidence1 num: %d' % (len(prediction1)))
print('evidence2 num: %d' % (len(prediction2)))
print('combine evidence num: %d' % (len(new_prediction)))
with open(os.path.join(args.output_dir, 'union_label.json'), 'w') as f:
    json.dump(new_prediction, f)

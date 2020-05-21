import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--predict1', type=str, required=True)
parser.add_argument('--predict2', type=str, required=True)

args = parser.parse_args()

prediction1 = json.load(open(args.predict1, 'r'))
prediction2 = json.load(open(args.predict2, 'r'))

new_prediction = {}
for key, value1 in prediction1.items():
    if key not in prediction2:
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

with open('./combine.json', 'w') as f:
    json.dump(new_prediction, f)


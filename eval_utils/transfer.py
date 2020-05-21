import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')

args = parser.parse_args()

with open(args.input_file, 'r') as f:
    data = json.load(f)

output = []
for prediction in data:
    keys = prediction.split('--')
    pred = dict()
    pred['id'] = keys[0]
    pred['turn_id'] = int(keys[1])
    pred['answer'] = data[prediction]
    output.append(pred)

with open(args.output_file, 'w', encoding='utf-8') as file:
    # print(args.output_file)
    file.write(json.dumps(output, indent=4))

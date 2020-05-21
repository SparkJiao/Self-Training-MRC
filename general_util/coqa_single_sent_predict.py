import json
import os
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)

args = parser.parse_args()

iter_times = 10

for i in range(iter_times):

    tar_dir = os.path.join(args.input_dir, f'recurrent{i}/single_sent')
    prediction = json.load(open(os.path.join(tar_dir, 'predictions.json'), 'r'))

    output_dir = defaultdict(dict)

    for pred in prediction:
        story_id, sent_id = pred['id'].split("##")
        sent_id = int(sent_id[1:])
        turn_id = pred['turn_id']
        gold_answer_prob = pred['raw_choice_logits'][pred['gold_answer'] + 1]

        pred['gold_prob'] = gold_answer_prob
        pred['sent_id'] = sent_id

        if turn_id not in output_dir[story_id]:
            output_dir[story_id][turn_id] = []

        output_dir[story_id][turn_id].append(pred)

    evidence_dict = defaultdict(dict)
    for story_id in output_dir:
        for turn_id in output_dir[story_id]:

            sorted_seq = sorted(output_dir[story_id][turn_id], key=lambda x: x['gold_prob'], reverse=True)

            for pred in output_dir[story_id][turn_id]:
                assert pred['gold_answer'] == sorted_seq[0]['gold_answer']

            evidence_dict[story_id][turn_id] = sorted_seq[0]['sent_id']

    with open(os.path.join(tar_dir, 'evidence_id.json'), 'w') as f:
        json.dump(evidence_dict, f, indent=2)

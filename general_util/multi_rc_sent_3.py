import json
import os
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)

args = parser.parse_args()

iter_times = 10

for i in range(iter_times):

    tar_dir = os.path.join(args.input_dir, f'recurrent{i}/sent3_predict')
    prediction = json.load(open(os.path.join(tar_dir, 'predictions.json'), 'r'))

    output_dir = defaultdict(list)

    for pred_id, pred in prediction.items():
        passage_id, q_id, op_id = pred_id.split("--")
        tuples = passage_id.split("#")
        passage_id = tuples[0]
        sent_ids = [int(x) for x in tuples[1:]]
        assert len(sent_ids) == 3, len(sent_ids)

        """
        We search the evidence sentences for each question by the prob of gold answer instead of searching evidences for each option.
        """
        new_id = f'{passage_id}--{q_id}'

        if pred['prediction'] == pred['answer']:
            gold_answer_prob = pred['pred_prob']
        else:
            gold_answer_prob = 1 - pred['pred_prob']

        pred['gold_prob']  = gold_answer_prob
        pred['sent_id'] = sent_ids

        output_dir[new_id].append(pred)

    evidence_dict = dict()
    for passage_id in output_dir:
        sorted_seq = sorted(output_dir[passage_id], key=lambda x: x['gold_prob'], reverse=True)

        # for pred in output_dir[passage_id]:
        #     assert pred['answer'] == sorted_seq[0]['answer']

        evidence_dict[passage_id] = sorted_seq[0]['sent_id']

    with open(os.path.join(tar_dir, 'evidence_id.json'), 'w') as f:
        json.dump(evidence_dict, f, indent=2)

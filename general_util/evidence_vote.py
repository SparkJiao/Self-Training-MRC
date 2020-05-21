import argparse
import json
import pickle
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file1', type=str, required=True)
parser.add_argument('--input_file2', type=str, required=True)
parser.add_argument('--old_file', type=str, default=None)
parser.add_argument('--output_file', type=str, required=True)

args = parser.parse_args()
opt = vars(args)

with open(opt['input_file1'], 'r') as f:
    evidence_pred1 = json.load(f)
with open(opt['input_file2'], 'r') as f:
    evidence_pred2 = json.load(f)

logger.info(f'Labels from {opt["input_file1"]} is {len(evidence_pred1)}')
logger.info(f'Labels from {opt["input_file2"]} is {len(evidence_pred2)}')

output_dict = dict()
for qas_id in evidence_pred1:
    if qas_id in evidence_pred2:
        pred1 = evidence_pred1[qas_id]
        pred2 = evidence_pred2[qas_id]
        if pred1['sentence_id'] == pred2['sentence_id']:
            # Some times following things will happen:
            # The evidence is contained in both segments so probably the doc_span_index and doc_span_sentence_id may be different.
            # We could add them both.
            # assert pred1['doc_span_index'] == pred2['doc_span_index'], (pred1, pred2)
            # assert pred1['doc_span_sentence_id'] == pred2['doc_span_sentence_id'], (pred1, pred2)
            # output_dict[qas_id] = {
            #     'sentence_id': pred1['sentence_id'],
            #     'doc_span_indexes': [pred1['doc_span_index']],
            #     'doc_span_sentence_ids': [pred1['doc_span_sentence_id']]
            # }
            # if pred1['doc_span_index'] == pred2['doc_span_index']:
            #     assert pred1['doc_span_sentence_id'] == pred2['doc_span_sentence_id']
            # else:
            #     output_dict[qas_id]['doc_span_indexes'].append(pred2['doc_span_index'])
            #     output_dict[qas_id]['doc_span_sentence_ids'].append(pred2['doc_span_sentence_id'])
            key1 = (qas_id, pred1['doc_span_index'])
            output_dict[key1] = {
                'sentence_id': pred1['sentence_id'],
                'doc_span_sentence_id': pred1['doc_span_sentence_id']
            }
            if pred1['doc_span_index'] == pred2['doc_span_index']:
                assert pred1['doc_span_sentence_id'] == pred2['doc_span_sentence_id']
            else:
                key2 = (qas_id, pred2['doc_span_index'])
                output_dict[key2] = {
                    'sentence_id': pred2['sentence_id'],
                    'doc_span_sentence_id': pred2['doc_span_sentence_id']
                }

"""
There are few strategies to label data:
    1. No replacement. Only labeled the unlabeled data after previous labeling process.
       
       Under this strategy, the threshold for attention weight and yesno prediction can be fixed.
       The process won't shut down until no 

    2. With replacement. For any data, use the label predicted more recently.
    3. Current label only. Only use the label predicted during current process.
"""

stop_flag = False

if opt['old_file'] is not None:
    # 2. No replacement.
    with open(opt['old_file'], 'rb') as f:
        old_dict = pickle.load(f)
    exist = 0
    for key in old_dict:
        if key in output_dict:
            exist += 1
        output_dict[key] = old_dict[key]
    print(f'Labeled {len(output_dict)} data in total, {len(old_dict)} data have been labeled before, {exist} data are selected again.')
    if len(output_dict) == len(old_dict):
        print('No new labeled data added into training set. Process shuts down.')
        stop_flag = True
else:
    if len(output_dict) == 0:
        print('No agreement between predictions of model1 and model2. Process shuts down.')
        stop_flag = True
    print(f'Labeled {len(output_dict)} data in total.')

if not stop_flag:
    with open(opt['output_file'], 'wb') as f:
        pickle.dump(output_dict, f)

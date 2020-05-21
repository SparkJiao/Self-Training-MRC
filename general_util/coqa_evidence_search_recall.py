import json
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)

args = parser.parse_args()

output_writer = open(os.path.join(args.input_dir, 'evidence_search_results.txt'), 'w')

def k_precise(p_ls, g_ls):
    cnt = 0
    for x in p_ls:
        if x in g_ls:
            cnt += 1
    return cnt * 1.0 / len(p_ls)

def k_recall(p_ls, g_ls):
    cnt = 0
    for x in p_ls:
        if x in g_ls:
            cnt += 1
    return cnt * 1.0 / len(g_ls)

recurrent_times = 10

for i in range(recurrent_times):

    print(f'============== recurrent {i} ================', file=output_writer)

    data_dir = os.path.join(args.input_dir, f'recurrent{i}')

    prediction_file = os.path.join(data_dir, f'single_sent/evidence_id.json')
    predictions = json.load(open(prediction_file, 'r'))

    data_file = os.path.join(data_dir, f'sentence_predictions.json')
    data_dict = json.load(open(data_file, 'r'))

    recall_ls = []
    precise_ls = []

    for data_id in data_dict:
        story_id, q_id = data_id.split('--')
        pred_ls = [predictions[story_id][q_id]]
        gold_ls = [data_dict[data_id]['gold_sentence_id']]  # don't ignore -1
        recall_ls.append(k_recall(pred_ls, gold_ls))
        precise_ls.append(k_precise(pred_ls, gold_ls))

    # recall = sum(recall_ls) * 1.0 / len(recall_ls)
    precise = sum(precise_ls) * 1.0 / len(precise_ls)

    # print(f'Recall @ 1: {recall}', file=output_writer)
    print(f'Precise @ 1: {precise}', file=output_writer)

    print('', file=output_writer)
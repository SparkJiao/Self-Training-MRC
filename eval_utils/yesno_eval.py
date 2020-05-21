import json
from general_util.utils import normalize_answer
from collections import Counter


def choose_common_answer(answer_list):
    counter = Counter()

    for answer in answer_list:
        counter[normalize_answer(answer)] += 1

    most_two = counter.most_common(2)
    if most_two[0][1] == most_two[1][1] and most_two[0][1] in ['yes', 'no'] and most_two[1][1] in ['yes', 'no']:
        return 'unknown'
    else:
        return most_two[0][1]


class YesNoEvaluator(object):
    def __init__(self, gold_file, predictions):
        with open(gold_file, 'r') as f:
            data = json.load(f)['data']

        self.golds = dict()
        for article in data:
            answers = [answer['input_text'] for answer in article['answers']]
            answers += [answer['input_text'] for answer in article['additional_answers'].values()]


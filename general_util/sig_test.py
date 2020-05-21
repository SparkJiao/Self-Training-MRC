import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, required=True)
parser.add_argument('--our', type=str, required=True)

args = parser.parse_args()


def coqa_marco_get_accuracy(prediction_file):
    """ coqa/marco """
    all_predictions = json.load(open(prediction_file, 'r'))

    correct = 0
    total = 0

    for prediction in all_predictions:
        if prediction['gold_answer'] == 0:
            gold = 'yes'
        elif prediction['gold_answer'] == 1:
            gold = 'no'
        else:
            raise RuntimeError(f'Wrong gold answer type: {prediction["gold_answer"]}')

        total += 1
        if gold == prediction['answer']:
            correct += 1

    return correct * 1.0 / total


def race_get_accuracy(prediction_file):
    """ race/dream """
    all_predictions = json.load(open(prediction_file, 'r'))

    correct = 0
    total = 0
    for prediction in all_predictions.values():
        total += 1
        if prediction['answer'] == prediction['prediction']:
            correct += 1

    return correct * 1.0 / total

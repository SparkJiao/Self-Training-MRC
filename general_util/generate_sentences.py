import argparse
import json

import nltk

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def coqa(args):
    opt = vars(args)
    with open(opt['input_file'], 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    output = dict()
    for article in data:
        story = article['story']
        story_id = article['id']
        output[story_id] = sentence_tokenizer.tokenize(story)
    with open(opt['output_file'], 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def squad(args):
    opt = vars(args)
    with open(opt['input_file'], 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    paragraphs = []
    qas = dict()
    for entry in data:
        for paragraph in entry['paragraphs']:
            paragraphs.append(sentence_tokenizer.tokenize(paragraph['context']))

            for qa in paragraph['qas']:
                qas_id = qa['id']
                qas[qas_id] = {'context_index': len(paragraphs) - 1}
    with open(opt['output_file'], 'w') as f:
        json.dump({'contexts': paragraphs, 'qas': qas}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Load dataset and get the list of sentences of single passage')
    parser.add_argument('input_file')
    parser.add_argument('output_file')

    sub_parser = parser.add_subparsers(title='tasks', description='Name of different dataset.',
                                       help='coqa, squad, quac, marco')

    coqa_sen = sub_parser.add_parser('coqa')
    coqa_sen.set_defaults(func=coqa)

    squad_sen = sub_parser.add_parser('squad')
    squad_sen.set_defaults(func=squad)

    args = parser.parse_args()
    args.func(args)

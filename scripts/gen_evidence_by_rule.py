#!/usr/bin/env python
# coding: utf-8

import json
import nltk
import jsonlines
from tqdm import tqdm
import math

import argparse
import itertools

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class Jaccard(object):
    """docstring for Jaccard"""

    def __init__(self):
        super(Jaccard, self).__init__()
        pass

    def get_sim(self, str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / len(b) - len(a) * 1e-9


class IDF(object):
    """docstring for IDF"""

    def __init__(self):
        super(IDF, self).__init__()
        pass

    def get_sim(self, str1, str2, idf):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)

        b = sum([idf[w] if w in idf else max(idf.values()) for w in b])
        c = sum([idf[w] for w in c])
        return (c + 1e-15) / (b + 1e-15) - len(a) * 1e-9

    def get_idf(self, sentences):
        idf = {}
        for sentence in sentences:
            sentence = set(sentence.lower().split())
            for word in sentence:
                if word not in idf:
                    idf[word] = 0
                idf[word] += 1
        num_sentence = len(sentences)
        for word, count in idf.items():
            idf[word] = math.log((num_sentence + 1) / float(count + 1))
        return idf


class ILP(object):
    """docstring for ILP"""

    def __init__(self):
        super(ILP, self).__init__()
        pass

    def get_sim(self, sentences, word_weights, max_k=1):
        sentences = [set(sentence.lower().split()) for sentence in sentences]
        max_set, max_value = [-1], -1
        for sentence_ids in itertools.combinations(range(len(sentences)), max_k):
            words = sentences[sentence_ids[0]]
            for sentence_id in sentence_ids[1:]:
                words = words + sentences[sentence_id]
            words = set(words)
            value = sum([word_weights[word] if word in word_weights else 0. for word in words]) / sum(word_weights.values())
            if value > max_value:
                max_set = sentence_ids
                max_value = value
        return max_set, max_value

    def get_word_weights0(self, query):
        query = set(query.lower().split())
        word_weights = {word: 1. for word in query}
        return word_weights

    def get_word_weights1(self, query, historys):
        query = set(query.lower().split())
        word_weights = {word: 1. for word in query}
        historys = [set(history.lower().split()) for history in historys]
        for history in historys:
            for word in history:
                if word in word_weights:
                    continue
                word_weights[word] = 0.1
        return word_weights


class DatasetBase(object):
    """docstring for DatasetBase"""

    def __init__(self, mode: str = 'IDF'):
        super(DatasetBase, self).__init__()
        '''
        mode: `Jaccard/IDF/ILP`
        '''
        self.evidence = []
        self.mode = mode
        if self.mode == 'Jaccard':
            self.rule = Jaccard()
        elif self.mode == 'IDF':
            self.rule = IDF()
        elif self.mode == 'ILP':
            self.rule = ILP()
        else:
            raise ValueError('`mode`[input:%s] should be one of `Jaccard/IDF/ILP`' % (self.mode))

    def sort(self, top_k):
        threshold = sorted(self.evidence, key=lambda x: x[1], reverse=True)[top_k - 1][1]
        for _evidence in self.evidence:
            if _evidence[1] < threshold:
                _evidence[0] = -1
        return self.evidence

    def get_sim(self, *args):
        if self.mode == 'Jaccard':
            max_ids, max_value = self.jaccard(*args)
        elif self.mode == 'IDF':
            max_ids, max_value = self.idf(*args)
        elif self.mode == 'ILP':
            max_ids, max_value = self.ilp(*args)
        self.evidence.append([max_ids, max_value])

    def jaccard(self, sentences, query, top_k=1):
        values = []
        for sentence in sentences:
            values.append(self.rule.get_sim(sentence, query))
        values = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:top_k]
        max_ids = [x[0] for x in values]
        max_value = sum([x[1] for x in values])
        return max_ids, max_value

    def idf(self, sentences, query, top_k=1):
        idf = self.rule.get_idf(sentences)
        values = []
        for sentence in sentences:
            values.append(self.rule.get_sim(sentence, query, idf))
        values = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:top_k]
        max_ids = [x[0] for x in values]
        max_value = sum([x[1] for x in values])
        return max_ids, max_value

    def ilp(self, sentences, query, top_k=1):
        word_weights = self.rule.get_word_weights0(query)
        max_set, max_value = self.rule.get_sim(sentences, word_weights, top_k)
        return max_set, max_value

    def save(self, data, output_file):
        with open(output_file, 'w') as w:
            json_data = json.dumps(data, sort_keys=False, indent=4, separators=(',', ': '))
            w.write(json_data)


class CoQA(DatasetBase):
    """docstring for CoQA"""

    def __init__(self, mode):
        super(CoQA, self).__init__(mode)
        pass

    def evaluate(self, golden_evidence):
        right, total = 0., 0.
        for _evidence, _golden_evidence in zip(self.evidence, golden_evidence):
            if _evidence[0] == -1:
                continue
            if _evidence[0][0] == _golden_evidence:
                right += 1.
            total += 1.
        print('CoQA: label %d, right %d, acc %f' % (total, right, right / total))

    def find_gold_evidence(self, sentences, span_start):
        start, end = 0, 0
        for pid, sentence in enumerate(sentences):
            end += len(sentence)
            if span_start >= start and span_start < end:
                return pid
            start = end

    def find_gold_evidence2(self, paragraph, question, answer):
        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        paragraph_text = paragraph["story"]
        story_id = paragraph['id']
        doc_tokens = []
        prev_is_whitespace = True
        char_to_word_offset = []
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        # Split context into sentences
        sentence_start_list, sentence_end_list = utils.split_sentence(paragraph_text, sentence_tokenizer)
        sentence_span_list = []
        for c_start, c_end in zip(sentence_start_list, sentence_end_list):
            t_start = char_to_word_offset[c_start]
            t_end = char_to_word_offset[c_end]
            sentence_span_list.append((t_start, t_end))

        question_text = question['input_text']

        # Add rationale start and end as extra supervised label.
        rationale_start_position = char_to_word_offset[answer['span_start']]
        rationale_end_position = char_to_word_offset[answer['span_end'] - 1]

        sentence_id = utils.find_evidence_sentence(sentence_span_list, rationale_start_position, rationale_end_position)

        return sentence_id

    def process_file(self, input_file, mode: str = 'q', top_k: int = 1000):
        with open(input_file, 'r') as f:
            data = json.load(f)['data']

        golden_evidence = []

        for instance in tqdm(data):
            article = instance['story']
            article_sentences = sentence_tokenizer.tokenize(article)

            questions = instance['questions']
            answers = instance['answers']

            for turn_id, (que, ans) in enumerate(zip(questions, answers)):
                que_text = que['input_text']

                if mode == 'q':
                    pass
                elif mode == 'aq':
                    if turn_id > 0:
                        que_text = answers[turn_id - 1]['input_text'] + '<A>' + que_text
                elif mode == 'qaq':
                    if turn_id > 0:
                        que_text = questions[turn_id - 1]['input_text'] + '<Q>' + answers[turn_id - 1]['input_text'] + '<A>' + que_text
                else:
                    raise RuntimeError(f'No compatible mode for {mode}')

                self.get_sim(article_sentences, que_text, 1)
                golden_evidence.append(self.find_gold_evidence2(instance, que, ans))

        self.sort(top_k)
        self.evaluate(golden_evidence)

        for instance in tqdm(data):
            questions = instance['questions']
            answers = instance['answers']

            for turn_id, (que, ans) in enumerate(zip(questions, answers)):
                questions[turn_id]['sentence_id'] = self.evidence[0][0]
                self.evidence = self.evidence[1:]

        assert 'sentence_id' in data[0]['questions'][0]
        return {
            'data': data
        }


class BoolQ(DatasetBase):
    """docstring for BoolQ"""

    def __init__(self, mode):
        super(BoolQ, self).__init__(mode)
        pass

    def process_file(self, input_file, top_k: int = 1000):
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                data.append(line)

        for article in tqdm(data):
            question = article['question']
            passage = article['passage']

            sentences = sentence_tokenizer.tokenize(passage)

            self.get_sim(sentences, question, 1)

        self.sort(top_k)

        for article in tqdm(data):
            article['sentence_id'] = self.evidence[0][0]
            self.evidence = self.evidence[1:]

        assert 'sentence_id' in data[0]
        return data

    def save(self, data, output_file):
        with open(output_file, 'w') as w:
            for item in data:
                w.write(json.dumps(item) + '\n')


class RACE(DatasetBase):
    """docstring for CoQA"""

    def __init__(self, mode):
        super(RACE, self).__init__(mode)
        pass

    def process_file(self, input_file, num_evidences=2, top_k: int = 1000):
        with open(input_file, 'r') as f:
            data = json.load(f)

        # examples = []
        for instance in tqdm(data):
            passage = instance['article']
            article_id = instance['id']

            article_sentences = sentence_tokenizer.tokenize(passage)

            questions = instance['questions']
            answers = list(map(lambda x: {'A': 0, 'B': 1, 'C': 2, 'D': 3}[x], instance['answers']))
            options = instance['options']

            for q_id, (question, answer, option_list) in enumerate(zip(questions, answers, options)):
                # qas_id = f"{article_id}--{q_id}"

                for option in option_list:
                    self.get_sim(article_sentences, question + ' ' + option, num_evidences)

        self.sort(top_k)
        output = {}
        for instance in tqdm(data):
            article_id = instance['id']
            questions = instance['questions']
            options = instance['options']

            for q_id, (_, option_list) in enumerate(zip(questions, options)):
                qas_id = f'{article_id}--{q_id}'

                output[qas_id] = {'sentence_ids': []}
                for op_index, op in enumerate(option_list):
                    if self.evidence[0][0] == -1:
                        output[qas_id]['sentence_ids'].append([])
                    else:
                        output[qas_id]['sentence_ids'].append(self.evidence[0][0])
                    self.evidence = self.evidence[1:]

        return output


class MultiRC(DatasetBase):
    """docstring for CoQA"""

    def __init__(self, mode):
        super(MultiRC, self).__init__(mode)
        pass

    def process_file(self, input_file, num_evidences=3, top_k: int = 1000):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # examples = []
        for instance in tqdm(data):
            article_id = instance['id']

            article_sentences = instance['article']

            questions = instance['questions']
            options = instance['options']
            # gold_evidences = instance['evidence']

            for q_id, (question, option_list) in enumerate(zip(questions, options)):
                # # qas_id = f"{article_id}--{q_id}"
                #
                # for option in option_list:
                #     self.get_sim(article_sentences, question + ' ' + option, 2)
                self.get_sim(article_sentences, question, num_evidences)

        self.sort(top_k)
        output = {}
        for instance in tqdm(data):
            article_id = instance['id']
            questions = instance['questions']
            options = instance['options']

            for q_id, (_, option_list) in enumerate(zip(questions, options)):
                pseudo_evidence = self.evidence[0][0]
                self.evidence = self.evidence[1:]
                for op_index, op in enumerate(option_list):
                    qas_id = f'{article_id}--{q_id}--{op_index}'
                    # if self.evidence[0][0] == -1:
                    #     output[qas_id]['sentence_ids'].append([])
                    # else:
                    #     output[qas_id]['sentence_ids'].append(self.evidence[0][0])
                    # self.evidence = self.evidence[1:]
                    output[qas_id] = {
                        'sentence_ids': pseudo_evidence if pseudo_evidence != -1 else []
                    }

        return output


class MARCO(DatasetBase):
    def __init__(self, mode):
        super(MARCO, self).__init__(mode)
        pass

    def process_file(self, input_file, num_evidences=2, top_k: int = 1000):
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        for articles, qas_id, question in tqdm(zip(input_data['passages'], input_data['ids'], input_data['questions'])):
            passage = ''
            for doc in articles:
                passage = passage + doc['text']
            article_sentences = sentence_tokenizer.tokenize(passage)
            self.get_sim(article_sentences, question, num_evidences)

        self.sort(top_k=top_k)
        output = {}
        for qas_id in tqdm(input_data['ids']):
            pseudo_evidence = self.evidence[0][0]
            if pseudo_evidence == -1:
                pseudo_evidence = []
            self.evidence = self.evidence[1:]
            output[qas_id] = {
                'sentence_id': pseudo_evidence,
                'doc_span_index': 0
            }

        return output


if __name__ == '__main__':

    # task_name = 'RACE'
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--num_evidences', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=100000)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    task_name = args.task_name

    if task_name == 'BoolQ':
        boolq_train = './boolq-enlarge/train.jsonl.enlarge'
        boolq_dev = './boolq-enlarge/dev.jsonl.enlarge'

        train_boolq = BoolQ(mode='IDF')
        train_data = train_boolq.process_file(input_file=boolq_train, top_k=150)
        train_boolq.save(train_data, output_file='./data-rules/boolq/idf/train-enlarge-rule.json')

        dev_boolq = BoolQ(mode='IDF')
        dev_data = dev_boolq.process_file(input_file=boolq_dev, top_k=1)
        dev_boolq.save(dev_data, output_file='./data-rules/boolq/idf/dev-enlarge-rule.json')

    if task_name == 'CoQA':
        coqa_train = './coqa/coqa-train-v1.0.json'
        coqa_dev = './coqa/coqa-dev-v1.0.json'

        train_coqa = CoQA(mode='IDF')
        train_data = train_coqa.process_file(input_file=coqa_train, mode='q', top_k=1000)
        train_coqa.save(train_data, output_file='./data-rules/coqa/idf/train-enlarge-rule.json')

        dev_coqa = CoQA(mode='IDF')
        dev_data = dev_coqa.process_file(input_file=coqa_dev, mode='q', top_k=1)
        dev_coqa.save(dev_data, output_file='./data-rules/coqa/idf/dev-enlarge-rule.json')

    if task_name == 'MARCO':
        train_marco = MARCO(mode='IDF')
        train_data = train_marco.process_file(args.input_file, top_k=args.top_k, num_evidences=args.num_evidences)
        with open(args.output_file, 'w') as f:
            json.dump(train_data, f, indent=2)

    if task_name == 'RACE':
        train_race = RACE(mode='IDF')
        train_data = train_race.process_file(args.input_file, top_k=args.top_k, num_evidences=args.num_evidences)
        with open(args.output_file, 'w') as f:
            json.dump(train_data, f, indent=2)

    if task_name == 'MultiRC':
        train_multi_rc = MultiRC(mode='IDF')
        train_data = train_multi_rc.process_file(args.input_file, top_k=args.top_k, num_evidences=args.num_evidences)
        with open(args.output_file, 'w') as f:
            json.dump(train_data, f, indent=2)

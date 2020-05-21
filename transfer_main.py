# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import random
from typing import Tuple
import math

import numpy as np
import torch
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from bert_model import initialize_model, prepare_model_params
from data.data_instance import RawResultChoice, ModelState, ReadState, FullResult
from general_util.utils import AverageMeter
from general_util.logger import setting_logger
from reader import initialize_reader, prepare_read_params


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


"""
This script has several usages:
    - Train a bert model from scratch.
    - Train a bert model from a pretrained model.
"""


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--vocab_file", default='bert-base-uncased-vocab.txt', type=str, required=True)
    parser.add_argument("--model_file", default='bert-base-uncased.tar.gz', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--predict_dir", default=None, type=str, required=True,
                        help="The output directory where the predictions will be written.")
    parser.add_argument('--predict_output_file', type=str, default='predictions.json')
    parser.add_argument('--label_output_file', type=str, default='evidence_predictions.json')

    # Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    # Base setting
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--max_ctx', type=int, default=2)
    parser.add_argument('--task_name', type=str, default='coqa_yesno')
    parser.add_argument('--bert_name', type=str, default='baseline')
    parser.add_argument('--reader_name', type=str, default='coqa')
    # model parameters
    parser.add_argument('--evidence_lambda', type=float, default=0.8)
    parser.add_argument('--negative_lambda', type=float, default=1.0)
    parser.add_argument('--cls_sup', default=False, action='store_true')
    parser.add_argument('--extra_yesno_lambda', type=float, default=0.5)
    # negative sample parameters
    parser.add_argument('--do_negative_sampling', default=False, action='store_true')
    parser.add_argument('--read_extra_self', default=False, action='store_true')
    parser.add_argument('--sample_ratio', type=float, default=0.5)
    parser.add_argument('--extra_sen_file', type=str, default=None)
    parser.add_argument('--multi_inputs', default=False, action='store_true')
    parser.add_argument('--add_entropy', default=False, action='store_true')
    # label parameters
    parser.add_argument('--do_label', default=False, action='store_true')
    parser.add_argument('--weight_threshold', type=float, default=1.0)
    parser.add_argument('--label_threshold', type=float, default=1.0)

    args = parser.parse_args()

    logger = setting_logger(args.output_dir)
    logger.info('================== Program start. ========================')

    # model parameters
    model_params = prepare_model_params(args)
    # read parameters
    read_params = prepare_read_params(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict and not args.do_label:
        raise ValueError("At least one of `do_train` or `do_predict` or `do_label` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_predict:
        os.makedirs(args.predict_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)

    data_reader = initialize_reader(args.reader_name)

    num_train_steps = None
    if args.do_train:
        if args.multi_inputs:
            train_examples = []
            train_features = []
            features_total_num = 0
            for i in range(int(args.num_train_epochs)):
                train_examples.append(data_reader.read(input_file=args.train_file, **read_params))

                task_name = args.task_name + '-epoch{}'.format(i)
                cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}_{4}_{5}'.format(
                    args.bert_model, str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length), str(args.max_ctx),
                    str(task_name))

                try:
                    with open(cached_train_features_file, "rb") as reader:
                        train_features.append(pickle.load(reader))
                    logger.info(" Loaded train features from {}".format(cached_train_features_file))
                except FileNotFoundError:
                    train_features.append(data_reader.convert_examples_to_features(examples=train_examples[i], tokenizer=tokenizer,
                                                                                   max_seq_length=args.max_seq_length,
                                                                                   doc_stride=args.doc_stride,
                                                                                   max_query_length=args.max_query_length,
                                                                                   is_training=True))
                    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                        logger.info("  Saving train features into cached file %s", cached_train_features_file)
                        with open(cached_train_features_file, "wb") as writer:
                            pickle.dump(train_features[-1], writer)
                features_total_num += len(train_features[-1])
            # for x in train_features:
            #     assert len(x) == len(train_features[0])

            # num_train_steps = int(
            #     len(train_features[0]) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

            num_train_steps = math.ceil(features_total_num / args.train_batch_size / args.gradient_accumulation_steps)
        else:
            task_name = args.task_name
            train_examples = data_reader.read(input_file=args.train_file, **read_params)

            cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}_{4}_{5}'.format(
                args.bert_model, str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length), str(args.max_ctx),
                str(task_name))

            try:
                with open(cached_train_features_file, "rb") as reader:
                    train_features = pickle.load(reader)
                logger.info(" Loaded train features from {}".format(cached_train_features_file))
            except FileNotFoundError:
                train_features = data_reader.convert_examples_to_features(examples=train_examples, tokenizer=tokenizer,
                                                                          max_seq_length=args.max_seq_length,
                                                                          doc_stride=args.doc_stride,
                                                                          max_query_length=args.max_query_length,
                                                                          is_training=True)
                if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                    logger.info("  Saving train features into cached file %s", cached_train_features_file)
                    with open(cached_train_features_file, "wb") as writer:
                        pickle.dump(train_features, writer)

            num_train_steps = math.ceil(
                len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.pretrain is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrain))
        model_state_dict = torch.load(args.pretrain, map_location='cuda:0')
        model = initialize_model(args.bert_name, args.model_file, state_dict=model_state_dict, **model_params)
    else:
        model = initialize_model(args.bert_name, args.model_file, **model_params)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # Remove params without gradient
    param_optimizer = [n for n in param_optimizer if n[1].requires_grad]

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    # Prepare data
    if 'read_state' in read_params:
        read_params['read_state'] = ReadState.NoNegative
    eval_examples = data_reader.read(input_file=args.predict_file, **read_params)
    eval_features = data_reader.convert_examples_to_features(examples=eval_examples, tokenizer=tokenizer,
                                                             max_seq_length=args.max_seq_length,
                                                             doc_stride=args.doc_stride,
                                                             max_query_length=args.max_query_length,
                                                             is_training=False)

    eval_tensors = data_reader.data_to_tensors(eval_features)
    eval_data = TensorDataset(*eval_tensors)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    if args.do_train:
        logger.info("Start training")
        train_loss = AverageMeter()
        best_acc = 0.0
        summary_writer = SummaryWriter(log_dir=args.output_dir)
        global_step = 0
        eval_loss = AverageMeter()

        if args.do_negative_sampling and args.multi_inputs:
            train_dataloaders = []
            for i in range(int(args.num_train_epochs)):
                train_tensors = data_reader.data_to_tensors(train_features[i])
                train_data = TensorDataset(*train_tensors)
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_data)
                else:
                    train_sampler = DistributedSampler(train_data)
                train_dataloaders.append(DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size))
        else:
            train_tensors = data_reader.data_to_tensors(train_features)
            train_data = TensorDataset(*train_tensors)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloaders = [DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)]

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Train
            model.train()
            if args.multi_inputs:
                train_dataloader = train_dataloaders[epoch]
                current_train_features = train_features[epoch]
            else:
                train_dataloader = train_dataloaders[0]
                current_train_features = train_features
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = batch_to_device(batch, device)  # multi-gpu does scattering it-self
                inputs = data_reader.generate_inputs(batch, current_train_features, model_state=ModelState.Train)
                output_dict = model(**inputs)
                loss = output_dict['loss']
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    train_loss.update(loss.item(), args.train_batch_size)
                    summary_writer.add_scalar('train_loss', train_loss.avg, epoch * len(train_dataloader) + step)

            # Evaluation
            model.eval()
            all_results = []
            logger.info("Start evaluating")
            for eval_step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                if n_gpu == 1:
                    batch = batch_to_device(batch, device)  # multi-gpu does scattering it-self
                inputs = data_reader.generate_inputs(batch, eval_features, model_state=ModelState.Evaluate)
                with torch.no_grad():
                    output_dict = model(**inputs)
                    loss, batch_choice_logits = output_dict['loss'], output_dict['yesno_logits']
                    eval_loss.update(loss.item(), args.predict_batch_size)
                    summary_writer.add_scalar('eval_loss', eval_loss.avg, epoch * len(eval_dataloader) + eval_step)
                example_indices = batch[-1]
                for i, example_index in enumerate(example_indices):
                    choice_logits = batch_choice_logits[i].detach().cpu().tolist()

                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResultChoice(unique_id=unique_id, choice_logits=choice_logits))

            data_reader.write_predictions(eval_examples, eval_features, all_results, None, null_score_diff_threshold=0.0)
            yes_metric = data_reader.yesno_cate.f1_measure('yes', 'no')
            no_metric = data_reader.yesno_cate.f1_measure('no', 'yes')
            current_acc = yes_metric['accuracy']
            summary_writer.add_scalar('eval_yes_f1', yes_metric['f1'], epoch)
            summary_writer.add_scalar('eval_yes_recall', yes_metric['recall'], epoch)
            summary_writer.add_scalar('eval_yes_precision', yes_metric['precision'], epoch)
            summary_writer.add_scalar('eval_no_f1', no_metric['f1'], epoch)
            summary_writer.add_scalar('eval_no_recall', no_metric['recall'], epoch)
            summary_writer.add_scalar('eval_no_precision', no_metric['precision'], epoch)
            summary_writer.add_scalar('eval_yesno_acc', current_acc, epoch)
            torch.cuda.empty_cache()

            if current_acc > best_acc:
                best_acc = current_acc
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
            logger.info('Epoch: %d, Accuracy: %f (Best Accuracy: %f)' % (epoch, current_acc, best_acc))
            data_reader.yesno_cate.reset()

        summary_writer.close()

    # Loading trained model.
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    model_state_dict = torch.load(output_model_file, map_location='cuda:0')
    model = initialize_model(args.bert_name, args.model_file, state_dict=model_state_dict, **model_params)
    model.to(device)

    # Write Yes/No predictions
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        test_examples = eval_examples
        test_features = eval_features

        test_tensors = data_reader.data_to_tensors(test_features)
        test_data = TensorDataset(*test_tensors)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.predict_batch_size)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(test_examples))
        logger.info("  Num split examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start predicting yes/no on Dev set.")
        for batch in tqdm(test_dataloader, desc="Testing"):
            if n_gpu == 1:
                batch = batch_to_device(batch, device)  # multi-gpu does scattering it-self
            inputs = data_reader.generate_inputs(batch, test_features, model_state=ModelState.Test)
            with torch.no_grad():
                output_dict = model(**inputs)
                batch_choice_logits = output_dict['yesno_logits']
            example_indices = batch[-1]
            for i, example_index in enumerate(example_indices):
                choice_logits = batch_choice_logits[i].detach().cpu().tolist()

                test_feature = test_features[example_index.item()]
                unique_id = int(test_feature.unique_id)

                all_results.append(RawResultChoice(unique_id=unique_id, choice_logits=choice_logits))

        output_prediction_file = os.path.join(args.predict_dir, 'predictions.json')
        data_reader.write_predictions(eval_examples, eval_features, all_results,
                                      output_prediction_file, null_score_diff_threshold=0.0)
        yes_metric = data_reader.yesno_cate.f1_measure('yes', 'no')
        no_metric = data_reader.yesno_cate.f1_measure('no', 'yes')
        logger.info('Yes Metrics: %s' % json.dumps(yes_metric, indent=2))
        logger.info('No Metrics: %s' % json.dumps(no_metric, indent=2))

    # Labeling sentence id.
    if args.do_label and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        test_examples = data_reader.read(input_file=args.train_file, **read_params)
        test_features = data_reader.convert_examples_to_features(examples=test_examples, tokenizer=tokenizer,
                                                                 max_seq_length=args.max_seq_length,
                                                                 doc_stride=args.doc_stride,
                                                                 max_query_length=args.max_query_length,
                                                                 is_training=False)

        test_tensors = data_reader.data_to_tensors(test_features)
        test_data = TensorDataset(*test_tensors)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.predict_batch_size)

        logger.info("***** Running labeling *****")
        logger.info("  Num orig examples = %d", len(test_examples))
        logger.info("  Num split examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start labeling.")
        for batch in tqdm(test_dataloader, desc="Testing"):
            if n_gpu == 1:
                batch = batch_to_device(batch, device)
            inputs = data_reader.generate_inputs(batch, test_features, model_state=ModelState.Test)
            with torch.no_grad():
                output_dict = model(**inputs)
                batch_choice_logits = output_dict['yesno_logits']
                batch_max_weight_indexes = output_dict['max_weight_index']
                batch_max_weights = output_dict['max_weight']
            example_indices = batch[-1]
            for i, example_index in enumerate(example_indices):
                choice_logits = batch_choice_logits[i].detach().cpu().tolist()
                max_weight_index = batch_max_weight_indexes[i].detach().cpu().tolist()
                max_weight = batch_max_weights[i].detach().cpu().tolist()

                test_feature = test_features[example_index.item()]
                unique_id = int(test_feature.unique_id)

                all_results.append(
                    FullResult(unique_id=unique_id, choice_logits=choice_logits, max_weight_index=max_weight_index,
                               max_weight=max_weight))

        output_prediction_file = os.path.join(args.predict_dir, args.label_output_file)
        data_reader.write_sentence_predictions(test_examples, test_features, all_results,
                                               output_prediction_file, weight_threshold=args.weight_threshold,
                                               label_threshold=args.label_threshold)


def batch_to_device(batch: Tuple[torch.Tensor], device):
    # batch[-1] don't move to gpu.
    output = []
    for t in batch[:-1]:
        output.append(t.to(device))
    output.append(batch[-1])
    return output


if __name__ == "__main__":
    main()

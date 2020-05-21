import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# model
bert_base_model = "../BERT/bert-base-uncased.tar.gz"
bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
bert_large_model = "../BERT/bert-large-uncased.tar.gz"
bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"
train_file = '../../ms-marco/train-yesno-throw-yes-uni-dp0.7.json'
dev_file = '../../ms-marco/dev-yesno-throw-yes-uni-dp0.75.json'

task_name = 'marco-dp0.7-0.75'

reader_name = 'ms-marco'

coqa_mlp_pretrained_model = 'experiments/coqa-mlp/v3.0/pytorch_model.bin'
coqa_hie_pretrained_model = 'experiments/coqa-hie/v3.0/pytorch_model.bin'
coqa_hie_super_pretrained_model = 'experiments/coqa-hie-super/v2.0/pytorch_model.bin'

# mlp pretrained

# bert_name = 'mlp'
#
# output_dir = f"experiments/marco-dp0.7-0.75-mlp-pretrain/v1.0/"
#
# cmd = f"python main.py \
#             --bert_model bert-base-uncased \
#             --vocab_file {bert_base_vocab} \
#             --model_file {bert_base_model} \
#             --output_dir {output_dir} \
#             --predict_dir {output_dir} \
#             --train_file {train_file} \
#             --predict_file {dev_file} \
#             --max_seq_length 400 --max_query_length 50 \
#             --do_train --do_predict --train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 \
#             --pretrain {coqa_mlp_pretrained_model} \
#             --learning_rate 3e-5 \
#             --num_train_epochs 3.0 \
#             --max_ctx 3 \
#             --bert_name {bert_name} \
#             --task_name {task_name} \
#             --reader_name {reader_name} "
#
# print(cmd)
# os.system(cmd)

bert_name = 'hie'

output_dir = f"experiments/marco-dp0.7-0.75-hie-pretrain/v1.3/"

cmd = f"python main.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir} \
            --train_file {train_file} \
            --predict_file {dev_file} \
            --max_seq_length 400 --max_query_length 50 \
            --do_train --do_predict --train_batch_size  8 --predict_batch_size 8 --max_answer_length 15 \
            --pretrain {coqa_hie_pretrained_model} \
            --learning_rate 3e-5 \
            --num_train_epochs 2.0 \
            --max_ctx 3 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} " \
    f"--evidence_lambda 0.0"

print(cmd)
os.system(cmd)

output_dir = f"experiments/marco-dp0.7-0.75-hie-super-pretrain/v1.3/"

cmd = f"python main.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir} \
            --train_file {train_file} \
            --predict_file {dev_file} \
            --max_seq_length 400 --max_query_length 50 \
            --do_train --do_predict --train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 \
            --pretrain {coqa_hie_super_pretrained_model} \
            --learning_rate 3e-5 \
            --num_train_epochs 2.0 \
            --max_ctx 3 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} " \
    f"--evidence_lambda 0.0"

print(cmd)
os.system(cmd)

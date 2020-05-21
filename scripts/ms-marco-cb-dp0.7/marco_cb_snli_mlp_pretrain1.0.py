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
train_file = '../../ms-marco/dp0.7/train-yesno-cb-dp70.json'
dev_file = '../../ms-marco/dp0.7/dev-yesno-cb-dp70.json'

task_name = 'marco-cb-dp0.7'

reader_name = 'cb-marco'

coqa_mlp_pretrained_model = 'experiments/snli-mlp/v1.0/pytorch_model.bin'
coqa_hie_pretrained_model = 'experiments/snli-hie/v1.0/pytorch_model.bin'
coqa_hie_super_pretrained_model = 'experiments/snli-hie-super/v1.0/pytorch_model.bin'

# mlp pretrained

bert_name = 'mlp'

output_dir = f"experiments/marco-cb-dp0.7-mlp-snli-pretrained/v1.0/"

cmd = f"python main.py \
            --bert_model bert-base-uncased \
            --vocab_file {bert_base_vocab} \
            --model_file {bert_base_model} \
            --output_dir {output_dir} \
            --predict_dir {output_dir} \
            --train_file {train_file} \
            --predict_file {dev_file} \
            --max_seq_length 480 --max_query_length 50 \
            --do_train --do_predict --train_batch_size 8 --predict_batch_size 8 --max_answer_length 15 \
            --pretrain {coqa_mlp_pretrained_model} \
            --learning_rate 3e-5 \
            --num_train_epochs 3.0 \
            --max_ctx 3 \
            --bert_name {bert_name} \
            --task_name {task_name} \
            --reader_name {reader_name} "

print(cmd)
os.system(cmd)

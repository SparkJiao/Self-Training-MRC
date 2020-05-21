"""
vote_2_0_1_3.py means using model{0} and model{1} to vote for model{3}. All models come from version{2.0}
"""
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# model0_pred = 'experiments/coqa-tri-training/v2.0/coqa-neg-tri-1/evidence-predictions-0.9-0.8.json'
# model1_pred = 'experiments/coqa-tri-training/v2.0/coqa-neg-tri-3/evidence-predictions-0.9-0.8.json'
# model2_dir = 'experiments/coqa-tri-training/v2.0/coqa-neg-tri-0/'

# Re1
# model0_pred = 'experiments/coqa-tri-training/v2.0/coqa-re1-tri-1/evidence-predictions-0.9-0.8.json'
# model1_pred = 'experiments/coqa-tri-training/v2.0/coqa-re1-tri-3/evidence-predictions-0.9-0.8.json'
# model2_dir = 'experiments/coqa-tri-training/v2.0/coqa-re1-tri-0/'
# old_file = 'experiments/coqa-tri-training/v2.0/coqa-neg-tri-0/v2.0-1-3-vote.pkl'
# Labeled 11390 data in total

model0_pred = 'experiments/coqa-tri-training/v2.0/coqa-re2-tri-1/evidence-predictions-0.9-0.8.json'
model1_pred = 'experiments/coqa-tri-training/v2.0/coqa-re2-tri-3/evidence-predictions-0.9-0.8.json'
model2_dir = 'experiments/coqa-tri-training/v2.0/coqa-re2-tri-0/'
old_file = 'experiments/coqa-tri-training/v2.0/coqa-re1-tri-0/v2.0-1-3-vote.pkl'
# Labeled 16008 data in total

output_file = model2_dir + 'v2.0-1-3-vote.pkl'

cmd = f"python general_util/evidence_vote.py --input_file1 {model0_pred} --input_file2 {model1_pred} --output_file {output_file} " \
    f"--old_file {old_file}"

logger.info(cmd)
os.system(cmd)

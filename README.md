# A Self-Training Method for MRC

This is the pytorch implementation of the paper: **A Self-Training Method for Machine Reading Comprehension with Soft Evidence Extraction**. Yilin Niu, Fangkai Jiao, Mantong Zhou, Ting Yao, Jingfang Xu and Minlie Huang. ***ACL 2020***.

[PDF]()

## Requirements

``pip install -r requirements.txt``  
Besides, the [apex](https://github.com/NVIDIA/apex) toolkit should be installed.  

**Note**: For experiments based on RoBERTa, the [transformers](https://github.com/huggingface/transformers) should be installed. Version is specified as `transformers==2.1.0`.
We recommend to use another virtual environment to conduct the experiments with RoBERTa, since we have observed this package has a effect on the performance for other experiments.  

**Note**: The runtime experiments are different in our experiments. Most of them are reproducible using TITAN XP and others were conducted on RTX 2080 Ti.

## Scripts for Dataset Preprocess

Dream: `data_preprocess/dream_data_preprocess.ipynb`  
MS Marco: `data_preprocess/ms_marco_extract_yesno_combine.ipynb`  
Multi-RC: `data_preprocess/multi_rc_data_process.ipynb`  
RACE: `data_preprocess/race_data_processing.ipynb`

To use these scripts, change the data file path in the notebook as your own path and run it. Then change the input file path in the experiment scripts to the output file.
For CoQA, you don't need extra preprocess and the initial data file can be assigned as the input file.

Since the dataset MS Marco is extract from the initial dataset through a randomly drop process, we also release the dataset we extracted by ourselves for testing in ``data/ms``

## Scripts for Experiments
### RACE
#### Middle
MLP: `scripts/race-f-multiple-evidence/topk_evidence/middle/scratch/scratch1.0.py`  
rule: `scripts/race-f-multiple-evidence/topk_evidence/middle/rule/rule_idf_1.0.py`  
Reinforce: `scripts/race-f-multiple-evidence/topk_evidence/middle/reinforce/reinforce_pipeline.py` (TITAN XP)    
Co-Training: `scripts/race-f-multiple-evidence/topk_evidence/middle/co-training/co-training1.0.py`  
Self-Training: `scripts/race-f-multiple-evidence/topk_evidence/middle/self-training/self-training1.0.py`

#### High
MLP: `scripts/race-f-multiple-evidence/topk_evidence/high/scratch/scratch1.0.py`  
rule: `scripts/race-f-multiple-evidence/topk_evidence/high/rule/rule_idf_1.0.py`  
Reinforce: `scripts/race-f-multiple-evidence/topk_evidence/high/reinforce/reinforce_pipeline.py`  (TITAN XP)  
Co-Training: `scripts/race-f-multiple-evidence/topk_evidence/high/co-training/co-training2.0.py`  
Self-Training: `scripts/race-f-multiple-evidence/topk_evidence/high/self-training/self-training1.2.py`  

#### All
MLP: `scripts/race-f-multiple-evidence/topk_evidence/combine/scratch/scratch1.0.py`  
rule: `scripts/race-f-multiple-evidence/topk_evidence/combine/rule/rule_idf_1.0.py`  
Reinforce: `scripts/race-f-multiple-evidence/topk_evidence/combine/reinforce/reinforce_pipeline.py`  (TITAN XP)  
Self-Training: `scripts/race-f-multiple-evidence/topk_evidence/combine/self-training/self-training4.0.py`  
Co-Training: `scripts/race-f-multiple-evidence/topk_evidence/combine/co-training/co-training2.0.py`

### CoQA

BERT-MLP/HA/HA+Gold: `scripts/coqa/coqa_scratch_lr_test2.py`  
Self-Training: `scripts/coqa-co-training/topk_evidence_self_training/self_training1.0.py`  
Co-Training: `scripts/coqa-co-training/topk_evidence/cotraining_top_k1.3.py`  
RoBERTa-Self-Training: `2.0第一轮参数+1.2后续轮参数`  
Rule:   
Reinforce: `scripts/coqa-co-training/reinforce/gumbel_pretrain2.0.py` + `scripts/coqa-co-training/reinforce/reinforce3.0.py`

### MARCO

Rule: `scripts/marco-cb-dp0.7-co-training/top-k-rule/rule_idf1.1.py`  
BERT-MLP/HA: `scripts/marco-cb-dp0.7-co-training/scratch1.0.py`  
Reinforce: `scripts/marco-cb-dp0.7-co-training/reinforce/pipeline1.0.py`  
Self-Training: `scripts/marco-cb-dp0.7-co-training/top-k-self-training/self_training1.0.py`  
Co-Training: `scripts/marco-cb-dp0.7-co-training/top-k-co-training/cotraining_top_k1.2.py`  

### BoolQ

### Multi-RC

MLP: `scripts/multi_rc/scratch/mlp1.0.py`  
HA: `scripts/multi_rc/topk_scratch/hie.py`  
HA-super: `scripts/multi_rc/topk_scratch/hie-super.py`  
Rule: `scripts/multi_rc/topk_evidence_rule/rule_idf1.1.py`  
Reinforce: `scripts/multi_rc/reinforce/reinforce_fine_tune1.0.py`  
Self-Training: `scripts/multi_rc/topk_evidence_self_training/self_training2.0.py`  
Co-Training: `scripts/multi_rc/topk_evidence_co_training/cotraining_top_k2.0.py`  

### Dream   

MLP: `scripts/dream/scratch/mlp1.0.py`  
HA: `scripts/dream/self-training/self-training4.0.py` -recurrent0
Self-Training: `scripts/dream/self-training/self-training4.0.py`  
Co-Training: `scripts/dream/co-training/co-training2.0.py`  
Rule: `scripts/dream/rule/rule_idf_1.0.py`  
Reinforce: `scripts/dream/reinforce/reinforce_fine_tune1.1.py`


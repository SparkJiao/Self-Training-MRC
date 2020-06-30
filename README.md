# A Self-Training Method for MRC

This is the pytorch implementation of the paper: **A Self-Training Method for Machine Reading Comprehension with Soft Evidence Extraction**. Yilin Niu, Fangkai Jiao, Mantong Zhou, Ting Yao, Jingfang Xu and Minlie Huang. ***ACL 2020***.

[PDF](https://arxiv.org/pdf/2005.05189.pdf)

## Requirements

Our CUDA toolkit version is 10.1. Firstly, you need to create a new virtual environment and install the requirements:

````
$ conda create -n self-training-mrc python=3.6.9
$ conda activate self-training-mrc
$ cd Self-Training-MRC
$ pip install -r requirements.txt
````

Besides, the [apex](https://github.com/NVIDIA/apex) toolkit with specified version should be installed:  

````
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ git reset --hard 66158f66a027a2e7de483d9b3e6ca7c889489b13
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
````

**Note**: If your CUDA version is 10.1 rather than 10.0, the installation of apex may throw a error caused by minor version mismatch since Pytorch only provides the package ``torch==1.1.0`` complied with ``CUDA==10.0``. As a solution, please change the check code in ``apex/setup.py`` line 75 from  
``if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):``  
to  
``if (bare_metal_major != torch_binary_major):``  
and the installation will be completed.

Before running the experiments, the nltk data should be downloaded if it haven't been:
````
python  
>>> import nltk  
>>> nltk.download('punkt')   
````

**Note**: For experiments based on RoBERTa, the [transformers](https://github.com/huggingface/transformers) should be installed. Version is specified as `transformers==2.1.0`.
We recommend to use another virtual environment to conduct the experiments with RoBERTa, since we have observed this package has a effect on the performance for other experiments.  

**Note**: The hardware environments are different in our experiments. Most of them are reproducible using RTX 2080Ti and others were conducted on TITAN XP. We will give the notes.

## Scripts for Dataset Preprocess

Dream: `data_preprocess/dream_data_preprocess.ipynb`  
MS Marco: `data_preprocess/ms_marco_extract_yesno_combine.ipynb`  
Multi-RC: `data_preprocess/multi_rc_data_process.ipynb`  
RACE: `data_preprocess/race_data_processing.ipynb`

To use these scripts, change the data file path in the notebook as your own path and run it. Then change the input file path in the experiment scripts to the output file.
For CoQA, you don't need extra preprocess and the initial data file can be assigned as the input file.

### Data files affected by the randomness of local system

To make sure that our experiments is reproducible, we will release the processed data files including:
- Processed RACE dataset file
- Processed MS Marco dataset file
- Extracted evidence sentence files of RACE-High for reproducing BERT-HA-Rule

In the dataset file, the randomness will affect the order of instances while in the evidence files, it will be affected when there are several sentences with the same similarity with the passage and question. 
The dataset files can be found under the `data` directory and the rule-based evidence sentence files can be found under the `experiments/race/topk-evidence/high/rule/` directory.  

## Scripts for Experiments
### RACE
#### Middle
MLP: `scripts/race-f-multiple-evidence/topk_evidence/middle/scratch/scratch1.0.py`  (RTX 2080Ti|checked)   
rule: `scripts/race-f-multiple-evidence/topk_evidence/middle/rule/rule_idf_1.0.py`  (RTX 2080Ti|checked)  
Reinforce: `scripts/race-f-multiple-evidence/topk_evidence/middle/reinforce/reinforce_pipeline.py` (TITAN XP|checked)    
Co-Training: `scripts/race-f-multiple-evidence/topk_evidence/middle/co-training/co-training1.0.py`  
Self-Training: `scripts/race-f-multiple-evidence/topk_evidence/middle/self-training/self-training1.0.py`  (RTX 2080Ti|checked)  

#### High
MLP: `scripts/race-f-multiple-evidence/topk_evidence/high/scratch/scratch1.0.py`  (RTX 2080Ti|checked)   
rule: `scripts/race-f-multiple-evidence/topk_evidence/high/rule/rule_idf_1.0.py`  (RTX 2080Ti|checking)  
Reinforce: `scripts/race-f-multiple-evidence/topk_evidence/high/reinforce/reinforce_pipeline.py`  (TITAN XP|checked)  
Co-Training: `scripts/race-f-multiple-evidence/topk_evidence/high/co-training/co-training2.0.py`  
Self-Training: `scripts/race-f-multiple-evidence/topk_evidence/high/self-training/self-training1.2.py`  (RTX 2080Ti|checked)  

#### All
MLP: `scripts/race-f-multiple-evidence/topk_evidence/combine/scratch/scratch1.0.py`  
rule: `scripts/race-f-multiple-evidence/topk_evidence/combine/rule/rule_idf_1.0.py`  
Reinforce: `scripts/race-f-multiple-evidence/topk_evidence/combine/reinforce/reinforce_pipeline.py`  (TITAN XP)  
Self-Training: `scripts/race-f-multiple-evidence/topk_evidence/combine/self-training/self-training4.0.py`  
Co-Training: `scripts/race-f-multiple-evidence/topk_evidence/combine/co-training/co-training2.0.py`

### CoQA

BERT-MLP/HA/HA+Gold: `scripts/coqa/coqa_scratch_lr_test2.py`  (TITAN XP|checked)  
Self-Training: `scripts/coqa-co-training/topk_evidence_self_training/self_training1.0.py`  (TITAN XP|checked)  
Co-Training: `scripts/coqa-co-training/topk_evidence/cotraining_top_k1.3.py`  
RoBERTa-Self-Training: `2.0 for iteration 0 and 1.2 for iterations afterwards.`  
Rule:   
Reinforce: `scripts/coqa-co-training/reinforce/gumbel_pretrain2.0.py` + `scripts/coqa-co-training/reinforce/reinforce3.0.py`  (TITAN XP|checked)  

### MARCO

Rule: `scripts/marco-cb-dp0.7-co-training/top-k-rule/rule_idf1.1.py`    (TITAN XP|checked)  
BERT-MLP/HA: `scripts/marco-cb-dp0.7-co-training/scratch1.0.py`  (TITAN XP|checked)  
Reinforce: `scripts/marco-cb-dp0.7-co-training/reinforce/pipeline1.0.py`  (TITAN XP|checked)  
Self-Training: `scripts/marco-cb-dp0.7-co-training/top-k-self-training/self_training1.0.py`  (TITAN XP|checked)  
Co-Training: `scripts/marco-cb-dp0.7-co-training/top-k-co-training/cotraining_top_k1.2.py`  

### BoolQ

### Multi-RC

MLP: `scripts/multi_rc/scratch/mlp1.0.py`  (RTX 2080Ti|checked)  
HA: `scripts/multi_rc/topk_scratch/hie.py`  (RTX 2080Ti|checked)  
HA-super: `scripts/multi_rc/topk_scratch/hie-super.py`  (RTX 2080Ti|checked)  
Rule: `scripts/multi_rc/topk_evidence_rule/rule_idf1.1.py`  (RTX 2080Ti|checked)  
Reinforce: `scripts/multi_rc/reinforce/reinforce_fine_tune1.0.py`  (TITAN XP|checked)  
Self-Training: `scripts/multi_rc/topk_evidence_self_training/self_training2.0.py`  (RTX 2080Ti|checked)  
Co-Training: `scripts/multi_rc/topk_evidence_co_training/cotraining_top_k2.0.py`  

### Dream   

MLP: `scripts/dream/scratch/mlp1.0.py`  (RTX 2080Ti|checked)  
HA: `scripts/dream/self-training/self-training4.0.py` -recurrent0 (RTX 2080Ti|checked)  
Self-Training: `scripts/dream/self-training/self-training4.0.py`  (RTX 2080Ti|checked)  
Co-Training: `scripts/dream/co-training/co-training2.0.py`  
Rule: `scripts/dream/rule/rule_idf_1.0.py`  (RTX 2080Ti|checked)  
Reinforce: `scripts/dream/reinforce/reinforce_fine_tune1.1.py`  (TITAN XP|checked)  

## Citing

Please kindly cite our paper if this paper and the code are helpful.

````
@inproceedings{stm-mrc-2020,
  author    = {Yilin Niu and Fangkai Jiao and Mantong Zhou and Ting Yao and Jingfang Xu and Minlie Huang},
  title     = {A Self-Training Method for Machine Reading Comprehension with Soft Evidence Extraction},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages     = {3916--3927},
  publisher = {{ACL}},
  year      = {2020}
}
````


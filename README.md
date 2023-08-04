# Structure-free Graph Condensation (SFGC): From Large-scale Graphs to Condensed Graph-free Data

This is the Pytorch implementation for "Structure-free Graph Condensation (SFGC): From Large-scale Graphs to Condensed Graph-free Data"

### Requirements

torch==1.7.1+cu110

torch-geometric==1.6.3

torch-sparse==0.6.9

torch-scatter==2.0.7

tensorboardx==2.6

deeprobust==0.2.4

matplotlib==3.5.3

scikit-learn==1.0.2


## Instructions

Following is the step-by-step instruction to reproduce our proposed method SFGC. 
Taking citeseer and reddit for examples of transductive and inductive settings, we also provide some of our checkpoints 
and saved models to facilite the separate execution of each step in README.md.

More detailed hyperparameter settings for all datasets of all steps are provided in the scripts.
Due to the large file limitation in Github, we will release the large buffer files later by googledrive link after the review.
This is only for illustrating the reproducibility of our proposed SFGC for review.

(1) Run to generate the buffer for keeping the model's training parameter distribution (training trajectory)

For examples:

Dataset: Citeseer

```
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.001 \
--teacher_epochs 800 --dataset citeseer --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0
```

Dataset: Reddit

```
CUDA_VISIBLE_DEVICES=2 python buffer_inductive.py --device cuda:0 --lr_teacher 0.001 \
--teacher_epochs 1000 --dataset reddit --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0
```

(2) Use the coreset method to initialize the synthesized small-scale graph node features

For examples:

Dataset: Citeseer

```
CUDA_VISIBLE_DEVICES=3 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy ''
```

Dataset: Reddit

```
CUDA_VISIBLE_DEVICES=3 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''
```

(3) Distill under training trajectory and coreset initialization to generate synthesized small-scale structure-free graph data

For examples:

Dataset: Citeseer

```
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 \
--lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=400 --lr_student=0.6 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=300 \
--buffer_path=${your-buffer} \
--coreset_init_path './logs/Coreset/citeseer-reduce_1.0-coreset' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 
```

Dataset: Reddit

```
CUDA_VISIBLE_DEVICES=1 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.05 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=50 --syn_steps=900 \
--buffer_path=${your-buffer} \
--coreset_init_path './logs/Coreset/reddit-reduce_0.001-coreset' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg=0.01 --eval_interval 1 --ITER 3000 --samp_iter 1 --samp_num_per_class 50
```

(4) Training with the small-scale structure free graph data and test on the large-scale graph test set:

For example:

Dataset: Citeseer

```
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.5-distill'
```
Dataset: Reddit

```
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.001 \
--test_lr_model=0.01 --test_wd=0.005 --test_model_iters 300 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/reddit-reduce_0.001-distill'
```

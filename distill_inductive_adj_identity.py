import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from metagtt_inductive_adj_identity import MetaGtt
from utils_graphsaint import DataGraphSAINT
import logging
import sys
import datetime
import os
from tensorboardX import SummaryWriter
import deeprobust.graph.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dataset', type=str, default='flickr',choices=['flickr', 'reddit'])
parser.add_argument('--student_nlayers', type=int, default=2)
parser.add_argument('--student_hidden', type=int, default=256)
parser.add_argument('--student_dropout', type=float, default=0.0)

parser.add_argument('--lr_feat', type=float, default=0.0001)
parser.add_argument('--lr_student', type=float, default=0.01, help='initialization for synthetic learning rate')
parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')

parser.add_argument('--reduction_rate', type=float, default=0.5)
parser.add_argument('--seed_student', type=int, default=15, help='Random seed for distill student model')
parser.add_argument('--save_log', type=str, default='logs', help='path to save logs')
parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
parser.add_argument('--max_files', type=int, default=None,
                    help='number of expert files to read (leave as None unless doing ablations)')
parser.add_argument('--max_experts', type=int, default=None,
                    help='number of experts to read per file (leave as None unless doing ablations)')

parser.add_argument('--expert_epochs', type=int, default=50,help='how many expert epochs the target params are')
parser.add_argument('--start_epoch', type=int, default=10, help='max epoch we can start at')
parser.add_argument('--ITER', type=int, default=5000, help='how many distillation steps to perform')
parser.add_argument('--syn_steps', type=int, default=20,help = 'how many steps to take on synthetic data')
parser.add_argument('--buffer_path', type=str, default='./logs/Buffer/cora-20220925-225653-091173', help='buffer path')
parser.add_argument('--condense_model', type=str, default='GCN', help='Default condensation model')

parser.add_argument('--eval_model', type=str, default='GCN', help='evaluation model for saving best feat')
parser.add_argument('--eval_type', type=str, default='S',help='eval_mode, check utils.py for more info')
parser.add_argument('--eval_interval', type=int, default=20, help='how often to evaluate')
parser.add_argument('--eval_nums', type=int, default=5, help='how many networks to evaluate on')
parser.add_argument('--eval_train_iters', type=int, default=100, help='for evaluating updated feat and adj')
parser.add_argument('--eval_wd', type=float, default=5e-4)
#parser.add_argument('--eval_lr', type=float, default=0.01, help='lr for evaluation')

parser.add_argument('--initial_save', type=int, default=0, help='whether save initial feat and syn')
parser.add_argument('--k', type=int, default=10, help='k for initializing with knn')
parser.add_argument('--knn_metric', type=str, default='cosine', help='See choices', choices=['cosine', 'minkowski'])
parser.add_argument('--interval_buffer', type=int, default=1, choices=[0,1],help='whether use interval buffer')
parser.add_argument('--rand_start', type=int, default=1,choices=[0,1], help='whether use random start')
parser.add_argument('--optimizer_con', type=str, default='Adam', help='See choices', choices=['Adam', 'SGD'])
parser.add_argument('--optim_lr', type=int, default=0, help='whether use LR lr learning optimizer')
parser.add_argument('--optimizer_lr', type=str, default='Adam', help='See choices', choices=['Adam', 'SGD'])
parser.add_argument('--optimizer_eval', type=str, default='Adam', help='See choices', choices=['Adam', 'SGD'])

parser.add_argument('--ntk_reg', type=float, default=5e-2, help='L2 penalty reg param')
parser.add_argument('--samp_iter', type=int, default=5, help='sampling numbers in the validation for ntk')
parser.add_argument('--samp_num_per_class', type=int, default=10, help='sampling numbers for each class')

parser.add_argument('--coreset_init_path', type=str, default='logs/Coreset/cora-reduce_0.5-20221024-112028-667459')
parser.add_argument('--coreset_method', type=str, default='kcenter')
parser.add_argument('--coreset_seed', type=int, default=15)

args = parser.parse_args()


log_dir = './' + args.save_log + '/Distill/{}-reduce_{}-{}'.format(args.dataset, str(args.reduction_rate),
                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('This is the log_dir: {}'.format(log_dir))


# random seed setting
random.seed(args.seed_student)
np.random.seed(args.seed_student)
torch.manual_seed(args.seed_student)
torch.cuda.manual_seed(args.seed_student)
device = torch.device(args.device)
logging.info('args = {}'.format(args))
#set = ['cora','citeseer','ogbn-arxiv','flickr', 'reddit']
#args.dataset ='cora'
#args.reduction_rate =0.01
data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset)
    data = Transd2Ind(data_full)
args.log_dir = log_dir
agent = MetaGtt(data, args, device=device)
writer = SummaryWriter(log_dir + '/tbx_log')

agent.distill(writer)

logging.info('Finish! Log_dir: {}'.format(log_dir))

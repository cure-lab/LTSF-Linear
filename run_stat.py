import argparse
import torch
from ltsf_linear.exp.exp_stat import Exp_Main
import random
import numpy as np


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='Not implemented')
parser.add_argument('--embed', type=str, default='timeF',
                    help='Not implemented')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--sample', type=float, default=1, help='Sampling percentage, the inference time of ARIMA and SARIMA is too long, you might sample 0.01')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
                    
# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length') # Just for reusing data loader
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='Not implemented')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--batch_size', type=int, default=100, help='batch size of train input data')
parser.add_argument('--des', type=str, default='test', help='exp description')

args = parser.parse_args()
args.use_gpu = False
print('Args in experiment:')
print(args)
Exp = Exp_Main

setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}'.format(
    args.model_id,
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.pred_len,
    args.des, 0)

exp = Exp(args)  # set experiments
print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting)
       

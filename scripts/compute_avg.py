import pandas as pd

import os
import argparse
import json
from types import SimpleNamespace


parser = argparse.ArgumentParser(description='program args')
parser.add_argument('-c', '--config', type=str, required=True, help='config file path')
args = parser.parse_args()
with open(args.config, 'r') as json_data:
    config_data = json.load(json_data)
train_args = SimpleNamespace(**config_data['train'])

data = pd.read_csv(os.path.join(train_args.save_path, 'testing_indicators.csv'))

print('--------------------')
col_names = ['PSNR1', 'SSIM1', 'LPIPS1', 'PSNR2', 'SSIM2', 'LPIPS2']
for name in col_names:
    average = data[name].mean()
    print(name, average)
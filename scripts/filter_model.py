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

df = pd.read_csv(os.path.join(train_args.save_path, 'training_losses.csv'))
sorted_df = df.sort_values('Val Total Loss')
sorted_epochs = sorted_df['Epoch'].head(5).values
print('top 5 models:', sorted_epochs)

model_fold_path = os.path.join(train_args.save_path, 'models')
file_names = os.listdir(model_fold_path)
for file_name in file_names:
    if file_name.endswith(".pt"):
        file_number = int(file_name[5:-3])
        if file_number not in sorted_epochs:
            file_path = os.path.join(model_fold_path, file_name)
            os.remove(file_path)
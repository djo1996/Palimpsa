# import pandas as pd 
# import wandb
# api = wandb.Api()

# # Project is specified by <entity/project-name>
# runs = api.runs("djohan-bonnet-technologiezentrum-am-europaplatz/MQAR_1")

# summary_list, config_list, name_list = [], [], []
# for run in runs: 
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files 
#     summary_list.append(run.summary._json_dict)

#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append(
#         {k: v for k,v in run.config.items()
#           if not k.startswith('_')})

#     # .name is the human-readable name of the run.
#     name_list.append(run.name)

# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })

# runs_df.to_csv("project.csv")

import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("djohan-bonnet-technologiezentrum-am-europaplatz/MQAR_2")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")

import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load the CSV file
# -----------------------------
csv_path = "project.csv"  # Adjust the path if needed
runs_df = pd.read_csv(csv_path)

# -----------------------------
# Step 2: Convert string representations of dicts back into dictionaries
# -----------------------------
# The 'summary' and 'config' columns are saved as strings.
runs_df['summary'] = runs_df['summary'].apply(ast.literal_eval)
runs_df['config'] = runs_df['config'].apply(ast.literal_eval)

# -----------------------------
# Step 3: Extract required fields from the dictionaries
# -----------------------------
# Extract validation accuracy from the summary dictionary.

# Extract fields from summary and config
runs_df['valid/accuracy'] = runs_df['summary'].apply(lambda s: s.get('valid/accuracy'))

# Properly extract the nested sequence_mixer name
runs_df['name_model'] = runs_df['config'].apply(
    lambda s: s.get('model', {}).get('sequence_mixer', {}).get('name')
)

runs_df['d_model'] = runs_df['config'].apply(
    lambda s: s.get('model', {}).get('d_model')
)


print(runs_df['valid/accuracy'])
print(runs_df['name'])
print(runs_df['config'][0])
print(runs_df['name_model'])
print(runs_df['d_model'])

import matplotlib.pyplot as plt

# Group by model type and d_model, get the best accuracy for each (d_model, model_type) pair
grouped = runs_df.groupby(['name_model', 'd_model'])['valid/accuracy'].max().reset_index()

# Plot one curve per model type

markers = {
    'zoology.mixers.hybrid.Hybrid': 'o',
    'zoology.mixers.mamba.Mamba': 's',
    'zoology.mixers.mambayes.Mambayes': 'D'
}
colors = {
    'zoology.mixers.hybrid.Hybrid': '#1f77b4',
    'zoology.mixers.mamba.Mamba': '#ff7f0e',
    'zoology.mixers.mambayes.Mambayes': '#2ca02c'
}
labels = {
    'zoology.mixers.mamba.Mamba': 'Mamba',
    'zoology.mixers.mambayes.Mambayes': 'Metaplastic-self-attention',
    'zoology.mixers.hybrid.Hybrid': 'Based'
}
# d_model is 32 64 128 256 
# Convert d_model to string for categorical plotting
grouped['d_model_str'] = grouped['d_model'].astype(str)

# Set up plot
plt.figure(figsize=(8, 4))

for model_name in grouped['name_model'].unique():
    model_data = grouped[grouped['name_model'] == model_name].sort_values(by='d_model')
    label_name = {
        'zoology.mixers.mamba.Mamba': 'Mamba',
        'zoology.mixers.mambayes.Mambayes': 'Metaplastic-self-attention',
        'zoology.mixers.hybrid.Hybrid': 'Based'
    }.get(model_name, model_name)
    
    # Use categorical x-axis
    if model_name != "zoology.mixers.msa.Msa": 
        plt.plot(model_data['d_model_str'], model_data['valid/accuracy'], '--',
                    marker=markers.get(model_name, 'o'),
                    color=colors.get(model_name, None),
                    label=labels.get(model_name, model_name),
                    linewidth=2,
                    markersize=8)
# Labels and formatting
plt.xlabel('Model dimension', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('Sequence lenght: 512 / kv pairs: 64', fontsize=14, weight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=10, frameon=True, loc='lower right')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Save and show
plt.savefig("results_d_model_ticks.png", dpi=300)
plt.show()


# BUT I would prefer to plot with respect of state size instead of d_model

# for based it is : 
# 5508.0

# 11016.0

# 20196.0

# 40392.0 

# for mamba and mambayes it is 

# 32*32

# 64*64

# 128*128

# 256*256

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

# Compute state size
def compute_state_size(row):
    if row['name_model'] == 'zoology.mixers.hybrid.Hybrid':
        based_sizes = {32: 5508.0, 64: 11016.0, 128: 20196.0, 256: 40392.0}
        return based_sizes.get(row['d_model'], None)
    elif row['name_model'] == "zoology.mixers.attention.MHA":
        based_sizes = {4: 64*64, 8: 128*64, 16: 256*64, 32: 512*64}
        return based_sizes.get(row['d_model'], None)
    else:
        based_sizes = {32: 64*64, 64: 128*64, 128: 256*64, 256: 512*64}
        return based_sizes.get(row['d_model'], None)

# Calculate state size
grouped['state_size'] = grouped.apply(compute_state_size, axis=1)

# Plot setup
plt.figure(figsize=(8, 6))

markers = {
    'zoology.mixers.hybrid.Hybrid': 'o',
    'zoology.mixers.mamba.Mamba': 's',
    'zoology.mixers.mambayes.Mambayes': 'D',
    'zoology.mixers.msa.Msa': 'D',
    "zoology.mixers.attention.MHA" : "x"
}
colors = {
    'zoology.mixers.hybrid.Hybrid': '#1f77b4',
    'zoology.mixers.mamba.Mamba': 'C2',
    'zoology.mixers.mambayes.Mambayes': 'C1',
    'zoology.mixers.msa.Msa': 'C1',
    "zoology.mixers.attention.MHA" : "C4"
}
labels = {
    'zoology.mixers.mamba.Mamba': 'Mamba',
    'zoology.mixers.mambayes.Mambayes': 'Metaplastic-self-attention',
    'zoology.mixers.msa.Msa': 'Metaplastic-self-attention',
    'zoology.mixers.hybrid.Hybrid': 'Based',
    "zoology.mixers.attention.MHA" : "Transformer"
}

# Plot each model
for model_name in grouped['name_model'].unique():
    model_data = grouped[grouped['name_model'] == model_name].sort_values(by='state_size')
    print(model_name)
    # if model_name != 'zoology.mixers.mamba.Mamba' and model_name != "zoology.mixers.attention.MHA": 

    if model_name != 'zoology.mixers.mambayes.Mambayes' : 
        plt.plot(
                model_data['state_size'],
                model_data['valid/accuracy'],'--',
                marker=markers.get(model_name, 'o'),
                color=colors.get(model_name, None),
                label=labels.get(model_name, model_name),
                linewidth=2,
                markersize=8
            )

# Log scale base 2
plt.xscale('log', base=2)

# Set major ticks at powers of 2
ax = plt.gca()
ax.xaxis.set_major_locator(LogLocator(base=2.0, numticks=12))

# Format ticks as powers of 2
def power_of_two(x, _):
    if x == 0:
        return "0"
    exponent = int(round(np.log2(x)))
    return fr"$2^{{{exponent}}}$"

import numpy as np
ax.xaxis.set_major_formatter(FuncFormatter(power_of_two))

# Labels and formatting
plt.xlabel('State Size', fontsize=22)
plt.ylabel('Validation Accuracy', fontsize=22)
plt.title('Sequence length: 512 / kv pairs: 64', fontsize=24, weight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=20, frameon=True, loc='lower right')
plt.xticks([64*64,128*64,256*64,512*64],['16 kB','32 kB','64 kB','128 kB'], fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()

# Save and show
plt.savefig("results_pow2_ticks.png", dpi=300)
plt.show()

import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np

# =================================================================
# 1. Configuration
# =================================================================
METRIC_KEY = "valid/accuracy"
PROJECT_PATH = "d-bonnet/Palimpsa_MQAR_seeds_very_small_forgetting-2"

# =================================================================
# 2. Fetch Data from WandB
# =================================================================
print(f"Fetching runs from {PROJECT_PATH}...")
api = wandb.Api()
runs = api.runs(PROJECT_PATH)

summary_list, name_list = [], []

for run in runs: 
    summary = run.summary._json_dict
    
    # Check if the key exists
    if METRIC_KEY in summary:
        summary_list.append(summary)
        name_list.append(run.name)

# Safety Check
if not summary_list:
    print(f"\n[ERROR] No runs found containing the metric: '{METRIC_KEY}'")
    sys.exit(1)

df = pd.DataFrame(summary_list)
df["run_name"] = name_list
df["run_name"] = df["run_name"].astype(str)

print(f"Total runs found with metric: {len(df)}")

# =================================================================
# 3. Parse Metadata (FIXED REGEX)
# =================================================================
# We use .+? (lazy match) for LR to capture scientific notation (e.g. 1e-04) correctly
pattern = r"(?P<model>[^-]+)-seqlen(?P<seq_len>\d+)-dmodel(?P<d_model>\d+)-lr(?P<lr>.+?)-seed(?P<seed>\d+)-.*"

metadata = df["run_name"].str.extract(pattern)
df = pd.concat([df, metadata], axis=1)

# Debug: Check if parsing worked this time
if df["model"].isna().all():
    print("\n[CRITICAL ERROR] Regex parsing failed for ALL runs.")
    print("Sample run name:", df["run_name"].iloc[0])
    sys.exit(1)

# Drop rows where parsing failed
df = df.dropna(subset=["model"])

# Convert types
df["seq_len"] = pd.to_numeric(df["seq_len"])
df["lr"] = pd.to_numeric(df["lr"])
df["seed"] = pd.to_numeric(df["seed"])
df["accuracy"] = df[METRIC_KEY]


df = df[~df["seed"].isin([5, 6])]

# =================================================================
# 4. Filter for Best LR (Per Model, Per SeqLen, Per Seed)
# =================================================================
# 1. Sort by accuracy descending (best first)
df = df.sort_values("accuracy", ascending=False)

# 2. Group by seed/model/seq_len and take the top entry (the best LR)
best_runs_df = df.drop_duplicates(subset=["model", "seq_len", "seed"], keep="first")

print(f"Runs after filtering for best LR: {len(best_runs_df)}")
# You expect ~64 here (4 models * 4 lengths * 4 seeds)

# =================================================================
# 5. Plotting
# =================================================================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

ax = sns.lineplot(
    data=best_runs_df,
    x="seq_len",
    y="accuracy",
    hue="model",
    marker="o",
    errorbar="sd",
    linewidth=2.5
)

# Formatting
plt.xscale("log", base=2)
plt.title(f"MQAR Accuracy vs Sequence Length\n(Best LR per seed, shaded area = ±std)", fontsize=14)
plt.ylabel("Validation Accuracy", fontsize=12)
plt.xlabel("Sequence Length (log scale)", fontsize=12)

# Set ticks manually to your specific sequence lengths
if not best_runs_df.empty:
    xticks = sorted(best_runs_df["seq_len"].unique())
    plt.xticks(xticks, xticks, fontsize=10)

plt.ylim(0, 1.05)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig("mqar_results_very_small_forgetting-2.png", dpi=300)
print("Plot saved to mqar_very_small_forgetting-2.png")


# =================================================================
# 5. Plotting (One curve per seed, Color by Model)
# =================================================================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# We use 'units' to draw a separate line for every seed.
# We must set 'estimator=None' to prevent averaging.
ax = sns.lineplot(
    data=best_runs_df,
    x="seq_len",
    y="accuracy",
    hue="model",      # Color stays consistent per model
    units="seed",     # One line per seed
    estimator=None,   # Don't average the seeds
    marker="o",
    alpha=0.6,        # Lower alpha helps see overlapping seed lines
    linewidth=1.5
)

# Formatting
plt.xscale("log", base=2)
plt.title(f"MQAR Accuracy: Individual Seed Trajectories\n(Best LR per seed)", fontsize=14)
plt.ylabel("Validation Accuracy", fontsize=12)
plt.xlabel("Sequence Length (log2 scale)", fontsize=12)

# Set ticks manually
if not best_runs_df.empty:
    xticks = sorted(best_runs_df["seq_len"].unique())
    plt.xticks(xticks, xticks, fontsize=10)

plt.ylim(-0.05, 1.05)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig("mqar_results_per_seed.png", dpi=300)
print("Plot saved to mqar_results_per_seed.png")


# =================================================================
# 6. Bar Plots (512 and 1024)
# =================================================================
# Define your specific order
model_order = ["Palimpsa", "MetaMamba2", "GatedDeltaNet", "NotPalimpsa", "Mamba2"]

lengths_to_plot = [512, 1024]
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, slen in enumerate(lengths_to_plot):
    subset = best_runs_df[best_runs_df["seq_len"] == slen]
    
    if subset.empty:
        continue

    # 1. Main Bar Plot
    sns.barplot(
        data=subset,
        x="model",
        y="accuracy",
        order=model_order,  # Force your specific order
        ax=axes[i],
        palette="viridis",
        errorbar="sd",      # Standard Deviation
        capsize=.05,        # Small caps for the error bars
        edgecolor=".2",
        width=0.5           # Makes the bars thinner (default is 0.8)
    )
    
    # 2. Individual Seed Points
    sns.stripplot(
        data=subset,
        x="model",
        y="accuracy",
        order=model_order,
        ax=axes[i],
        color="black",
        alpha=0.4,
        size=5,
        jitter=True
    )

    axes[i].set_title(f"Sequence Length: {slen}", fontsize=14, fontweight='bold')
    axes[i].set_ylim(0.25, 1.05)
    axes[i].set_ylabel("Validation Accuracy" if i == 0 else "")
    axes[i].set_xlabel("") # Cleaner look
    
    # Rotate x-labels if they overlap
    axes[i].tick_params(axis='x', rotation=15)

# Fix the Title Clipping
plt.suptitle(f"MQAR Performance: small forgetting", fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.88) # Create space for the suptitle

plt.savefig("mqar_bar_comparison_fixed.png", dpi=300, bbox_inches='tight')
print("Fixed bar plots saved to mqar_bar_comparison_fixed.png")

# =================================================================
# 7. Performance at the Lowest Learning Rate (All Configs & Seeds)
# =================================================================

# 1. Identify the lowest LR for each (model, seq_len, seed)
# We sort by LR ascending so the first entry in each group is the minimum LR
lowest_lr_df = df.sort_values("lr", ascending=True).drop_duplicates(
    subset=["model", "seq_len", "seed"], 
    keep="first"
).copy()

plt.figure(figsize=(12, 7))

# 2. Bar plot showing accuracy at the lowest LR
# Using 'hue="model"' and 'x="seq_len"' to see how the lowest LR scales
sns.barplot(
    data=lowest_lr_df,
    x="seq_len",
    y="accuracy",
    hue="model",
    hue_order=model_order,
    errorbar="sd",
    capsize=.05,
    palette="viridis"
)

# 3. Overlay individual seed points to see the spread at this low LR
sns.stripplot(
    data=lowest_lr_df,
    x="seq_len",
    y="accuracy",
    hue="model",
    hue_order=model_order,
    dodge=True,
    color="black",
    alpha=0.3,
    size=4,
    legend=False
)

plt.title("MQAR Accuracy: Forced Lowest Learning Rate\n(All Seeds included)", fontsize=14, fontweight='bold')
plt.ylim(0., 1.05) # Your requested y-axis cut
plt.ylabel("Validation Accuracy")
plt.xlabel("Sequence Length")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig("mqar_lowest_lr_performance.png", dpi=300)
print(f"Lowest LR plot saved. Total runs plotted: {len(lowest_lr_df)}")


# =================================================================
# 7. Performance at the SECOND Lowest Learning Rate
# =================================================================

# 1. Sort by LR ascending
df_sorted = df.sort_values(["model", "seq_len", "seed", "lr"], ascending=True)

# 2. Group and take the Nth entry (0 is 1st, 1 is 2nd)
second_lowest_df = df_sorted.groupby(["model", "seq_len", "seed"]).nth(1)

# Note: .nth() might return fewer rows if some configs only had 1 LR run.
print(f"Runs using the 2nd lowest LR: {len(second_lowest_df)}")

# 3. Plotting (Same logic as before)
plt.figure(figsize=(12, 7))
sns.barplot(
    data=second_lowest_df.reset_index(), # reset_index because .nth() makes keys the index
    x="seq_len",
    y="accuracy",
    hue="model",
    hue_order=model_order,
    errorbar="sd",
    palette="viridis"
)

plt.ylim(0., 1.05)
plt.title("MQAR Accuracy: Forced SECOND Lowest Learning Rate", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("mqar_second_lowest_lr.png", dpi=300)
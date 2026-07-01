import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import numpy as np

# Set modern styling
sns.set_theme(style="whitegrid", context="notebook", palette="husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'sans-serif'

# === CONFIG ===
root_dir = "../../../M2M-MT-files/Youtube_Download_2_Transcription_Whisper"
output_root = "../../../M2M-MT-files/Youtube_Download_2_Whisper_EDA_reports"
train_size = 2000
test_size = 500
output_dir = os.path.join(output_root, "splits")
os.makedirs(output_dir, exist_ok=True)

# === STEP 1: COLLECT SENTENCES & WORD COUNTS ===
data = []

for subdir, _, files in os.walk(root_dir):
    folder_name = os.path.basename(subdir)
    target_file = f"{folder_name}_sentences_clean.txt"
    target_path = os.path.join(subdir, target_file)

    if target_file in files:
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]

            for idx, sentence in enumerate(sentences, start=1):
                word_count = len(sentence.split())
                data.append({
                    "folder_name": folder_name,
                    "Sentence ID": idx,
                    "Sentence": sentence,
                    "Word Count": word_count
                })

        except Exception as e:
            print(f"❌ Error reading {target_path}: {repr(e)}")

df = pd.DataFrame(data)
print(f"✅ Loaded {len(df)} total sentences")

if df.empty:
    raise ValueError("No data found — check folder paths or file naming")

# === STEP 2: BIN WORD COUNTS FOR STRATIFICATION ===
df["length_bin"] = pd.cut(
    df["Word Count"],
    bins=[0,5,10,15,20,30,50,100,200,float("inf")],
    labels=False,
    include_lowest=True
)

# Optional: check bin distribution
print("\nSentence counts per bin:")
print(df["length_bin"].value_counts().sort_index())

# === STEP 3: SPLIT TRAIN / REST WITH STRATIFY FALLBACK ===
try:
    train_df, rest_df = train_test_split(
        df,
        train_size=min(train_size, len(df)),
        stratify=df["length_bin"],
        random_state=42
    )
except ValueError:
    warnings.warn("Stratified split failed — using random split for train/rest")
    train_df, rest_df = train_test_split(
        df,
        train_size=min(train_size, len(df)),
        random_state=42
    )

# === STEP 4: SPLIT TEST / DEV WITH STRATIFY FALLBACK ===
if len(rest_df) > test_size:
    try:
        test_df, dev_df = train_test_split(
            rest_df,
            train_size=test_size,
            stratify=rest_df["length_bin"],
            random_state=42
        )
    except ValueError:
        warnings.warn("Stratified split failed — using random split for test/dev")
        test_df, dev_df = train_test_split(
            rest_df,
            train_size=test_size,
            random_state=42
        )
else:
    test_df = rest_df
    dev_df = pd.DataFrame()

print(f"\n✅ Split sizes → Train: {len(train_df)}, Test: {len(test_df)}, Dev: {len(dev_df)}")

# === STEP 5: SAVE CSVs ===
# for name, split_df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
#     if not split_df.empty:
#         split_df[["folder_name", "Sentence ID"]].to_csv(
#             os.path.join(output_dir, f"{name}_sentences.csv"),
#             index=False
#         )
#         print(f"✅ Saved CSV: {name}_sentences.csv")

# === STEP 6: ENHANCED PLOTS WITH CONSISTENT COLORS ===

# Define clean, professional color palette - SAME FOR ALL PLOTS
colors = {
    'train': '#3B82F6',    # Bright blue
    'dev': '#F59E0B',      # Amber
    'test': '#10B981'      # Emerald green
}

# --- PLOT 1: Combined distribution - MATCHING INDIVIDUAL PLOT STYLE ---
fig = plt.figure(figsize=(14, 8))

# Single axis setup matching individual plots
ax1 = fig.add_subplot(111)

# Histogram with area fill - MATCHING INDIVIDUAL PLOT STYLE
for name, (split_df, color) in [('train', (train_df, colors['train'])),
                                 ('dev', (dev_df, colors['dev'])),
                                 ('test', (test_df, colors['test']))]:
    if not split_df.empty:
        ax1.hist(split_df["Word Count"], bins=40, 
                color=color, alpha=0.6, edgecolor='white', linewidth=1.5,
                label=f'{name.capitalize()} (n={len(split_df):,})', density=True)

# KDE overlay for each dataset
from scipy import stats
for name, (split_df, color) in [('train', (train_df, colors['train'])),
                                 ('dev', (dev_df, colors['dev'])),
                                 ('test', (test_df, colors['test']))]:
    if not split_df.empty:
        density = stats.gaussian_kde(split_df["Word Count"])
        xs = np.linspace(split_df["Word Count"].min(), split_df["Word Count"].max(), 200)
        ax1.plot(xs, density(xs), color=color, linewidth=3, alpha=0.8)

# Add vertical lines for means
for name, split_df, color in [('Train', train_df, colors['train']), 
                               ('Dev', dev_df, colors['dev']), 
                               ('Test', test_df, colors['test'])]:
    if not split_df.empty:
        mean_val = split_df["Word Count"].mean()
        ax1.axvline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.7)

# Styling - MATCHING INDIVIDUAL PLOTS
ax1.set_xlabel("Number of Words per Sentence", fontsize=12, fontweight='bold')
ax1.set_ylabel("Density", fontsize=12, fontweight='bold')
# ax1.set_title("Distribution of Sentence Lengths Across All Splits", 
#              fontsize=15, fontweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax1.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
sns.despine(ax=ax1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sentence_length_distributions_all_new.pdf"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved enhanced combined plot: sentence_length_distributions_all_new.pdf")

# --- PLOT 2: Individual split plots ---
splits = {"train": (train_df, colors['train']), 
          "dev": (dev_df, colors['dev']), 
          "test": (test_df, colors['test'])}

for name, (split_df, color) in splits.items():
    if not split_df.empty:
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Main histogram with KDE
        ax1 = fig.add_subplot(gs[0, :])
        
        # Histogram with area fill
        counts, bins, patches = ax1.hist(split_df["Word Count"], bins=40, 
                                         color=color, alpha=0.6, edgecolor='white', linewidth=1.5)
        
        # KDE overlay
        from scipy import stats
        density = stats.gaussian_kde(split_df["Word Count"])
        xs = np.linspace(split_df["Word Count"].min(), split_df["Word Count"].max(), 200)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(xs, density(xs), color=color, linewidth=3, alpha=0.8, label='KDE')
        ax1_twin.fill_between(xs, density(xs), alpha=0.2, color=color)
        ax1_twin.set_ylabel("Density", fontsize=11, fontweight='bold')
        ax1_twin.legend(loc='upper right', fontsize=10)
        
        # Statistics
        mean_val = split_df["Word Count"].mean()
        median_val = split_df["Word Count"].median()
        
        # Add vertical lines for mean and median
        ax1.axvline(mean_val, color='#EF4444', linestyle='--', linewidth=2.5, alpha=0.7, label=f'Mean: {mean_val:.1f}')
        ax1.axvline(median_val, color='#8B5CF6', linestyle='-.', linewidth=2.5, alpha=0.7, label=f'Median: {median_val:.1f}')
        
        ax1.set_xlabel("Number of Words per Sentence", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Frequency", fontsize=12, fontweight='bold')
        ax1.set_title(f"Sentence Length Distribution — {name.capitalize()} Set", 
                     fontsize=15, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax1.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
        sns.despine(ax=ax1)
        
        # Box plot
        ax2 = fig.add_subplot(gs[1, 0])
        bp = ax2.boxplot(split_df["Word Count"], vert=False, widths=0.6,
                         patch_artist=True, 
                         boxprops=dict(facecolor=color, alpha=0.7, edgecolor='black', linewidth=1.5),
                         whiskerprops=dict(color='black', linewidth=1.5),
                         capprops=dict(color='black', linewidth=1.5),
                         medianprops=dict(color='#EF4444', linewidth=2.5),
                         flierprops=dict(marker='o', markerfacecolor=color, markersize=5, alpha=0.5))
        
        ax2.set_xlabel("Number of Words", fontsize=11, fontweight='bold')
        ax2.set_title("Box Plot", fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, axis='x', linestyle="--", alpha=0.4)
        ax2.set_yticks([])
        sns.despine(ax=ax2, left=True)
        
        # Statistics table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Calculate statistics
        q1 = split_df["Word Count"].quantile(0.25)
        q3 = split_df["Word Count"].quantile(0.75)
        iqr = q3 - q1
        min_val = split_df["Word Count"].min()
        max_val = split_df["Word Count"].max()
        std_val = split_df["Word Count"].std()
        
        stats_data = [
            ['Metric', 'Value'],
            ['Count', f'{len(split_df):,}'],
            ['Mean', f'{mean_val:.2f}'],
            ['Median', f'{median_val:.2f}'],
            ['Std Dev', f'{std_val:.2f}'],
            ['Min', f'{min_val}'],
            ['Q1 (25%)', f'{q1:.1f}'],
            ['Q3 (75%)', f'{q3:.1f}'],
            ['Max', f'{max_val}'],
            ['IQR', f'{iqr:.2f}']
        ]
        
        table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.3],
                         bbox=[0.1, 0, 0.8, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            cell = table[(0, i)]
            cell.set_facecolor(color)
            cell.set_text_props(weight='bold', color='white')
            cell.set_edgecolor('white')
            cell.set_linewidth(2)
        
        # Style data rows
        for i in range(1, len(stats_data)):
            for j in range(2):
                cell = table[(i, j)]
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
                cell.set_edgecolor('#dee2e6')
                cell.set_linewidth(1)
                if j == 0:
                    cell.set_text_props(weight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"sentence_length_distribution_{name}_new.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved enhanced {name} plot: {plot_path}")

# --- PLOT 3: Comparative analysis ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Overlapping KDE
ax1 = fig.add_subplot(gs[0, 0])
for name, (split_df, color) in splits.items():
    if not split_df.empty:
        sns.kdeplot(data=split_df["Word Count"], color=color, linewidth=3, 
                   label=f'{name.capitalize()}', 
                   fill=True, alpha=0.3, ax=ax1)

ax1.set_xlabel("Number of Words", fontsize=11, fontweight='bold')
ax1.set_ylabel("Density", fontsize=11, fontweight='bold')
ax1.set_title("Kernel Density Estimation", fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
sns.despine(ax=ax1)

# 2. Cumulative Distribution
ax2 = fig.add_subplot(gs[0, 1])
for name, (split_df, color) in splits.items():
    if not split_df.empty:
        sorted_data = np.sort(split_df["Word Count"])
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, cumulative, color=color, linewidth=2.5, 
                label=f'{name.capitalize()}', alpha=0.8)

ax2.set_xlabel("Number of Words", fontsize=11, fontweight='bold')
ax2.set_ylabel("Cumulative Probability", fontsize=11, fontweight='bold')
ax2.set_title("Cumulative Distribution Function", fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')
sns.despine(ax=ax2)

# 3. Box plot comparison
ax3 = fig.add_subplot(gs[1, 0])
box_data = []
box_labels = []
box_colors = []
for name, (split_df, color) in splits.items():
    if not split_df.empty:
        box_data.append(split_df["Word Count"])
        box_labels.append(name.capitalize())
        box_colors.append(color)

bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6,
                 boxprops=dict(linewidth=1.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5),
                 medianprops=dict(color='#EF4444', linewidth=2.5),
                 flierprops=dict(marker='o', markersize=5, alpha=0.5))

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel("Number of Words", fontsize=11, fontweight='bold')
ax3.set_title("Box Plot Comparison", fontsize=13, fontweight='bold')
ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
sns.despine(ax=ax3)

# 4. Statistical summary bars
ax4 = fig.add_subplot(gs[1, 1])
metrics = ['Mean', 'Median', 'Std Dev']
x = np.arange(len(metrics))
width = 0.25

for idx, (name, (split_df, color)) in enumerate(splits.items()):
    if not split_df.empty:
        values = [
            split_df["Word Count"].mean(),
            split_df["Word Count"].median(),
            split_df["Word Count"].std()
        ]
        offset = (idx - 1) * width
        bars = ax4.bar(x + offset, values, width, label=name.capitalize(), 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax4.set_ylabel("Value", fontsize=11, fontweight='bold')
ax4.set_title("Statistical Metrics Comparison", fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontweight='bold')
ax4.legend(fontsize=10, framealpha=0.95)
ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
sns.despine(ax=ax4)

plt.suptitle("Comprehensive Sentence Length Analysis", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparative_analysis_enhanced.png"), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved comprehensive comparative analysis plot")

print("\n🎉 All enhanced plots generated successfully!")
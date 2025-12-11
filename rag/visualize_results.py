import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

# 1. LOAD DATA
csv_data = """context_length,needle_depth,ground_truth,rag_time,rag_success,rag_tokens,lc_time,lc_success,lc_tokens
4000,0.51,NEBULA-UAPCC3RG-0,0.393,1,1327,0.309,1,3820
4000,0.93,NEBULA-72ZG80D2-1,0.249,1,1359,0.279,1,3788
4000,0.67,NEBULA-INAXY0MP-2,0.208,1,1654,0.252,1,4285
4000,0.6,NEBULA-QETNCMZG-3,0.228,1,1499,0.252,1,4433
4000,0.75,NEBULA-3R81D42Z-4,0.211,1,1554,0.262,1,4054
4000,0.27,NEBULA-0JRUZA0V-5,0.209,1,1504,0.256,1,3916
4000,0.43,NEBULA-4YBYEBHX-6,0.194,1,1513,0.294,1,3952
4000,0.53,NEBULA-JCWVPJ63-7,0.259,1,1344,0.263,1,3924
4000,0.74,NEBULA-W4EKOCQA-8,0.272,0,1510,0.241,1,3949
4000,0.33,NEBULA-I3YWSZVK-9,0.194,1,1527,0.249,1,4164
4000,0.68,NEBULA-Z78VQUUE-10,0.189,1,1505,0.252,1,3968
4000,0.74,NEBULA-WG1S0375-11,0.175,0,1414,0.268,1,4013
4000,0.25,NEBULA-TQ6W1U0N-12,0.133,0,1537,0.262,1,3964
4000,0.34,NEBULA-ZRSKQ7R9-13,0.189,1,1424,0.245,1,3754
4000,0.32,NEBULA-14AB7UCK-14,0.193,1,1670,0.263,1,4461
14000,0.8,NEBULA-FYWP9FBB-0,0.151,0,1766,0.987,1,15049
14000,0.61,NEBULA-CAQG92C4-1,0.147,0,1578,0.929,1,13960
14000,0.04,NEBULA-8T3N292I-2,0.147,0,1449,0.909,1,13399
14000,0.42,NEBULA-XEOKHQOZ-3,0.291,0,1535,0.957,1,14503
14000,0.08,NEBULA-UCJ433P4-4,0.298,0,1533,0.943,1,14041
14000,0.8,NEBULA-YKMNF9DX-5,0.210,1,1427,0.936,1,13442
14000,0.43,NEBULA-2UX4AMNR-6,0.232,0,1524,0.989,1,13466
14000,0.18,NEBULA-7YK7YLK8-7,0.227,0,1510,1.023,1,13627
14000,0.52,NEBULA-T697EOFH-8,0.336,1,1398,0.842,1,12775
14000,0.36,NEBULA-W11SVYEB-9,0.209,1,1403,0.841,1,13070
14000,0.84,NEBULA-B1WN7FI4-10,0.206,0,1528,0.897,1,13489
14000,0.73,NEBULA-NWPTMDZV-11,0.220,1,1460,0.930,1,13847
14000,0.52,NEBULA-NH5WAVVP-12,0.271,1,1514,0.985,1,13499
14000,0.76,NEBULA-NV7ZKDDS-13,0.361,1,1483,1.059,1,14605
14000,0.91,NEBULA-66ODNO3A-14,0.228,0,1287,1.033,1,13813
24000,1.0,NEBULA-JVMJ8176-0,0.207,0,1524,1.988,1,23513
24000,0.17,NEBULA-HYCNHMD6-1,0.378,1,1480,2.065,1,24071
24000,0.84,NEBULA-I7PLRV3I-2,0.336,0,1499,2.060,1,23673
24000,0.53,NEBULA-M4WSGYFJ-3,0.266,0,1168,2.237,1,25064
24000,0.24,NEBULA-10FQDU7T-4,0.496,0,1462,2.009,1,23365
24000,0.2,NEBULA-6H5SZ5SJ-5,0.260,0,1158,2.215,1,24651
24000,0.35,NEBULA-BAL41VSG-6,0.344,0,1501,1.996,1,23188
24000,0.52,NEBULA-941OTZIK-7,0.491,0,1527,1.984,1,23106
24000,0.78,NEBULA-FT28B48R-8,0.281,0,1433,1.853,1,22074
24000,0.25,NEBULA-38086GK3-9,0.317,0,1129,2.005,1,23078
24000,0.1,NEBULA-BJWC84LC-10,0.366,0,1403,2.006,1,23289
24000,0.75,NEBULA-N4ZFKW8Y-11,0.332,0,1530,2.218,1,24611
24000,0.83,NEBULA-WXROK028-12,0.251,0,1513,2.046,1,23605
24000,0.35,NEBULA-DSMPGMLS-13,0.376,1,1560,2.071,1,24029
24000,0.58,NEBULA-84QRM3O2-14,0.328,0,1435,2.022,1,23440"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. PREPARE AGGREGATED TABLE
summary = df.groupby('context_length').agg({
    'rag_success': 'mean',
    'lc_success': 'mean',
    'rag_time': 'mean',
    'lc_time': 'mean',
    'rag_tokens': 'mean',
    'lc_tokens': 'mean'
}).reset_index()

# Convert success to percentage for plotting
summary['rag_success_pct'] = summary['rag_success'] * 100
summary['lc_success_pct'] = summary['lc_success'] * 100

# 3. PLOTTING
# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Experimental Results: RAG vs. Long Context LLM (Qwen 3 0.6B)', fontsize=16, fontweight='bold', y=1.05)

# Color Palette
rag_color = '#E63946'  # Reddish
lc_color = '#1D3557'   # Navy Blue
width = 0.35
x = np.arange(len(summary['context_length']))

# --- CHART 1: ACCURACY (Bar Chart) ---
ax1 = axes[0]
rects1 = ax1.bar(x - width/2, summary['rag_success_pct'], width, label='RAG', color=rag_color, alpha=0.9)
rects2 = ax1.bar(x + width/2, summary['lc_success_pct'], width, label='Long Context', color=lc_color, alpha=0.9)

ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy vs Context Length', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(summary['context_length'])
ax1.set_ylim(0, 115) # Room for labels
ax1.legend()

# Add value labels
for rect in rects1 + rects2:
    height = rect.get_height()
    ax1.annotate(f'{height:.0f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

# --- CHART 2: LATENCY (Line Chart) ---
ax2 = axes[1]
ax2.plot(summary['context_length'], summary['rag_time'], marker='o', label='RAG', color=rag_color, linewidth=3, markersize=8)
ax2.plot(summary['context_length'], summary['lc_time'], marker='s', label='Long Context', color=lc_color, linewidth=3, markersize=8)

ax2.set_ylabel('Latency (Seconds)', fontsize=12)
ax2.set_xlabel('Context Length (Tokens)', fontsize=12)
ax2.set_title('Processing Time vs Context Length', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# --- CHART 3: COMPUTATIONAL COST (Bar Chart) ---
ax3 = axes[2]
rects1 = ax3.bar(x - width/2, summary['rag_tokens'], width, label='RAG', color=rag_color, alpha=0.9)
rects2 = ax3.bar(x + width/2, summary['lc_tokens'], width, label='Long Context', color=lc_color, alpha=0.9)

ax3.set_ylabel('Tokens Processed (Input)', fontsize=12)
ax3.set_title('Compute Cost (Tokens) vs Context', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(summary['context_length'])
ax3.legend()

# Add value labels for tokens (formatted with 'k')
for rect in rects1 + rects2:
    height = rect.get_height()
    ax3.annotate(f'{int(height):,}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('experiment_results_dashboard.png', dpi=300, bbox_inches='tight')

# 4. PRINT FORMATTED TABLE FOR REPORT
print("\n" + "="*80)
print("SUMMARY TABLE FOR REPORT")
print("="*80)
print(f"{'Context':<10} | {'Method':<15} | {'Accuracy':<10} | {'Time (s)':<10} | {'Tokens (Cost)':<15}")
print("-" * 80)

for index, row in summary.iterrows():
    ctx = str(int(row['context_length']))
    # RAG Row
    print(f"{ctx:<10} | {'RAG':<15} | {row['rag_success_pct']:<5.1f}%{'':<4} | {row['rag_time']:<8.3f}   | {int(row['rag_tokens']):<15}")
    # LC Row
    print(f"{'':<10} | {'Long Context':<15} | {row['lc_success_pct']:<5.1f}%{'':<4} | {row['lc_time']:<8.3f}   | {int(row['lc_tokens']):<15}")
    if index < len(summary) - 1:
        print("-" * 80)
print("="*80)
print("\n[INFO] Dashboard image saved as 'experiment_results_dashboard.png'")
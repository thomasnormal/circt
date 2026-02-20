#!/usr/bin/env python3
"""Generate charts for the CIRCT fork blog post."""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']

# Read checkpoint data
data = []
with open('blog_data/checkpoints.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'commit': int(row['commit_num']),
            'date': datetime.strptime(row['date'], '%Y-%m-%d'),
            'test_files': int(row['test_files']),
            'src_files': int(row['src_files']),
            'util_files': int(row['util_files']),
            'insertions': int(row['insertions']),
            'deletions': int(row['deletions']),
            'files_changed': int(row['files_changed']),
        })

data.sort(key=lambda x: x['commit'])

dates = [d['date'] for d in data]
commits = [d['commit'] for d in data]
test_files = [d['test_files'] for d in data]
insertions = [d['insertions'] for d in data]
net_lines = [d['insertions'] - d['deletions'] for d in data]
util_files = [d['util_files'] for d in data]
src_files = [d['src_files'] for d in data]

# Color palette
BLUE = '#2563eb'
BLUE_LIGHT = '#93c5fd'
RED = '#dc2626'
GREEN = '#16a34a'
GREEN_LIGHT = '#86efac'
ORANGE = '#ea580c'
PURPLE = '#9333ea'
PURPLE_LIGHT = '#c4b5fd'
GRAY = '#6b7280'
GRAY_LIGHT = '#d1d5db'
SLATE = '#334155'
CYAN = '#0891b2'
AMBER = '#d97706'

# Weekly commits data
weekly_dates = [
    datetime(2026, 1, 9),   # W02
    datetime(2026, 1, 13),  # W03
    datetime(2026, 1, 20),  # W04
    datetime(2026, 1, 27),  # W05
    datetime(2026, 2, 3),   # W06
    datetime(2026, 2, 10),  # W07
    datetime(2026, 2, 17),  # W08
]
weekly_counts = [176, 336, 325, 354, 573, 867, 337]

# ── Chart 1: Dense Timeline (LOC + commits + milestones) ──
fig, ax1 = plt.subplots(figsize=(12, 6.5))

# Stacked area: test code vs source code vs utils (approximate from file counts * avg lines)
# Use actual insertions as the main series, with test_files as overlay
ax1.fill_between(dates, net_lines, alpha=0.12, color=BLUE)
ax1.plot(dates, net_lines, color=BLUE, linewidth=2.5, label='Net lines added', zorder=5)

# Secondary axis for test file count
ax2 = ax1.twinx()
ax2.fill_between(dates, test_files, alpha=0.08, color=PURPLE)
ax2.plot(dates, test_files, color=PURPLE, linewidth=2, linestyle='--', label='Test files', zorder=4)
ax2.set_ylabel('Test files', fontsize=10, color=PURPLE)
ax2.tick_params(axis='y', colors=PURPLE)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))

# Line overlay for weekly commits
weekly_midpoints = [wd + timedelta(days=3) for wd in weekly_dates]
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('axes', 1.08))
ax3.plot(weekly_midpoints, weekly_counts, color=ORANGE, linewidth=1.8,
         marker='o', markersize=5, zorder=6, label='Commits/week')
ax3.set_ylabel('Commits / week', fontsize=10, color=ORANGE)
ax3.tick_params(axis='y', colors=ORANGE)
ax3.set_ylim(0, max(weekly_counts) * 1.5)

# Annotated milestones with colored phase bands
phases = [
    (datetime(2026, 1, 9), datetime(2026, 1, 12), '#dbeafe', 'Foundation'),
    (datetime(2026, 1, 13), datetime(2026, 1, 19), '#fef3c7', 'UVM parity'),
    (datetime(2026, 1, 20), datetime(2026, 2, 2), '#dcfce7', 'SV completeness'),
    (datetime(2026, 2, 3), datetime(2026, 2, 9), '#fce7f3', 'Formal + mutation'),
    (datetime(2026, 2, 10), datetime(2026, 2, 20), '#f0f9ff', 'VPI, cocotb, hardening'),
]
for start, end, color, label in phases:
    ax1.axvspan(start, end, alpha=0.25, color=color, zorder=0)
    mid = start + (end - start) / 2
    ax1.text(mid, -35000, label, ha='center', va='top',
             fontsize=7.5, color=SLATE, fontstyle='italic')

# Detailed milestone annotations
milestones = [
    (datetime(2026, 1, 9), 'circt-sim driver\n4-state types\nCoverage dialect', 55),
    (datetime(2026, 1, 13), 'UVM mailbox\nrandomize()\nAVIP baseline', 50),
    (datetime(2026, 1, 22), 'force/release\nUVM sequencer\nsv-tests', 45),
    (datetime(2026, 1, 26), 'OpenTitan\ngpio_reg_top\nend-to-end', 60),
    (datetime(2026, 2, 6), 'BMC k-induction\nJIT compilation\n800+ formal tests', 50),
    (datetime(2026, 2, 9), 'circt-mut\nMCY integration\nmutation coverage', 40),
    (datetime(2026, 2, 14), 'Mutation matrix\nnative backend\nquality gates', 55),
    (datetime(2026, 2, 17), 'VPI runtime\ncocotb integration\narcilator', 45),
]
for mdate, mlabel, offset in milestones:
    closest = min(data, key=lambda d: abs((d['date'] - mdate).days))
    y = closest['insertions'] - closest['deletions']
    ax1.annotate(mlabel, xy=(mdate, y), xytext=(0, offset),
                textcoords='offset points', fontsize=7,
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.7),
                color=SLATE, linespacing=1.3,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GRAY_LIGHT, alpha=0.9))

ax1.set_xlabel('', fontsize=10)
ax1.set_ylabel('Net lines of code (vs upstream)', fontsize=10, color=BLUE)
ax1.tick_params(axis='y', colors=BLUE)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
ax1.set_ylim(-45000, max(net_lines) * 1.2)
ax2.set_ylim(0, max(test_files) * 1.35)

# Legend
from matplotlib.lines import Line2D
blue_line = mpatches.Patch(color=BLUE, alpha=0.4, label='Net lines added')
purple_line = mpatches.Patch(color=PURPLE, alpha=0.3, label='Test files')
orange_line = Line2D([0], [0], color=ORANGE, linewidth=1.8, marker='o', markersize=4, label='Commits/week')
ax1.legend(handles=[blue_line, purple_line, orange_line], loc='upper left', fontsize=8,
           framealpha=0.9, edgecolor=GRAY_LIGHT)

ax1.set_title('Development Timeline: 2,968 commits over 43 days', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('blog_data/chart_timeline.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_timeline.png', format='png', dpi=150)
plt.close()

# ── Chart 2: Test Files Over Time ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(dates, test_files, alpha=0.15, color=PURPLE)
ax.plot(dates, test_files, color=PURPLE, linewidth=2.5, label='Test files')
ax.plot(dates, src_files, color=ORANGE, linewidth=2, label='Source files (lib/include/tools)')
ax.plot(dates, [u * 10 for u in util_files], color=GREEN, linewidth=1.5,
        linestyle='--', label='Utility scripts (x10)')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('File count', fontsize=11)
ax.set_title('Test and Source File Growth', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.savefig('blog_data/chart_test_growth.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_test_growth.png', format='png', dpi=150)
plt.close()

# ── Chart 3: AI Attribution Pie ──
fig, ax = plt.subplots(figsize=(6, 6))
sizes = [1185, 417, 1, 1365]
labels = ['Claude Opus 4.5\n(1,185)', 'Claude Opus 4.6\n(417)', 'Claude Sonnet 4.5\n(1)', 'Codex\n(1,365)']
colors = ['#8b5cf6', '#6d28d9', '#a78bfa', '#22c55e']
explode = (0.02, 0.02, 0.02, 0.05)
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.1f%%', startangle=140,
                                   textprops={'fontsize': 10})
for t in autotexts:
    t.set_fontsize(10)
    t.set_fontweight('bold')
ax.set_title('AI Co-Author Attribution\n(2,968 commits)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('blog_data/chart_ai_attribution.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_ai_attribution.png', format='png', dpi=150)
plt.close()

# ── Chart 4: Work area breakdown (fully expanded) ──
areas = {
    'Formal verification': 652,
    'Verilog frontend': 461,
    'Mutation testing': 372,
    'Simulation engine': 367,
    'Docs & iteration logs': 521,
    'Dialects & IR passes': 105,
    'Testing infra': 75,
    'Build & infra': 75,
    'Tools (LSP, reduce...)': 60,
    'SVA / LTL': 41,
    'UVM runtime': 36,
    'Bug fixes (untagged)': 79,
    'Feature additions': 46,
    'Merges & misc': 78,
}
fig, ax = plt.subplots(figsize=(10, 6))
sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
names = [a[0] for a in sorted_areas]
vals = [a[1] for a in sorted_areas]
# Color by category type
cmap = {
    'Formal verification': GREEN,
    'Verilog frontend': ORANGE,
    'Mutation testing': RED,
    'Simulation engine': BLUE,
    'Docs & iteration logs': GRAY,
    'Dialects & IR passes': CYAN,
    'Testing infra': PURPLE,
    'Build & infra': AMBER,
    'Tools (LSP, reduce...)': '#64748b',
    'SVA / LTL': '#0d9488',
    'UVM runtime': '#7c3aed',
    'Bug fixes (untagged)': '#f43f5e',
    'Feature additions': '#3b82f6',
    'Merges & misc': '#94a3b8',
}
bar_colors = [cmap.get(n, GRAY) for n in names]
bars = ax.barh(names[::-1], vals[::-1], color=[cmap.get(n, GRAY) for n in names[::-1]],
               alpha=0.85, edgecolor='white', height=0.7)
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2.,
            str(val), va='center', fontsize=9, fontweight='bold', color=SLATE)
ax.set_xlabel('Commits', fontsize=11)
ax.set_title('All 2,968 Commits by Category', fontsize=13, fontweight='bold')
ax.set_xlim(0, max(vals) * 1.15)
plt.tight_layout()
plt.savefig('blog_data/chart_work_areas.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_work_areas.png', format='png', dpi=150)
plt.close()

# ── Chart 5: AVIP Coverage Comparison (circt-sim vs Xcelium) ──
protocols = ['APB', 'AHB', 'AXI4', 'I2S', 'I3C', 'JTAG', 'SPI']
# Xcelium coverage (avg of cov_1 and cov_2)
xcelium_cov = [25.4, 85.7, 36.6, 45.6, 35.2, 47.9, 19.1]
# circt-sim coverage (avg of cov_1 and cov_2, 0 for compile failures)
circt_cov = [54.9, 50.3, 0, 36.8, 35.7, 0, 38.2]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(protocols))
width = 0.35
bars1 = ax.bar(x - width/2, xcelium_cov, width, label='Xcelium', color='#f59e0b', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, circt_cov, width, label='circt-sim', color=BLUE, alpha=0.85, edgecolor='white')

# Mark compile failures
for i, (xc, cc) in enumerate(zip(xcelium_cov, circt_cov)):
    if cc == 0:
        ax.text(x[i] + width/2, 2, 'COMPILE\nFAIL', ha='center', va='bottom',
                fontsize=7, color=RED, fontweight='bold')

# Value labels
for bar in bars1:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Coverage %', fontsize=11)
ax.set_title('AVIP Protocol Coverage: circt-sim vs Xcelium', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(protocols, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig('blog_data/chart_avip_comparison.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_avip_comparison.png', format='png', dpi=150)
plt.close()

# ── Chart 6: Compile and Simulation Speed Comparison ──
# Xcelium compile: ~1s per AVIP, sim: sub-microsecond
# circt-sim compile: 25-35s, sim: seconds to minutes
protos_speed = ['APB', 'AHB', 'I2S', 'I3C', 'SPI']
xcelium_compile = [1, 1, 1, 1, 1]
circt_compile = [25, 25, 35, 28, 30]
# Simulation wall time (seconds, approximate from timestamps)
xcelium_sim = [0.001, 0.01, 0.04, 0.004, 0.002]
circt_sim_time = [45, 35, 60, 241, 40]

fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.arange(len(protos_speed))
width = 0.35

# Compile time
ax_c.bar(x - width/2, xcelium_compile, width, label='Xcelium', color='#f59e0b', alpha=0.85)
ax_c.bar(x + width/2, circt_compile, width, label='circt-sim', color=BLUE, alpha=0.85)
ax_c.set_ylabel('Seconds', fontsize=10)
ax_c.set_title('Compile Time', fontsize=12, fontweight='bold')
ax_c.set_xticks(x)
ax_c.set_xticklabels(protos_speed, fontsize=10)
ax_c.legend(fontsize=9)

# Simulation time (log scale)
ax_s.bar(x - width/2, xcelium_sim, width, label='Xcelium', color='#f59e0b', alpha=0.85)
ax_s.bar(x + width/2, circt_sim_time, width, label='circt-sim', color=BLUE, alpha=0.85)
ax_s.set_ylabel('Seconds (log scale)', fontsize=10)
ax_s.set_title('Simulation Wall Time', fontsize=12, fontweight='bold')
ax_s.set_xticks(x)
ax_s.set_xticklabels(protos_speed, fontsize=10)
ax_s.set_yscale('log')
ax_s.legend(fontsize=9)
# Add ratio annotations
for i, (xs, cs) in enumerate(zip(xcelium_sim, circt_sim_time)):
    ratio = cs / xs
    ax_s.text(x[i] + width/2, cs * 1.3, f'{ratio:.0f}x', ha='center', va='bottom',
              fontsize=7.5, color=RED, fontweight='bold')

plt.suptitle('Performance: circt-sim vs Xcelium', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('blog_data/chart_speed_comparison.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_speed_comparison.png', format='png', dpi=150)
plt.close()

print("All charts generated in blog_data/")

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

# (commits/week overlay removed for cleaner chart)

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
blue_line = mpatches.Patch(color=BLUE, alpha=0.4, label='Net lines added')
purple_line = mpatches.Patch(color=PURPLE, alpha=0.3, label='Test files')
ax1.legend(handles=[blue_line, purple_line], loc='upper left', fontsize=8,
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

# ── Chart 5: AVIP Coverage (circt-sim only) ──
protocols = ['APB', 'AHB', 'AXI4', 'I2S', 'I3C', 'JTAG', 'SPI']
# circt-sim coverage (avg of cov_1 and cov_2, 0 for compile failures)
circt_cov = [54.9, 50.3, 0, 36.8, 35.7, 0, 38.2]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(protocols))
width = 0.5
bars = ax.bar(x, circt_cov, width, label='circt-sim', color=BLUE, alpha=0.85, edgecolor='white')

# Mark compile failures
for i, cc in enumerate(circt_cov):
    if cc == 0:
        ax.text(x[i], 2, 'COMPILE\nFAIL', ha='center', va='bottom',
                fontsize=7, color=RED, fontweight='bold')

# Value labels
for bar in bars:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Coverage %', fontsize=11)
ax.set_title('AVIP Protocol Coverage: circt-sim', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(protocols, fontsize=11)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig('blog_data/chart_avip_comparison.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_avip_comparison.png', format='png', dpi=150)
plt.close()

# ── Chart 6: Compile and Simulation Speed (circt-sim interpret vs compile) ──
protos_speed = ['APB', 'AHB', 'I2S', 'I3C', 'SPI']
circt_interpret_compile = [25, 25, 35, 28, 30]
circt_jit_compile = [44, 29, 72, 32, 52]  # JIT compile time (includes LLVM codegen)
# Simulation wall time (seconds)
circt_interpret_sim = [45, 35, 60, 241, 40]
# JIT sim: None = crashed/timeout; I2S=28s (2.1x faster), SPI=54s
circt_jit_sim = [None, None, 28, None, 54]

fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(13, 4.5))

x = np.arange(len(protos_speed))
width = 0.3

# Compile time (2 bars)
ax_c.bar(x - width/2, circt_interpret_compile, width, label='circt interpret', color=BLUE, alpha=0.85)
ax_c.bar(x + width/2, circt_jit_compile, width, label='circt compile', color='#10b981', alpha=0.85)
ax_c.set_ylabel('Seconds', fontsize=10)
ax_c.set_title('Compile Time', fontsize=12, fontweight='bold')
ax_c.set_xticks(x)
ax_c.set_xticklabels(protos_speed, fontsize=10)
ax_c.legend(fontsize=8)

# Simulation time (log scale, 2 bars — JIT bars only where they ran)
ax_s.bar(x - width/2, circt_interpret_sim, width, label='circt interpret', color=BLUE, alpha=0.85)
# Plot JIT bars only where data exists
jit_x = [i for i, v in enumerate(circt_jit_sim) if v is not None]
jit_v = [v for v in circt_jit_sim if v is not None]
ax_s.bar([x[i] + width/2 for i in jit_x], jit_v, width, label='circt compile', color='#10b981', alpha=0.85)
# Mark crashed protocols with X
for i, v in enumerate(circt_jit_sim):
    if v is None:
        ax_s.text(x[i] + width/2, circt_interpret_sim[i] * 0.5, 'crash', ha='center', va='center',
                  fontsize=7, color='#6b7280', fontstyle='italic')
ax_s.set_ylabel('Seconds (log scale)', fontsize=10)
ax_s.set_title('Simulation Wall Time', fontsize=12, fontweight='bold')
ax_s.set_xticks(x)
ax_s.set_xticklabels(protos_speed, fontsize=10)
ax_s.set_yscale('log')
ax_s.legend(fontsize=8)
# Add JIT speedup annotations where applicable
for i in jit_x:
    speedup = circt_interpret_sim[i] / circt_jit_sim[i]
    ax_s.text(x[i] + width/2, circt_jit_sim[i] * 0.5, f'{speedup:.1f}x\nfaster', ha='center', va='top',
              fontsize=6.5, color='#047857', fontweight='bold')

plt.suptitle('Performance: circt-sim Interpret vs Compile Mode', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('blog_data/chart_speed_comparison.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_speed_comparison.png', format='png', dpi=150)
plt.close()

# ── Chart 7: sv-tests Comparison ──
fig, ax = plt.subplots(figsize=(9, 5))
tools = ['Slang\n(circt parser)', 'circt', 'Verilator', 'Icarus\nVerilog']
passes = [1610, 1622, 1527, 1165]
totals = [1610, 1622, 1614, 1614]
rates = [p/t*100 for p, t in zip(passes, totals)]
colors_sv = ['#8b5cf6', BLUE, '#22c55e', '#6b7280']

bars = ax.bar(tools, rates, color=colors_sv, alpha=0.85, edgecolor='white', width=0.6)
for bar, p, t, r in zip(bars, passes, totals, rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f'{r:.1f}%\n({p}/{t})', ha='center', va='bottom', fontsize=9, fontweight='bold',
            color=SLATE, linespacing=1.4)

ax.set_ylabel('Pass rate (%)', fontsize=11)
ax.set_title('sv-tests Compliance (IEEE 1800-2017)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 115)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('blog_data/chart_sv_tests.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_sv_tests.png', format='png', dpi=150)
plt.close()

# ── Chart 8: sv-tests Progress Over Time ──
fig, ax = plt.subplots(figsize=(10, 5))

# Historical measurement data from fork checkpoints
progress_commits = [0, 500, 800, 1250, 1500, 2000, 2500, 2968]
progress_dates = [
    datetime(2026, 1, 9),   # Fork start (upstream baseline)
    datetime(2026, 1, 17),  # Commit 500
    datetime(2026, 1, 24),  # Commit 800
    datetime(2026, 2, 4),   # Commit 1250
    datetime(2026, 2, 8),   # Commit 1500
    datetime(2026, 2, 10),  # Commit 2000
    datetime(2026, 2, 14),  # Commit 2500
    datetime(2026, 2, 20),  # HEAD
]
progress_pass = [1211, 1247, 1467, 1592, 1605, 1605, 1605, 1622]
progress_total = 1622
progress_pct = [p / progress_total * 100 for p in progress_pass]

# Main line
ax.plot(progress_dates, progress_pct, color=BLUE, linewidth=2.5, marker='o',
        markersize=8, markerfacecolor='white', markeredgecolor=BLUE, markeredgewidth=2, zorder=5)
ax.fill_between(progress_dates, progress_pct, alpha=0.08, color=BLUE)

# Reference lines for other tools (from sv-tests dashboard)
slang_pct = 1610 / 1610 * 100  # 100%
verilator_pct = 1527 / 1614 * 100  # 94.6%
icarus_pct = 1165 / 1614 * 100  # 72.1%

ax.axhline(y=verilator_pct, color='#22c55e', linewidth=1.2, linestyle=':', alpha=0.7, zorder=2)
ax.text(progress_dates[0] - timedelta(days=0.5), verilator_pct, f'Verilator ({verilator_pct:.1f}%)',
        fontsize=8, color='#22c55e', ha='right', va='center', fontweight='bold')

ax.axhline(y=icarus_pct, color='#6b7280', linewidth=1.2, linestyle=':', alpha=0.7, zorder=2)
ax.text(progress_dates[0] - timedelta(days=0.5), icarus_pct, f'Icarus ({icarus_pct:.1f}%)',
        fontsize=8, color='#6b7280', ha='right', va='center', fontweight='bold')

ax.axhline(y=slang_pct, color='#8b5cf6', linewidth=1.2, linestyle=':', alpha=0.5, zorder=2)
ax.text(progress_dates[-1] + timedelta(days=0.5), slang_pct + 1.5, 'Slang (100%)',
        fontsize=8, color='#8b5cf6', ha='left', va='bottom', fontweight='bold')

# Annotate each point
annotations = [
    (0, f'Upstream\nbaseline\n(1,211 / {progress_pct[0]:.0f}%)', -30),
    (1, f'332 pass\n({progress_pct[1]:.0f}%)', -35),
    (2, f'1,605 pass\n({progress_pct[2]:.1f}%)', 30),
    (3, '', 0),  # Skip—same as previous
    (4, '', 0),  # Skip—same as previous
    (5, f'1,622 pass\n(100%)', -40),
]
for idx, (i, label, offset) in enumerate(annotations):
    if not label:
        continue
    ax.annotate(label, xy=(progress_dates[i], progress_pct[i]),
                xytext=(0, offset), textcoords='offset points',
                fontsize=8.5, ha='center', va='bottom' if offset > 0 else 'top',
                color=SLATE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=GRAY_LIGHT, alpha=0.9))

# Commit 1000 note (build failed)
ax.annotate('Commit 1000\n(build failed)', xy=(datetime(2026, 1, 30), 45),
            fontsize=7.5, ha='center', va='center', color=GRAY, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7',
                      edgecolor='#fbbf24', alpha=0.8))

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('sv-tests pass rate (%)', fontsize=11)
ax.set_title('sv-tests Progress: 75% → 100% Over 43 Days', fontsize=13, fontweight='bold')
ax.set_ylim(-5, 115)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add commit count as secondary x-axis labels
for d, c in zip(progress_dates, progress_commits):
    ax.text(d, -12, f'#{c}', ha='center', fontsize=7, color=GRAY)

plt.tight_layout()
plt.savefig('blog_data/chart_sv_tests_progress.svg', format='svg', dpi=150)
plt.savefig('blog_data/chart_sv_tests_progress.png', format='png', dpi=150)
plt.close()

print("All charts generated in blog_data/")

#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks (ops/sec in millions)
allocators = ['glibc', 'tcmalloc', 'Fancy\n(64MB)', 'jemalloc']
ops_per_sec = [7.6, 10.8, 49.8, 75.0]  # Million ops/sec
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

# Plot 1: Allocator Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars1 = ax1.bar(allocators, ops_per_sec, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Operations per Second (Millions)', fontsize=12)
ax1.set_title('Memory Allocator Performance Comparison\n(128 threads, 250K ops/thread)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 85)

# Add value labels on bars
for bar, val in zip(bars1, ops_per_sec):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add speedup annotations
ax1.annotate('6.5x faster\nthan glibc', xy=(2, 49.8), xytext=(2.5, 30),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

ax1.grid(axis='y', alpha=0.3)
ax1.set_axisbelow(True)

# Plot 2: Arena Size Impact
arena_sizes = [32, 64, 128, 256, 512]
arena_ops = [65.2, 50.7, 34.7, 24.8, 24.8]  # Million ops/sec
arena_allocs = [11.49, 11.61, 11.98, 12.80, 12.80]  # Million allocs

ax2_twin = ax2.twinx()

line1 = ax2.plot(arena_sizes, arena_ops, 'o-', color='#2ca02c', linewidth=2.5,
                  markersize=10, label='Throughput (Ops/sec)')
line2 = ax2_twin.plot(arena_sizes, arena_allocs, 's--', color='#9467bd', linewidth=2,
                       markersize=8, label='Successful Allocs')

ax2.set_xlabel('Arena Size per Thread (MB)', fontsize=12)
ax2.set_ylabel('Operations per Second (Millions)', fontsize=12, color='#2ca02c')
ax2_twin.set_ylabel('Successful Allocations (Millions)', fontsize=12, color='#9467bd')
ax2.set_title('Fancy Allocator: Arena Size vs Performance\n(Trade-off: Speed vs Memory Capacity)', fontsize=13, fontweight='bold')

ax2.tick_params(axis='y', labelcolor='#2ca02c')
ax2_twin.tick_params(axis='y', labelcolor='#9467bd')

ax2.set_ylim(0, 75)
ax2_twin.set_ylim(11, 13.5)
ax2.set_xticks(arena_sizes)
ax2.grid(alpha=0.3)
ax2.set_axisbelow(True)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('benchmark_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('benchmark_plots.svg', bbox_inches='tight')
print("Saved: benchmark_plots.png and benchmark_plots.svg")

# Also create a simple bar chart for the README
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Relative performance (normalized to glibc = 1x)
relative_perf = [1.0, 1.4, 6.5, 9.9]
bars2 = ax3.barh(allocators, relative_perf, color=colors, edgecolor='black', linewidth=1.2)

ax3.set_xlabel('Relative Performance (higher is better)', fontsize=12)
ax3.set_title('Memory Allocator Speed Comparison\n(Normalized to glibc = 1.0x)', fontsize=13, fontweight='bold')
ax3.set_xlim(0, 12)

for bar, val in zip(bars2, relative_perf):
    ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}x', ha='left', va='center', fontsize=11, fontweight='bold')

ax3.grid(axis='x', alpha=0.3)
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: performance_comparison.png")

print("\nBenchmark Summary:")
print("=" * 50)
print(f"{'Allocator':<12} {'Ops/sec':>12} {'vs glibc':>12}")
print("-" * 50)
for alloc, ops in zip(allocators, ops_per_sec):
    ratio = ops / 7.6
    print(f"{alloc.replace(chr(10), ' '):<12} {ops:>10.1f}M {ratio:>10.1f}x")

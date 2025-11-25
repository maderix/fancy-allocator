#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks
threads = [1, 2, 4, 8, 16, 32, 64, 128]

glibc_ops = [2.9, 5.1, 7.0, 8.3, 7.3, 7.7, 8.2, 8.5]  # Million ops/sec
jemalloc_ops = [17.3, 31.8, 52.3, 78.5, 71.3, 93.8, 91.7, 95.8]
fancy_ops = [14.8, 27.2, 35.7, 43.4, 43.2, 43.4, 44.7, 47.4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Absolute performance
ax1.plot(threads, glibc_ops, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='glibc')
ax1.plot(threads, jemalloc_ops, 's-', color='#1f77b4', linewidth=2.5, markersize=8, label='jemalloc')
ax1.plot(threads, fancy_ops, '^-', color='#2ca02c', linewidth=2.5, markersize=8, label='Fancy')

ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Operations per Second (Millions)', fontsize=12)
ax1.set_title('Thread Scaling: Allocator Performance\n(250K ops/thread, mixed sizes)', fontsize=13, fontweight='bold')
ax1.set_xscale('log', base=2)
ax1.set_xticks(threads)
ax1.set_xticklabels(threads)
ax1.set_ylim(0, 110)
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

# Plot 2: Speedup vs single-threaded
glibc_speedup = [x / glibc_ops[0] for x in glibc_ops]
jemalloc_speedup = [x / jemalloc_ops[0] for x in jemalloc_ops]
fancy_speedup = [x / fancy_ops[0] for x in fancy_ops]
ideal_speedup = [t / threads[0] for t in threads]

ax2.plot(threads, ideal_speedup, '--', color='gray', linewidth=1.5, label='Ideal (linear)')
ax2.plot(threads, glibc_speedup, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='glibc')
ax2.plot(threads, jemalloc_speedup, 's-', color='#1f77b4', linewidth=2.5, markersize=8, label='jemalloc')
ax2.plot(threads, fancy_speedup, '^-', color='#2ca02c', linewidth=2.5, markersize=8, label='Fancy')

ax2.set_xlabel('Number of Threads', fontsize=12)
ax2.set_ylabel('Speedup vs Single-Threaded', fontsize=12)
ax2.set_title('Thread Scaling Efficiency\n(Higher = Better Scaling)', fontsize=13, fontweight='bold')
ax2.set_xscale('log', base=2)
ax2.set_xticks(threads)
ax2.set_xticklabels(threads)
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('thread_scaling.png', dpi=150, bbox_inches='tight')
plt.savefig('thread_scaling.svg', bbox_inches='tight')
print("Saved: thread_scaling.png and thread_scaling.svg")

# Print summary table
print("\n=== Thread Scaling Summary ===")
print(f"{'Threads':>8} {'glibc':>10} {'jemalloc':>10} {'Fancy':>10} {'Fancy/glibc':>12}")
print("-" * 55)
for i, t in enumerate(threads):
    ratio = fancy_ops[i] / glibc_ops[i]
    print(f"{t:>8} {glibc_ops[i]:>9.1f}M {jemalloc_ops[i]:>9.1f}M {fancy_ops[i]:>9.1f}M {ratio:>11.1f}x")

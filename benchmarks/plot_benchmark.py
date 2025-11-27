#!/usr/bin/env python3
"""
Generate benchmark plots from CSV data
Usage: python3 plot_benchmark.py <results_dir>
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'fancy': '#2ecc71',      # Green
    'glibc': '#e74c3c',      # Red
    'jemalloc': '#3498db',   # Blue
    'tcmalloc': '#f39c12'    # Orange
}
MARKERS = {'fancy': 'o', 'glibc': 's', 'jemalloc': '^', 'tcmalloc': 'D'}

def load_results(results_dir):
    """Load main results CSV"""
    csv_path = os.path.join(results_dir, 'results.csv')
    return pd.read_csv(csv_path)

def load_rss_timelines(results_dir):
    """Load all RSS timeline CSVs"""
    timelines = {}
    for f in glob.glob(os.path.join(results_dir, 'rss_*.csv')):
        basename = os.path.basename(f)
        # Parse rss_<allocator>_<threads>t.csv
        parts = basename.replace('rss_', '').replace('.csv', '').rsplit('_', 1)
        if len(parts) == 2:
            alloc = parts[0]
            threads = int(parts[1].replace('t', ''))
            df = pd.read_csv(f)
            if alloc not in timelines:
                timelines[alloc] = {}
            timelines[alloc][threads] = df
    return timelines

def plot_throughput(df, results_dir):
    """Plot throughput vs threads"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for alloc in df['allocator'].unique():
        data = df[df['allocator'] == alloc]
        ax.plot(data['threads'], data['mops_sec'],
                marker=MARKERS.get(alloc, 'o'),
                color=COLORS.get(alloc, '#333'),
                linewidth=2, markersize=8, label=alloc)

    ax.set_xlabel('Threads', fontsize=12)
    ax.set_ylabel('Throughput (Mops/sec)', fontsize=12)
    ax.set_title('Memory Allocator Throughput vs Thread Count', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'throughput.png'), dpi=150)
    plt.close()

def plot_latency_percentiles(df, results_dir):
    """Plot allocation latency percentiles"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    percentiles = [
        ('alloc_p50_us', 'Allocation P50'),
        ('alloc_p90_us', 'Allocation P90'),
        ('alloc_p99_us', 'Allocation P99'),
        ('alloc_p999_us', 'Allocation P99.9')
    ]

    for ax, (col, title) in zip(axes.flat, percentiles):
        for alloc in df['allocator'].unique():
            data = df[df['allocator'] == alloc]
            ax.plot(data['threads'], data[col],
                    marker=MARKERS.get(alloc, 'o'),
                    color=COLORS.get(alloc, '#333'),
                    linewidth=2, markersize=6, label=alloc)

        ax.set_xlabel('Threads', fontsize=10)
        ax.set_ylabel('Latency (μs)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Allocation Latency Percentiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'latency_percentiles.png'), dpi=150)
    plt.close()

def plot_latency_comparison(df, results_dir):
    """Bar chart comparing P99 latency at max threads"""
    max_threads = df['threads'].max()
    data = df[df['threads'] == max_threads]

    fig, ax = plt.subplots(figsize=(10, 6))

    allocators = data['allocator'].tolist()
    x = range(len(allocators))
    width = 0.35

    alloc_p99 = data['alloc_p99_us'].tolist()
    free_p99 = data['free_p99_us'].tolist()

    colors_list = [COLORS.get(a, '#333') for a in allocators]

    bars1 = ax.bar([i - width/2 for i in x], alloc_p99, width, label='Alloc P99', color=colors_list, alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], free_p99, width, label='Free P99', color=colors_list, alpha=0.5)

    ax.set_xlabel('Allocator', fontsize=12)
    ax.set_ylabel('Latency (μs)', fontsize=12)
    ax.set_title(f'P99 Latency Comparison @ {max_threads} Threads', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(allocators)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'latency_comparison.png'), dpi=150)
    plt.close()

def plot_memory(df, results_dir):
    """Plot peak RSS and fragmentation"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Peak RSS
    ax = axes[0]
    for alloc in df['allocator'].unique():
        data = df[df['allocator'] == alloc]
        ax.plot(data['threads'], data['peak_rss_mb'],
                marker=MARKERS.get(alloc, 'o'),
                color=COLORS.get(alloc, '#333'),
                linewidth=2, markersize=6, label=alloc)

    ax.set_xlabel('Threads', fontsize=10)
    ax.set_ylabel('Peak RSS (MB)', fontsize=10)
    ax.set_title('Peak Memory Usage', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Fragmentation
    ax = axes[1]
    for alloc in df['allocator'].unique():
        data = df[df['allocator'] == alloc]
        ax.plot(data['threads'], data['frag_pct'],
                marker=MARKERS.get(alloc, 'o'),
                color=COLORS.get(alloc, '#333'),
                linewidth=2, markersize=6, label=alloc)

    ax.set_xlabel('Threads', fontsize=10)
    ax.set_ylabel('Fragmentation (%)', fontsize=10)
    ax.set_title('Memory Fragmentation', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Memory Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'memory_metrics.png'), dpi=150)
    plt.close()

def plot_rss_timeline(timelines, results_dir, target_threads=None):
    """Plot RSS over time for specific thread count"""
    if not timelines:
        return

    # Find max threads available
    all_threads = set()
    for alloc_data in timelines.values():
        all_threads.update(alloc_data.keys())

    if target_threads is None:
        target_threads = max(all_threads) if all_threads else 128

    fig, ax = plt.subplots(figsize=(12, 6))

    for alloc, thread_data in timelines.items():
        if target_threads in thread_data:
            df = thread_data[target_threads]
            ax.plot(df['time_sec'], df['rss_mb'],
                    color=COLORS.get(alloc, '#333'),
                    linewidth=1.5, label=alloc, alpha=0.8)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('RSS (MB)', fontsize=12)
    ax.set_title(f'Memory Usage Over Time ({target_threads} threads)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'rss_timeline_{target_threads}t.png'), dpi=150)
    plt.close()

def plot_summary_dashboard(df, results_dir):
    """Create a summary dashboard"""
    fig = plt.figure(figsize=(16, 12))

    # Throughput (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    for alloc in df['allocator'].unique():
        data = df[df['allocator'] == alloc]
        ax1.plot(data['threads'], data['mops_sec'],
                marker=MARKERS.get(alloc, 'o'),
                color=COLORS.get(alloc, '#333'),
                linewidth=2, markersize=6, label=alloc)
    ax1.set_xlabel('Threads')
    ax1.set_ylabel('Mops/sec')
    ax1.set_title('Throughput', fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # P99 Latency (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    for alloc in df['allocator'].unique():
        data = df[df['allocator'] == alloc]
        ax2.plot(data['threads'], data['alloc_p99_us'],
                marker=MARKERS.get(alloc, 'o'),
                color=COLORS.get(alloc, '#333'),
                linewidth=2, markersize=6, label=alloc)
    ax2.set_xlabel('Threads')
    ax2.set_ylabel('Latency (μs)')
    ax2.set_title('Allocation P99 Latency', fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Peak RSS (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    for alloc in df['allocator'].unique():
        data = df[df['allocator'] == alloc]
        ax3.plot(data['threads'], data['peak_rss_mb'],
                marker=MARKERS.get(alloc, 'o'),
                color=COLORS.get(alloc, '#333'),
                linewidth=2, markersize=6, label=alloc)
    ax3.set_xlabel('Threads')
    ax3.set_ylabel('Peak RSS (MB)')
    ax3.set_title('Memory Usage', fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Summary table (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    max_threads = df['threads'].max()
    summary_data = df[df['threads'] == max_threads][['allocator', 'mops_sec', 'alloc_p99_us', 'peak_rss_mb']].round(2)
    summary_data.columns = ['Allocator', 'Mops/s', 'P99 (μs)', 'RSS (MB)']

    table = ax4.table(
        cellText=summary_data.values,
        colLabels=summary_data.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0']*4
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title(f'Summary @ {max_threads} Threads', fontweight='bold', pad=20)

    plt.suptitle('Memory Allocator Benchmark Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dashboard.png'), dpi=150)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_benchmark.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    print(f"Loading results from {results_dir}...")
    df = load_results(results_dir)
    timelines = load_rss_timelines(results_dir)

    print("Generating throughput plot...")
    plot_throughput(df, results_dir)

    print("Generating latency percentiles plot...")
    plot_latency_percentiles(df, results_dir)

    print("Generating latency comparison plot...")
    plot_latency_comparison(df, results_dir)

    print("Generating memory metrics plot...")
    plot_memory(df, results_dir)

    print("Generating RSS timeline plots...")
    for threads in [8, 16, 64, 128]:
        if any(threads in t for t in timelines.values()):
            plot_rss_timeline(timelines, results_dir, threads)

    print("Generating summary dashboard...")
    plot_summary_dashboard(df, results_dir)

    print(f"Plots saved to {results_dir}/")

if __name__ == '__main__':
    main()

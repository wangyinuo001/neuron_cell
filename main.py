import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
import gc

# ======================== 配置参数 ========================
base_dir = r'D:\!summer\ygx\data'
time_window = 2000
time_step = 0.05
sampling_rate = 20_000
trigger_threshold = 5

plt.rcParams.update({'figure.max_open_warning': 0})

# ==================== 动态创建输出目录 =====================
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_base_dir = rf'D:\!summer\ygx\output\analysis_{timestamp}'
heatmap_output_dir = os.path.join(output_base_dir, "spike_heatmaps")
waveforms_output_dir = os.path.join(output_base_dir, "waveforms")
stats_output_dir = os.path.join(output_base_dir, "statistics")

for path in [output_base_dir, heatmap_output_dir, waveforms_output_dir, stats_output_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

print(f"Output base directory: {os.path.abspath(output_base_dir)}")
print(f"  Heatmaps saved to: {heatmap_output_dir}")
print(f"  Waveforms saved to: {waveforms_output_dir}")
print(f"  Statistics saved to: {stats_output_dir}")
print(f"Analysis timestamp: {timestamp}")

# =================== 时间窗口信息 =======================
window_ms = time_window * time_step
print(f"\nConfiguration Summary:")
print(f"  Time window: {window_ms:.1f}ms ({time_window} data points)")
print(f"  Time step: {time_step}ms")
print(f"  Sampling rate: {sampling_rate / 1000:.0f}kHz")
print(f"  Trigger threshold: {trigger_threshold} standard deviations")
print(f"  Waveform resolution: 100% window displayed")

# ================== 热力图配色方案 ======================
colors = [
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 0.0, 0.0),
]
cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 波形颜色方案
wave_colors = {
    'background': '#f8f9fa',
    'grid': '#e9ecef',
    'inactive': '#adb5bd'
}


# ================== 简化波形数据提取函数 ====================
def extract_waveform_data(data, t, time_window, channel):
    """提取单个通道在时间窗口内的波形数据"""
    waveform = data[t:t + time_window, channel]
    return waveform


# ================== 简化波形可视化函数 ========================
def plot_waveforms(data, t, time_window, file_idx, event_id, output_dir):
    """为事件创建波形网格图（简化版）"""
    try:
        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        fig.suptitle(f"Spike Waveforms | File #{file_idx} - Event {event_id}\n"
                     f"Time Window: 0-100ms",
                     fontsize=18, y=0.98)
        fig.patch.set_facecolor(wave_colors['background'])

        # 计算所有通道的数据范围
        window_data = data[t:t + time_window, :]
        global_min = np.min(window_data)
        global_max = np.max(window_data)
        y_margin = (global_max - global_min) * 0.1

        # 在网格中绘制所有64个通道的波形
        for ch in range(64):
            row = ch // 8
            col = ch % 8
            ax = axes[row, col]

            # 提取波形数据
            waveform = extract_waveform_data(data, t, time_window, ch)

            # 生成时间轴
            time_axis = np.linspace(0, window_ms, len(waveform))

            # 绘制波形（黑色线）
            ax.plot(time_axis, waveform, 'k-', linewidth=0.8, alpha=0.8)

            # 设置Y轴范围
            ax.set_ylim(global_min - y_margin, global_max + y_margin)

            # 网格设置
            ax.grid(True, color=wave_colors['grid'], linewidth=0.4, linestyle=':', alpha=0.7)
            ax.tick_params(labelsize=7)
            ax.set_title(f"Ch{ch} (R{row + 1}/C{col + 1})", fontsize=9, pad=3)

        # 添加坐标轴标签
        for row in range(8):
            axes[row, 0].set_ylabel('Voltage', fontsize=8)
        for col in range(8):
            axes[7, col].set_xlabel('Time (ms)', fontsize=8)

        # 设置整体布局
        plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.5, w_pad=0.5)

        # 保存波形图
        wave_file = os.path.join(output_dir, f"File{file_idx}_Event{event_id}_waveforms.png")
        fig.savefig(wave_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        gc.collect()
        return wave_file

    except Exception as e:
        print(f"    Waveform generation error: {str(e)}")
        return None


# ================= 文件处理函数 ========================
def process_csv_file(csv_path, file_idx, stats_dir, heatmap_dir, wave_dir):
    filename = os.path.basename(csv_path)
    print(f"\n{'=' * 70}")
    print(f"PROCESSING FILE #{file_idx}: {filename}")
    start_time = time.time()

    try:
        df = pd.read_csv(csv_path)
        time_ms = df['time_ms'].values
        data = df.iloc[:, 1:].values
        print(f"  Loaded {len(time_ms)} data points ({time_ms[-1] - time_ms[0]:.2f}ms duration)")
        if len(time_ms) == 0:
            print("  No data found, skipping file...")
            return [], 0, 0

    except Exception as e:
        print(f"  ERROR loading file: {str(e)}")
        return [], 0, 0

    try:
        thresholds = df.iloc[:, 1:].mean() + trigger_threshold * df.iloc[:, 1:].std()
        th_min, th_max = thresholds.min(), thresholds.max()
        print(f"  Threshold range: {th_min:.4f} - {th_max:.4f} (Δ={th_max - th_min:.4f})")

        triggered = (data > thresholds.values).astype(np.int8)
        total_points = len(time_ms)

        spike_data = []
        t = 0
        event_count = 0

        print("  Starting event detection and waveform processing...")
        print("  Status: [Heatmap] [Waveforms] [ActiveCh] [Progress]")
        last_heartbeat = time.time()

        while t < total_points - time_window:
            if np.any(triggered[t, :]):
                event_count += 1
                start_time_val = time_ms[t]
                end_time_val = time_ms[t + time_window - 1]

                first_trigger_times = np.full(64, -1, dtype=np.int32)
                window_end = min(t + time_window, total_points)
                for i in range(t, window_end):
                    for ch in range(64):
                        if triggered[i, ch] and first_trigger_times[ch] == -1:
                            first_trigger_times[ch] = i

                active_channels = np.count_nonzero(first_trigger_times != -1)
                if active_channels > 1:
                    heatmap_data = np.full((8, 8), np.nan)
                    valid_data = []

                    for ch in range(64):
                        if first_trigger_times[ch] != -1:
                            i_grid, j_grid = np.unravel_index(ch, (8, 8))
                            time_proportion = (first_trigger_times[ch] - t) / time_window
                            heatmap_data[i_grid, j_grid] = time_proportion

                            spike_data.append({
                                'timestamp': timestamp,
                                'file_num': file_idx,
                                'filename': filename,
                                'event_id': event_count,
                                'start_time': start_time_val,
                                'end_time': end_time_val,
                                'trigger_time': time_ms[first_trigger_times[ch]],
                                'channel': ch,
                                'grid_row': i_grid + 1,
                                'grid_col': j_grid + 1,
                                'time_proportion': time_proportion,
                                'active_channels': active_channels
                            })
                            valid_data.append(ch)

                    heatmap_file = None
                    wave_file = None
                    try:
                        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
                        heatmap = ax_heat.imshow(heatmap_data, cmap=cmap, interpolation='nearest',
                                                 vmin=0, vmax=1, aspect='auto')

                        for i in range(1, 8):
                            ax_heat.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
                            ax_heat.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)

                        ax_heat.set_xticks(np.arange(8))
                        ax_heat.set_yticks(np.arange(8))
                        ax_heat.set_xticklabels(range(1, 9))
                        ax_heat.set_yticklabels(range(1, 9))
                        ax_heat.set_xlabel("Electrode Column", fontsize=10)
                        ax_heat.set_ylabel("Electrode Row", fontsize=10)

                        ax_heat.set_title(f"Heatmap | File {file_idx}, Event {event_count}", fontsize=12)
                        cbar = fig_heat.colorbar(heatmap, ax=ax_heat, label='Trigger Proportion')
                        cbar.ax.tick_params(labelsize=8)

                        heatmap_file = os.path.join(heatmap_dir, f"File{file_idx}_Event{event_count}.png")
                        fig_heat.savefig(heatmap_file, dpi=150, bbox_inches='tight')
                        plt.close(fig_heat)

                        # 生成简化版波形图（无标记）
                        wave_file = plot_waveforms(
                            data,
                            t,
                            time_window,
                            file_idx,
                            event_count,
                            wave_dir
                        )
                        status = "[✓] [✓]"
                    except Exception as e:
                        status = "[✗] [✗]"
                        print(f"    Visualization error: {str(e)}")

                    if time.time() - last_heartbeat > 10:
                        elapsed = time_ms[t] - time_ms[0]
                        time_percent = (t / total_points) * 100
                        active_ch_disp = f"{active_channels} ch"
                        print(f"  Heartbeat: {elapsed:.1f}ms ({time_percent:.1f}%), Events: {event_count}")
                        last_heartbeat = time.time()
                else:
                    event_count -= 1

                t += time_window
            else:
                t += 1

            if t % 10000 == 0 and time.time() - last_heartbeat > 1:
                elapsed = time_ms[t] - time_ms[0]
                time_percent = (t / total_points) * 100
                print(f"  Progress: {elapsed:.1f}ms ({time_percent:.1f}%), Events: {event_count}")
                last_heartbeat = time.time()

        if spike_data:
            event_df = pd.DataFrame(spike_data)
            csv_file = os.path.join(stats_dir, f"SpikeEvents_File{file_idx}.csv")
            event_df.to_csv(csv_file, index=False)

    except Exception as e:
        print(f"  ERROR during processing: {str(e)}")
        print(f"  Skipping file {filename}")
        processing_time = time.time() - start_time
        return [], processing_time, 0

    processing_time = time.time() - start_time
    points_per_sec = len(time_ms) / processing_time if processing_time > 0 else 0
    total_spikes = len(spike_data) if spike_data else 0
    total_events = event_df['event_id'].nunique() if spike_data else 0

    print(f"\n  Processing complete for file #{file_idx}")
    print(f"  Events detected: {event_count}")
    print(f"  Multi-spike events: {total_events}")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Processing rate: {points_per_sec / 1000:.1f}K points/sec")
    print(f"  Time taken: {processing_time:.1f} seconds")

    for name in ['data', 'triggered', 'thresholds', 'df']:
        if name in locals():
            del locals()[name]
    gc.collect()

    return spike_data, processing_time, total_events


# ====================== 主处理流程 =======================
if __name__ == '__main__':
    print("\nStarting neural spike analysis...")

    csv_files = [f for f in os.listdir(base_dir) if f.lower().endswith('.csv') and f[:3].isdigit()]
    csv_files.sort(key=lambda x: int(x.split('.')[0]))
    print(f"Found {len(csv_files)} CSV files to process:")

    all_events = []
    summary_stats = []

    total_start = time.time()
    processed_files = 0
    total_files = len(csv_files)

    for idx, csv_file in enumerate(csv_files, start=1):
        file_path = os.path.join(base_dir, csv_file)
        file_start = time.time()

        print(f"\n{'=' * 70}")
        print(f"Processing file {idx}/{total_files}: {csv_file}")
        print('=' * 70)

        try:
            spike_data, proc_time, multi_events = process_csv_file(
                file_path,
                idx,
                stats_output_dir,
                heatmap_output_dir,
                waveforms_output_dir
            )

            events_detected = multi_events if spike_data else 0
            file_stats = {
                'file_id': idx,
                'filename': csv_file,
                'events_detected': events_detected,
                'spikes_recorded': len(spike_data),
                'processing_time_s': proc_time,
                'start_datetime': timestamp
            }
            summary_stats.append(file_stats)

            if spike_data:
                all_events.extend(spike_data)

            processed_files += 1
            elapsed_time = time.time() - total_start
            est_total = (elapsed_time / processed_files) * total_files
            remaining = est_total - elapsed_time

            print(f"\nCumulative progress: {processed_files}/{total_files} files")
            print(f"Elapsed time: {elapsed_time / 60:.1f} min")
            print(f"Estimated remaining: {remaining / 60:.1f} min")

        except Exception as e:
            import traceback

            print(f"\nFATAL ERROR processing file #{idx}: {str(e)}")
            traceback.print_exc()
            print('-' * 50)
            summary_stats.append({
                'file_id': idx,
                'filename': csv_file,
                'error': str(e),
                'processing_time_s': time.time() - file_start,
                'start_datetime': timestamp
            })

    if all_events:
        all_events_df = pd.DataFrame(all_events)
        all_events_file = os.path.join(stats_output_dir, f"Combined_Spike_Events_{timestamp}.csv")
        all_events_df.to_csv(all_events_file, index=False)

    if summary_stats:
        stats_df = pd.DataFrame(summary_stats)
        stats_file = os.path.join(stats_output_dir, f"Analysis_Summary_{timestamp}.csv")
        stats_df.to_csv(stats_file, index=False)

    total_time = time.time() - total_start
    total_spikes = len(all_events)
    total_events = stats_df['events_detected'].sum() if summary_stats else 0

    print("\n" + "=" * 70)
    print("NEURAL SPIKE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Files processed: {len(csv_files)}")
    print(f"Total spike events detected: {total_events}")
    print(f"Total spikes recorded: {total_spikes}")
    print(f"\nProcessing performance:")
    print(f"  Total processing time: {total_time / 60:.1f} min")
    print(f"  Average per file: {total_time / len(csv_files):.2f} sec\n" if csv_files else "")
    print(f"Output directories:")
    print(f"  Base: {output_base_dir}")
    print(f"  Heatmaps: {heatmap_output_dir}")
    print(f"  Waveforms: {waveforms_output_dir}")
    print(f"  Statistics: {stats_output_dir}")
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

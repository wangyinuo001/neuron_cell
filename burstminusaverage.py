import os
import time
import gc
import traceback
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks

# ===================== 可调参数 =====================
# 数据参数
base_dir = r'D:\!summer\ygx\data'  # CSV文件目录
sampling_freq = 20000  # 采样频率 (Hz)
time_step = 1000 / sampling_freq  # 时间步长 (ms)

# 爆发检测参数
min_burst_duration = 10  # 最小爆发时长 (ms)
max_burst_duration = 100  # 最大爆发时长 (ms)
burst_merge_threshold = 25  # 合并爆发最大时间间隔 (ms)
pre_post_window = 50  # 窗口大小 (ms)
spike_threshold_factor = 5.0  # 尖峰检测阈值倍数 (基于MAD)
min_spikes_per_burst = 3  # 爆发最小尖峰数
safety_margin = 5  # 安全边界 (样本数)

# 滤波参数
car_enabled = True  # 启用共同平均参考
filter_cutoff = 800  # 低通滤波截止频率 (Hz)
savgol_window = 5  # Savitzky-Golay平滑窗口
interp_factor = 4  # 波形插值因子

# 可视化参数
scale_bar_time = 20  # 时间比例尺长度 (ms)
scale_bar_voltage = 50  # 电压比例尺幅度 (单位)

# ===================== 自动配置 =====================
# 输出目录设置
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_base_dir = rf'D:\!summer\ygx\output\burst_analysis_{timestamp}'
burst_waveforms_dir = os.path.join(output_base_dir, "burst_waveforms")
detection_debug_dir = os.path.join(output_base_dir, "detection_debug")
stats_output_dir = os.path.join(output_base_dir, "statistics")

# 创建输出目录
for path in [output_base_dir, burst_waveforms_dir, detection_debug_dir, stats_output_dir]:
    os.makedirs(path, exist_ok=True)

# 可视化后端
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 颜色方案
color_palette = {
    'raw': '#BDC3C7',  # 原始波形
    'filtered': '#2C3E50',  # 滤波后波形
    'spike': '#E74C3C',  # 尖峰标记
    'burst': '#3498DB',  # 爆发区域
    'threshold': '#F39C12',  # 阈值线
    'grid': '#D6DBDF'  # 网格线
}


# ===================== 核心函数 =====================
def load_and_validate_csv(csv_path):
    """加载CSV文件并验证数据结构"""
    print(f"正在加载文件: {os.path.basename(csv_path)}")

    try:
        # 加载CSV
        df = pd.read_csv(csv_path)

        # 查找有效数据列
        numeric_cols = []
        for col in df.columns:
            # 跳过非数值列
            if not np.issubdtype(df[col].dtype, np.number):
                continue

            # 跳过全零或常数列
            if df[col].var() < 1e-6:
                continue

            numeric_cols.append(col)

        # 确保有足够通道
        if len(numeric_cols) < 16:
            if len(df.columns) >= 64:
                print("  警告: 使用所有数值列 - 可能包含无效数据")
                numeric_cols = df.columns[:64]
            else:
                raise ValueError(f"仅找到{len(numeric_cols)}个有效通道 (需要至少16)")

        # 提取数据
        data = df[numeric_cols].values.astype(np.float32)
        n_samples, n_channels = data.shape
        print(f"  数据尺寸: {n_samples}采样点 × {n_channels}通道")

        return data, numeric_cols

    except Exception as e:
        print(f"文件加载失败: {str(e)}")
        traceback.print_exc()
        return None, None


def apply_common_average_reference(data):
    """应用共同平均参考处理 (可选)"""
    if not car_enabled or data.shape[1] < 4:
        print("  跳过CAR处理")
        return data.copy()

    print("  应用CAR...")
    mean_signal = np.nanmean(data, axis=1, keepdims=True)
    return data - mean_signal


def design_bandpass_filter(fs, lowcut=None, highcut=None, order=4):
    """设计带通滤波器"""
    nyquist = 0.5 * fs

    if lowcut is None:
        b, a = signal.butter(order, highcut / nyquist, btype='low')
    elif highcut is None:
        b, a = signal.butter(order, lowcut / nyquist, btype='high')
    else:
        b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

    return b, a


def apply_filter(data, b, a):
    """应用零相位滤波器"""
    filtered = np.zeros_like(data)

    for ch in range(data.shape[1]):
        # 处理NaN
        channel_data = data[:, ch].copy()
        non_nan_mask = ~np.isnan(channel_data)

        if np.sum(non_nan_mask) < 10:
            filtered[:, ch] = channel_data
            continue

        # 填充NaN
        valid_indices = np.where(non_nan_mask)[0]
        channel_data = np.interp(np.arange(len(channel_data)), valid_indices, channel_data[valid_indices])

        # 滤波
        try:
            filtered[:, ch] = signal.filtfilt(b, a, channel_data)
        except:
            filtered[:, ch] = channel_data

    return filtered


def calculate_mad_threshold(data, factor=5.0):
    """基于中位数绝对偏差计算阈值 (MAD)"""
    thresholds = []
    print("  计算通道阈值:")

    for ch in range(data.shape[1]):
        channel_data = data[:, ch]

        # 移除NaN
        clean_data = channel_data[~np.isnan(channel_data)]
        if len(clean_data) < 100:
            thresholds.append(np.nan)
            continue

        # 计算中位数和绝对偏差
        median_val = np.median(clean_data)
        abs_dev = np.abs(clean_data - median_val)

        # 计算MAD (中位数绝对偏差)
        mad = np.median(abs_dev)

        # 转换为标准差估算 (正态分布)
        std_estimate = 1.4826 * mad

        # 设置阈值 (高于基线)
        threshold = median_val + factor * std_estimate
        thresholds.append(threshold)

        # 打印前几个通道
        if ch < 5:
            print(f"    通道 {ch + 1}: 中位数={median_val:.2f}, MAD={mad:.2f}, 阈值={threshold:.2f}")

    return np.array(thresholds)


def detect_spikes(data, thresholds):
    """检测超过阈值的尖峰事件"""
    if np.all(np.isnan(thresholds)):
        return []

    all_spikes = []

    for ch in range(data.shape[1]):
        if np.isnan(thresholds[ch]):
            continue

        # 寻找峰值
        peaks, _ = find_peaks(data[:, ch], height=thresholds[ch])

        # 收集尖峰
        for peak_idx in peaks:
            all_spikes.append((int(peak_idx), ch))

    return all_spikes


def merge_spikes_to_bursts(all_spikes, sampling_rate, min_duration=10, max_duration=100, merge_threshold=25):
    """将尖峰聚类为爆发事件"""
    if not all_spikes:
        return []

    # 按时间排序
    sorted_spikes = sorted(all_spikes, key=lambda x: x[0])

    # 聚类参数
    min_duration_samples = int(min_duration * sampling_rate / 1000)
    max_duration_samples = int(max_duration * sampling_rate / 1000)
    merge_threshold_samples = int(merge_threshold * sampling_rate / 1000)

    bursts = []
    current_burst = [sorted_spikes[0]]

    for i in range(1, len(sorted_spikes)):
        prev_idx, _ = sorted_spikes[i - 1]
        curr_idx, _ = sorted_spikes[i]

        if curr_idx - prev_idx <= merge_threshold_samples:
            current_burst.append(sorted_spikes[i])
        else:
            # 结束当前爆发
            start_idx = min(x[0] for x in current_burst)
            end_idx = max(x[0] for x in current_burst)
            duration = end_idx - start_idx

            # 检查持续时间
            if min_duration_samples <= duration <= max_duration_samples:
                bursts.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration_ms': duration * 1000 / sampling_rate,
                    'spikes': current_burst.copy()
                })

            # 开始新爆发
            current_burst = [sorted_spikes[i]]

    # 处理最后一个爆发
    if current_burst:
        start_idx = min(x[0] for x in current_burst)
        end_idx = max(x[0] for x in current_burst)
        duration = end_idx - start_idx

        if min_duration_samples <= duration <= max_duration_samples:
            bursts.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration_ms': duration * 1000 / sampling_rate,
                'spikes': current_burst.copy()
            })

    return bursts


def plot_burst_waveforms(raw_data, filtered_data, burst, channel_names, file_name, event_id):
    """为单个爆发绘制64通道波形图"""
    try:
        # 确定时间窗口
        start_idx = max(0, burst['start_idx'] - int(pre_post_window * sampling_freq / 1000))
        end_idx = min(len(raw_data), burst['end_idx'] + int(pre_post_window * sampling_freq / 1000))

        # 创建时间轴
        time_axis = np.arange(start_idx, end_idx) * time_step

        # 创建图像
        fig = plt.figure(figsize=(16, 16), dpi=100)
        fig.suptitle(
            f"{os.path.basename(file_name)} - Burst #{event_id}",
            fontsize=18, y=0.98
        )

        # 计算全局Y轴范围
        channel_min = np.nanmin(filtered_data[start_idx:end_idx, :])
        channel_max = np.nanmax(filtered_data[start_idx:end_idx, :])
        y_range = max(60, channel_max - channel_min)  # 最小范围60μV
        y_min = channel_min - 0.2 * y_range
        y_max = channel_max + 0.2 * y_range

        # 绘制每个通道
        for ch in range(filtered_data.shape[1]):
            ax = plt.subplot(8, 8, ch + 1)

            # 原始波形
            ax.plot(
                time_axis, raw_data[start_idx:end_idx, ch],
                color=color_palette['raw'],
                linewidth=0.8,
                alpha=0.6,
                label='Raw'
            )

            # 过滤后波形
            ax.plot(
                time_axis, filtered_data[start_idx:end_idx, ch],
                color=color_palette['filtered'],
                linewidth=1.4,
                alpha=0.9,
                label='Filtered'
            )

            # 标记当前爆发
            burst_start = burst['start_idx'] * time_step
            burst_end = burst['end_idx'] * time_step

            ax.axvspan(
                burst_start, burst_end,
                alpha=0.1, color=color_palette['burst'],
                label='Burst'
            )

            # 标记当前通道的尖峰
            ch_spikes = [idx for idx, spike_ch in burst['spikes'] if spike_ch == ch]
            for spike_idx in ch_spikes:
                spike_time = spike_idx * time_step
                spike_val = filtered_data[spike_idx, ch]

                ax.scatter(
                    spike_time, spike_val,
                    s=25, color=color_palette['spike'],
                    marker='*', zorder=10
                )

            # 设置轴范围
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(time_axis[0], time_axis[-1])

            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.2, color=color_palette['grid'])

            # 去除坐标标签
            ax.set_xticks([])
            ax.set_yticks([])

            # 通道标题
            channel_name = channel_names[ch] if ch < len(channel_names) else f"Ch {ch + 1}"
            ax.set_title(channel_name, fontsize=9, pad=2)

            # 只在底部通道添加时间轴
            if ch >= 56:
                ax.set_xlabel("Time (ms)", fontsize=8)
                ax.set_xticks(
                    time_axis[0] + np.array([0, (time_axis[-1] - time_axis[0]) / 2, time_axis[-1] - time_axis[0]]))
                ax.set_xticklabels(
                    ["0", f"{int((time_axis[-1] - time_axis[0]) / 2)}", f"{int(time_axis[-1] - time_axis[0])}"],
                    fontsize=6)

        # 添加全局图例
        plt.figlegend(
            ['Raw', 'Filtered', 'Burst', 'Spike'],
            loc='lower center',
            frameon=True,
            ncol=4,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.02)
        )

        # 添加比例尺
        plt.figtext(
            0.95, 0.05,
            f"Time Scale: {scale_bar_time} ms\nVoltage Scale: {scale_bar_voltage} μV",
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7),
            fontsize=9
        )

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05, top=0.95)

        # 保存文件
        output_file = os.path.join(burst_waveforms_dir,
                                   f"{os.path.basename(file_name).replace('.csv', '')}_burst_{event_id:03d}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=120)
        plt.close(fig)
        gc.collect()

        print(f"      保存图像: {os.path.basename(output_file)}")
        return output_file

    except Exception as e:
        print(f"  绘图错误: {str(e)}")
        traceback.print_exc()
        return None


def plot_detection_debug(raw_data, filtered_data, thresholds, bursts, channel_names, file_name):
    """绘制检测过程调试图"""
    try:
        # 随机选择一个通道用于调试
        if raw_data.shape[1] < 1:
            return

        ch = min(5, raw_data.shape[1] - 1)
        channel_name = channel_names[ch] if ch < len(channel_names) else f"Ch {ch + 1}"

        # 创建图形
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 原始数据 (顶部)
        ax1.set_title(f"{file_name} - {channel_name}", fontsize=12)
        ax1.set_ylabel("Amplitude (μV)", fontsize=10)

        # 绘制原始和过滤后数据
        t = np.arange(len(raw_data)) * time_step
        ax1.plot(t, raw_data[:, ch], color=color_palette['raw'], linewidth=0.8, alpha=0.7, label="Raw Data")
        ax1.plot(t, filtered_data[:, ch], color=color_palette['filtered'], linewidth=1.2, alpha=0.9, label="Filtered")

        # 绘制阈值线
        if not np.isnan(thresholds[ch]):
            ax1.axhline(thresholds[ch], color=color_palette['threshold'], linestyle='--', alpha=0.8, label="Threshold")

        # 标记爆发
        for i, burst in enumerate(bursts):
            burst_start_t = burst['start_idx'] * time_step
            burst_end_t = burst['end_idx'] * time_step

            # 标记爆发区域
            ax1.axvspan(
                burst_start_t, burst_end_t,
                alpha=0.15, color=color_palette['burst'],
                label="Burst Region" if i == 0 else None
            )

            # 标记尖峰
            ch_spikes = [idx for idx, spike_ch in burst['spikes'] if spike_ch == ch]
            spike_times = [i * time_step for i in ch_spikes]
            spike_vals = [filtered_data[i, ch] for i in ch_spikes]

            ax1.scatter(
                spike_times, spike_vals,
                color=color_palette['spike'], s=25,
                marker='*', zorder=10,
                label="Spikes" if i == 0 else None
            )

        # 添加图例
        ax1.legend(loc='upper right', fontsize=9)

        # 设置轴范围
        view_t = 1000  # 只显示前1000ms
        if len(t) > int(view_t / time_step):
            ax1.set_xlim(0, view_t)

        # 美化
        ax1.grid(True, linestyle='--', alpha=0.2, color=color_palette['grid'])
        plt.tight_layout()

        # 保存文件
        output_file = os.path.join(detection_debug_dir,
                                   f"{os.path.basename(file_name).replace('.csv', '')}_detection_debug.png")
        plt.savefig(output_file, dpi=120)
        plt.close(fig)
        gc.collect()

        print(f"  保存调试图: {os.path.basename(output_file)}")
        return output_file

    except Exception as e:
        print(f"  调试图错误: {str(e)}")
        return None


def process_file(csv_path):
    """处理单个CSV文件"""
    start_time = time.time()
    filename = os.path.basename(csv_path)
    print(f"\n{'=' * 50}")
    print(f"处理文件: {filename}")

    try:
        # 1. 加载数据
        data, channel_names = load_and_validate_csv(csv_path)
        if data is None:
            return 0

        # 2. 数据预处理
        # a. 共同平均参考
        car_data = apply_common_average_reference(data)

        # b. 滤波
        b, a = design_bandpass_filter(sampling_freq, highcut=filter_cutoff)
        filtered_data = apply_filter(car_data, b, a)

        # 3. 设置阈值
        thresholds = calculate_mad_threshold(filtered_data, factor=spike_threshold_factor)

        # 4. 尖峰检测
        all_spikes = detect_spikes(filtered_data, thresholds)
        print(f"  检测到尖峰总数: {len(all_spikes)}")

        # 5. 爆发检测
        bursts = merge_spikes_to_bursts(
            all_spikes,
            sampling_rate=sampling_freq,
            min_duration=min_burst_duration,
            max_duration=max_burst_duration,
            merge_threshold=burst_merge_threshold
        )
        print(f"  检测到的爆发数: {len(bursts)}")

        # 6. 绘制调试信息
        plot_detection_debug(data, filtered_data, thresholds, bursts, channel_names, filename)

        # 7. 处理每个爆发
        burst_info = []
        for i, burst in enumerate(bursts):
            # 跳过太小的爆发
            if len(burst['spikes']) < min_spikes_per_burst:
                continue

            print(f"    爆发 {i + 1}: "
                  f"起始={burst['start_idx']} ({burst['start_idx'] * time_step:.1f}ms), "
                  f"结束={burst['end_idx']} ({burst['end_idx'] * time_step:.1f}ms), "
                  f"持续={burst['duration_ms']:.2f}ms, "
                  f"尖峰={len(burst['spikes'])}")

            # 绘制爆发波形
            plot_file = plot_burst_waveforms(
                data, filtered_data,
                burst,
                channel_names,
                filename,
                i + 1
            )

            # 保存爆发信息
            burst_channels = set(ch for _, ch in burst['spikes'])
            burst_info.append({
                'file': filename,
                'burst_id': i + 1,
                'start_time_ms': burst['start_idx'] * time_step,
                'end_time_ms': burst['end_idx'] * time_step,
                'duration_ms': burst['duration_ms'],
                'spike_count': len(burst['spikes']),
                'channel_count': len(burst_channels),
                'channels': ",".join(str(c) for c in burst_channels),
                'image_file': os.path.basename(plot_file) if plot_file else ""
            })

        # 8. 保存统计信息
        df_bursts = pd.DataFrame(burst_info)
        if not df_bursts.empty:
            stats_file = os.path.join(stats_output_dir, f"{filename.replace('.csv', '')}_burst_stats.csv")
            df_bursts.to_csv(stats_file, index=False)
            print(f"  保存统计: {stats_file}")

        # 清理内存
        del data, filtered_data
        gc.collect()

        return len(bursts)

    except Exception as e:
        print(f"  文件处理错误: {str(e)}")
        traceback.print_exc()
        return 0


def main():
    print("\n" + "=" * 60)
    print("爆发事件检测器")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据目录: {base_dir}")
    print("=" * 60 + "\n")

    # 收集CSV文件
    csv_files = []
    for f in os.listdir(base_dir):
        if f.lower().endswith('.csv'):
            csv_files.append(os.path.join(base_dir, f))

    if not csv_files:
        print("错误: 未找到CSV文件!")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 处理文件
    total_bursts = 0
    processed_files = 0

    for file_idx, csv_path in enumerate(csv_files):
        file_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"文件 {file_idx + 1}/{len(csv_files)}: {os.path.basename(csv_path)}")

        num_bursts = process_file(csv_path)
        total_bursts += num_bursts
        processed_files += 1

        print(f"  耗时: {time.time() - file_start:.1f}秒 | 爆发数: {num_bursts}")

    # 最终统计
    print("\n" + "=" * 60)
    print(f"处理完成! 共处理 {processed_files} 个文件")
    print(f"检测到 {total_bursts} 个爆发事件")
    print(f"输出目录: {output_base_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

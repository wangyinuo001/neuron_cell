import os
import time
import gc
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================== 配置参数 ========================
base_dir = r'D:\!summer\ygx\data'  # CSV 文件夹
time_step = 0.05  # ms
sampling_rate = 20000
burst_merge_threshold = 25  # 连续尖峰被视为同一爆发的最大时间间隔(ms)
pre_post_window = 50  # 爆发事件前后的时间窗口(ms)
burst_length_limit = 100  # 最大爆发时长(ms)

# 输出目录
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_base_dir = rf'D:\!summer\ygx\output\burst_analysis_{timestamp}'
burst_waveforms_dir = os.path.join(output_base_dir, "burst_waveforms")
cluster_output_dir = os.path.join(output_base_dir, "clustering")
stats_output_dir = os.path.join(output_base_dir, "statistics")

for path in [output_base_dir, burst_waveforms_dir, cluster_output_dir, stats_output_dir]:
    os.makedirs(path, exist_ok=True)

# 颜色方案 (去除背景色)
color_palette = {
    'grid': '#DEE2E6',
    'waveform': '#2D3436',
    'cluster_colors': ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#2ecc71', '#34495e']
}


# ======================== 函数 ========================
def calculate_thresholds(data):
    """计算每个通道的MAD阈值"""
    thresholds = []
    for ch in range(data.shape[1]):
        channel_data = data[:, ch]
        median_val = np.median(channel_data)
        mad = np.median(np.abs(channel_data - median_val))
        threshold = median_val + 5 * 1.4826 * mad
        thresholds.append(threshold)
    return np.array(thresholds)


def detect_global_burst_events(data, thresholds):
    """检测全局爆发事件"""
    all_spikes = []
    # 检测所有通道的所有尖峰
    for ch in range(data.shape[1]):
        spike_idxs = np.where(data[:, ch] > thresholds[ch])[0]
        all_spikes.extend([(idx, ch) for idx in spike_idxs])

    if not all_spikes:
        return []

    # 按时间排序尖峰
    all_spikes.sort(key=lambda x: x[0])

    burst_events = []
    current_burst = [all_spikes[0]]
    merge_threshold_samples = int(burst_merge_threshold / time_step)

    # 尝试合并接近的尖峰
    for i in range(1, len(all_spikes)):
        prev_idx, _ = all_spikes[i - 1]
        curr_idx, _ = all_spikes[i]

        # 检查是否属于同一爆发事件
        if curr_idx - prev_idx < merge_threshold_samples:
            current_burst.append(all_spikes[i])
        else:
            # 提取爆发事件的起止位置
            start_idx = min(idx for idx, _ in current_burst)
            end_idx = max(idx for idx, _ in current_burst)
            burst_duration = (end_idx - start_idx) * time_step

            # 过滤太长的爆发
            if 0 < burst_duration <= burst_length_limit:
                burst_events.append({
                    'channel_spikes': current_burst.copy(),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'burst_duration': burst_duration
                })

            # 开始新的爆发
            current_burst = [all_spikes[i]]

    # 处理最后的爆发
    if current_burst:
        start_idx = min(idx for idx, _ in current_burst)
        end_idx = max(idx for idx, _ in current_burst)
        burst_duration = (end_idx - start_idx) * time_step
        if 0 < burst_duration <= burst_length_limit:
            burst_events.append({
                'channel_spikes': current_burst.copy(),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'burst_duration': burst_duration
            })

    return burst_events


def plot_burst_waveforms(data, burst, file_name, event_id, output_dir):
    """绘制64通道波形"""
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"{file_name} Burst {event_id}\n"
                 f"Duration: {burst['burst_duration']:.1f}ms | Spikes: {len(burst['channel_spikes'])}",
                 fontsize=20, y=0.98)

    # 计算时间窗口
    pre_samples = int(pre_post_window / time_step)
    start_idx = max(0, burst['start_idx'] - pre_samples)
    end_idx = min(data.shape[0], burst['end_idx'] + pre_samples)
    time_axis = np.arange(start_idx, end_idx) * time_step
    burst_start = time_axis[0]
    burst_end = time_axis[-1]

    axes = []
    for i in range(64):
        ax = plt.subplot(8, 8, i + 1)

        # 网格设置
        ax.grid(color=color_palette['grid'], linestyle='--', linewidth=0.5, alpha=0.7)

        # 绘制原始波形
        channel_data = data[start_idx:end_idx, i]
        ax.plot(time_axis, channel_data, '-', color=color_palette['waveform'], linewidth=1.0, alpha=0.8)

        # 标记此通道的所有尖峰
        channel_spikes = [sp[0] for sp in burst['channel_spikes'] if sp[1] == i]
        for spike_idx in channel_spikes:
            if start_idx <= spike_idx < end_idx:
                # 只用垂直线表示尖峰位置，不使用彩色
                ax.axvline(spike_idx * time_step, color='k', linewidth=0.5, alpha=0.5)

        # 设置坐标轴
        ax.set_xlim([burst_start, burst_end])
        ax.set_yticks([])  # 隐藏Y轴刻度

        # 刻度设置
        ax.set_xticks([])
        if i >= 56:  # 为底部通道添加刻度
            ax.set_xticks([np.min(time_axis), np.mean(time_axis), np.max(time_axis)])
            ax.set_xticklabels([f"{t:.0f}ms" for t in ax.get_xticks()])
            ax.set_xlabel('Time (ms)', fontsize=8, labelpad=2)
        else:
            ax.set_xticks([])

        ax.set_title(f"Ch {i + 1}", fontsize=9)
        axes.append(ax)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    # 保存图像
    wave_file = os.path.join(output_dir, f"{file_name.replace('_', '')}_Burst{event_id}.png")
    plt.savefig(wave_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return wave_file


def extract_burst_features(burst, data):
    """提取爆发特征用于聚类"""
    features = []

    # 1. 持续时间特征
    duration = burst['burst_duration']
    features.append(duration)

    # 2. 空间分布特征
    unique_channels = len(set([ch for _, ch in burst['channel_spikes']]))
    features.append(unique_channels)

    # 3. 尖峰数量特征
    total_spikes = len(burst['channel_spikes'])
    features.append(total_spikes)

    # 4. 密度特征
    spike_density = total_spikes / duration if duration > 0 else 0
    features.append(spike_density)

    # 5. 空间中心
    spike_channels = [ch for _, ch in burst['channel_spikes']]
    rows = [ch // 8 for ch in spike_channels]
    cols = [ch % 8 for ch in spike_channels]
    row_center = np.mean(rows) if rows else 0
    col_center = np.mean(cols) if cols else 0
    features.extend([row_center, col_center])

    # 6. 空间范围
    row_range = np.max(rows) - np.min(rows) if rows else 0
    col_range = np.max(cols) - np.min(cols) if cols else 0
    features.extend([row_range, col_range])

    # 7. 波形幅度
    try:
        burst_center = (burst['start_idx'] + burst['end_idx']) // 2
        segment = data[max(0, burst_center - 20):min(data.shape[0], burst_center + 20), :]

        max_values = np.max(segment, axis=0)
        min_values = np.min(segment, axis=0)

        features.extend([
            np.min(max_values),
            np.max(max_values),
            np.mean(max_values),
            np.min(min_values),
            np.max(min_values),
            np.mean(min_values)
        ])
    except:
        features.extend([0] * 6)

    return features


def perform_burst_clustering(all_features):
    """执行爆发事件聚类"""
    if len(all_features) == 0:
        return [], None

    try:
        features_array = np.array(all_features)
        if features_array.shape[0] < 5:
            return [], None

        silhouette_scores = []
        possible_n_clusters = range(3, min(8, features_array.shape[0] + 1))

        for n in possible_n_clusters:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)
            try:
                score = silhouette_score(features_array, cluster_labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(-1)

        # 寻找最佳聚类数
        if len(silhouette_scores) > 0:
            best_idx = np.argmax(silhouette_scores)
            best_n_clusters = possible_n_clusters[best_idx]

            # 使用最优聚类数再次聚类
            final_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = final_kmeans.fit_predict(features_array)
            return cluster_labels, best_n_clusters

    except Exception as e:
        print(f"聚类错误: {str(e)}")

    return [], None


def analyze_file(csv_path, output_dir):
    """分析单个CSV文件"""
    file_name = os.path.basename(csv_path)
    file_tag = file_name.replace('.csv', '')
    print(f"处理文件: {file_name}")

    try:
        # 1. 读取数据
        df = pd.read_csv(csv_path)

        # 确定数据列 (跳过第0列，如果是时间列)
        use_cols = [c for c in df.columns if 'ch' in c.lower()][:64]
        if not use_cols:
            use_cols = df.columns[1:65]
        data = df[use_cols].values

        # 2. 计算阈值和检测爆发
        thresholds = calculate_thresholds(data)
        bursts = detect_global_burst_events(data, thresholds)
        print(f"  检测到爆发事件: {len(bursts)}")

        # 3. 处理每个爆发事件
        all_features = []
        burst_info = []

        for i, burst in enumerate(bursts):
            # 3.1 绘制波形图
            try:
                plot_file = plot_burst_waveforms(data, burst,
                                                 file_name=file_tag,
                                                 event_id=i + 1,
                                                 output_dir=burst_waveforms_dir)
                print(f"    保存波形图: {os.path.basename(plot_file)}")
            except Exception as e:
                print(f"    画图错误: {str(e)}")

            # 3.2 提取特征
            try:
                features = extract_burst_features(burst, data)
                all_features.append(features)

                # 保存爆发统计信息
                spike_channels = {ch for _, ch in burst['channel_spikes']}
                burst_info.append({
                    'file': file_tag,
                    'burst_id': i + 1,
                    'start_time': burst['start_idx'] * time_step,
                    'duration': burst['burst_duration'],
                    'spike_count': len(burst['channel_spikes']),
                    'channel_count': len(spike_channels),
                    'channels': ",".join(map(str, sorted(spike_channels)))
                })

            except Exception as e:
                print(f"    特征提取错误: {str(e)}")

        # 4. 保存爆发信息
        if burst_info:
            info_df = pd.DataFrame(burst_info)
            info_file = os.path.join(stats_output_dir, f"{file_tag}_bursts.csv")
            info_df.to_csv(info_file, index=False)
            print(f"  爆发统计保存至: {os.path.basename(info_file)}")

        # 5. 聚类分析
        cluster_labels, n_clusters = perform_burst_clustering(all_features)
        if n_clusters:
            print(f"  聚类完成: {n_clusters}个簇")
            if burst_info:
                for j, burst_item in enumerate(burst_info[:len(cluster_labels)]):
                    burst_item['cluster'] = cluster_labels[j]
                cluster_info_file = os.path.join(cluster_output_dir, f"{file_tag}_clusters.csv")
                info_df.to_csv(cluster_info_file, index=False)

        return len(bursts)

    except Exception as e:
        print(f"文件处理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


# ======================== 主函数 ========================
def main():
    print(f"开始爆发检测分析 (时间: {timestamp})")
    print(f"数据目录: {base_dir}")
    print(f"输出目录: {output_base_dir}")

    # 获取CSV文件
    csv_files = sorted([f for f in os.listdir(base_dir) if f.lower().endswith('.csv')])
    if not csv_files:
        print(f"未找到任何CSV文件在 {base_dir}")
        return

    print(f"\n即将处理 {len(csv_files)} 个文件:")
    for i, f in enumerate(csv_files):
        print(f"  {i + 1}. {f}")

    # 处理文件
    total_bursts = 0
    start_time = time.time()

    for i, file_name in enumerate(csv_files):
        print(f"\n{'=' * 60}")
        print(f"处理文件 {i + 1}/{len(csv_files)}: {file_name}")
        file_start = time.time()

        # 处理文件
        file_path = os.path.join(base_dir, file_name)
        num_bursts = analyze_file(file_path, output_base_dir)
        total_bursts += num_bursts

        # 计算剩余时间
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / (i + 1)
        remaining_time = avg_time * (len(csv_files) - i - 1)

        print(
            f"  耗时: {time.time() - file_start:.1f}秒 | 检测到爆发: {num_bursts} | 剩余时间: {remaining_time / 60:.1f}分钟")

        # 内存清理
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"分析完成! 总共处理 {len(csv_files)} 个文件")
    print(f"检测到 {total_bursts} 个爆发事件")
    print(f"总耗时: {(time.time() - start_time) / 60:.1f} 分钟")
    print(f"结果保存在: {output_base_dir}")
    print(f"波形图在: {burst_waveforms_dir}")
    print(f"爆发统计在: {stats_output_dir}")
    print(f"聚类结果在: {cluster_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

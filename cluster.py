import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ----------------- 基础设置 -----------------
base_dir = r'D:\!summer\ygx\data'
output_base_dir = r'D:\!summer\ygx\output\clustering_results'
os.makedirs(output_base_dir, exist_ok=True)

TARGET_GROUPS = range(17, 27)  # 017-026 文件需要聚类
CLUSTER_METHOD = 'gmm'  # 可选: 'kmeans', 'dbscan', 'gmm'

# ----------------- 获取 CSV 文件 -----------------
csv_files = sorted([f for f in os.listdir(base_dir) if f.lower().endswith('.csv') and f[:3].isdigit()],
                   key=lambda x: int(x.split('.')[0]))

# ----------------- 尖峰检测函数 -----------------
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    # 检查列名，去掉空格
    df.columns = [col.strip() for col in df.columns]

    # 所有通道列 (ch1_mV ~ ch64_mV)
    target_cols = [f'ch{i}_mV' for i in range(1, 65)]
    data = df[target_cols].values

    # 阈值: 中位数 + 5 * MAD
    median = np.median(data, axis=0)
    mad = 1.4826 * np.median(np.abs(data - median), axis=0)
    thresholds = median + 5 * mad
    return df, data, thresholds

def detect_spikes(data, thresholds, pre_samples=20, post_samples=50):  # 修改为峰前1ms(20点)，峰后2.5ms(50点)
    spikes = []
    for ch in range(data.shape[1]):
        crossings = np.where(data[:, ch] > thresholds[ch])[0]
        if len(crossings) > 0:
            peaks = [crossings[0]]
            for x in crossings[1:]:
                if x - peaks[-1] > 10:
                    peaks.append(x)
            for peak in peaks:
                start = max(0, peak - pre_samples)
                end = min(data.shape[0], peak + post_samples)
                waveform = data[start:end, ch]
                # 补零
                if len(waveform) < (pre_samples + post_samples):
                    pad_width = (pre_samples + post_samples) - len(waveform)
                    if peak - pre_samples < 0:
                        waveform = np.pad(waveform, (pad_width, 0), 'constant')
                    else:
                        waveform = np.pad(waveform, (0, pad_width), 'constant')
                spikes.append({
                    'channel': ch + 1,
                    'peak_index': peak,
                    'waveform': waveform,
                    'peak_amplitude': data[peak, ch]
                })
    return spikes

def extract_features(spikes):
    features = []
    for spike in spikes:
        wf = spike['waveform']
        features.append([
            np.max(wf),
            np.min(wf),
            np.max(wf) - np.min(wf),
            (np.argmin(wf) - np.argmax(wf)) * 0.05,
            np.trapz(np.abs(wf)),
            len(wf) * 0.05,
            wf[0] - wf[-1]
        ])
    return np.array(features)

def perform_clustering(features):
    X = StandardScaler().fit_transform(features)
    # GMM 自动选簇
    best_score = -1
    best_n = 2
    for n in range(2, 6):
        gmm = GaussianMixture(n_components=n, random_state=42)
        labels = gmm.fit_predict(X)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_n = n
    print(f"最佳簇数: {best_n} (轮廓系数: {best_score:.3f})")
    model = GaussianMixture(n_components=best_n, random_state=42)
    labels = model.fit_predict(X)
    return labels

def visualize_results(spikes, features, labels):
    unique_labels = np.unique(labels)
    # 平均波形
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    for cluster in unique_labels:
        cluster_waveforms = [spikes[i]['waveform'] for i in np.where(labels==cluster)[0]]
        avg_wave = np.mean(cluster_waveforms, axis=0)
        plt.plot(np.arange(len(avg_wave))*0.05, avg_wave, label=f'Cluster {cluster} (n={len(cluster_waveforms)})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Average Waveforms')
    plt.legend()
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(features))
    plt.subplot(1,2,2)
    for cluster in unique_labels:
        idx = np.where(labels==cluster)[0]
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f'Cluster {cluster}', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Projection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir,'clustering_017-026.png'), dpi=300)
    plt.close()

# ----------------- 主流程 -----------------
all_spikes = []  # 所有尖峰，用于保留文件来源
spike_features = []  # 聚类特征，仅017-026文件
spike_indices = []   # 对应索引，用于回写

for f in csv_files:
    file_num = int(f[:3])
    file_path = os.path.join(base_dir, f)
    print(f"Processing file {f}...")
    df, data, thresholds = load_and_preprocess(file_path)
    spikes = detect_spikes(data, thresholds)
    time_ms = df['time_ms'].values
    # 保存所有尖峰
    for s in spikes:
        s_copy = s.copy()
        s_copy['file_name'] = f
        s_copy['time_ms'] = time_ms[s['peak_index']]
        all_spikes.append(s_copy)
    # 如果是017-026文件，保存特征用于统一聚类
    if 17 <= file_num <= 26:
        feats = extract_features(spikes)
        spike_features.append(feats)
        spike_indices.extend([len(all_spikes)-len(spikes)+i for i in range(len(spikes))])

# 合并017-026特征
if spike_features:
    spike_features = np.vstack(spike_features)
    labels = perform_clustering(spike_features)
    # 回写聚类结果
    for idx, lbl in zip(spike_indices, labels):
        all_spikes[idx]['cluster'] = lbl

# 输出 CSV
results = []
for s in all_spikes:
    results.append({
        'file_name': s['file_name'],
        'time_ms': s['time_ms'],
        'channel': s['channel'],
        'cluster': s.get('cluster', -1),
        'peak_amplitude': s['peak_amplitude'],
        'waveform_length': len(s['waveform'])*0.05
    })
result_df = pd.DataFrame(results)
output_path = os.path.join(output_base_dir,'all_spikes_clustered.csv')
result_df.to_csv(output_path,index=False)
print(f"所有结果已保存到 {output_path}")

# 可视化017-026聚类
if spike_features.size > 0:
    visualize_results([all_spikes[i] for i in spike_indices], spike_features, labels)
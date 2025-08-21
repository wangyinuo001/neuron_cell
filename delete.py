import os
import pandas as pd

# 设置工作目录
directory = r'D:\!summer\ygx\data'

# 循环处理001-031共31个文件
for i in range(1, 32):
    # 生成文件名，如001.csv, 002.csv,...031.csv
    filename = f"{i:03d}.csv"
    filepath = os.path.join(directory, filename)

    # 检查文件是否存在
    if os.path.exists(filepath):
        try:
            # 读取CSV文件，跳过前3行
            df = pd.read_csv(filepath, skiprows=3)

            # 保存回原文件，不保留索引，不保留原列名(因为被跳过了)
            df.to_csv(filepath, index=False, header=True)

            print(f"已处理文件: {filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    else:
        print(f"文件不存在: {filename}")

print("所有文件处理完成！")
# SEEG Manifold & Symmetry Analysis Toolkit

## 项目概述

这是一个用于分析立体脑电图(SEEG/iEEG)数据的Python工具包，专注于：

1. **流形学习与降维** - 从高维神经数据中发现低维结构
2. **表示相似性分析(RSA)** - 分析神经表征的几何结构
3. **拓扑数据分析(TDA)** - 发现数据的拓扑特征
4. **对称性分析** - 探索神经表征中的群结构和对称性

## 理论背景

本工具包基于以下核心概念：

- **神经流形假说**：神经群体活动存在于高维空间的低维流形上
- **群论视角**：神经表征可能携带某些群的表示，反映对刺激变换的系统性响应
- **对称性与不变性**：大脑编码可能具有特定的对称结构

## 项目结构

```
seeg_manifold_analysis/
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
├── environment.yml              # Conda环境配置
├── setup.py                     # 安装配置
│
├── config/                      # 配置文件
│   └── default_config.yaml      # 默认参数配置
│
├── src/                         # 源代码
│   ├── __init__.py
│   ├── io/                      # 数据输入输出
│   │   ├── __init__.py
│   │   ├── loaders.py           # 数据加载器
│   │   └── converters.py        # 格式转换
│   │
│   ├── preprocessing/           # 预处理模块
│   │   ├── __init__.py
│   │   ├── filters.py           # 滤波
│   │   ├── artifact_removal.py  # 伪迹去除
│   │   └── epoching.py          # 分段
│   │
│   ├── features/                # 特征提取
│   │   ├── __init__.py
│   │   ├── spectral.py          # 频谱特征
│   │   ├── connectivity.py      # 功能连接
│   │   └── time_frequency.py    # 时频分析
│   │
│   ├── manifold/                # 流形学习
│   │   ├── __init__.py
│   │   ├── dimensionality.py    # 维度估计
│   │   ├── reduction.py         # 降维方法
│   │   └── comparison.py        # 多方法比较
│   │
│   ├── rsa/                     # 表示相似性分析
│   │   ├── __init__.py
│   │   ├── rdm.py               # 表示差异矩阵
│   │   └── comparison.py        # RSA比较
│   │
│   ├── topology/                # 拓扑数据分析
│   │   ├── __init__.py
│   │   ├── persistent_homology.py
│   │   └── visualization.py
│   │
│   ├── symmetry/                # 对称性分析
│   │   ├── __init__.py
│   │   ├── detection.py         # 对称性检测
│   │   ├── groups.py            # 群结构分析
│   │   └── equivariance.py      # 等变性检验
│   │
│   └── visualization/           # 可视化
│       ├── __init__.py
│       ├── manifold_plots.py    # 流形可视化
│       ├── tda_plots.py         # 拓扑可视化
│       └── interactive.py       # 交互式可视化
│
├── notebooks/                   # Jupyter笔记本
│   ├── 01_data_loading.ipynb    # 数据加载示例
│   ├── 02_preprocessing.ipynb   # 预处理流程
│   ├── 03_manifold_analysis.ipynb    # 流形分析
│   ├── 04_rsa_analysis.ipynb    # RSA分析
│   ├── 05_topology_analysis.ipynb    # 拓扑分析
│   └── 06_symmetry_analysis.ipynb    # 对称性分析
│
├── tests/                       # 单元测试
│   └── ...
│
└── examples/                    # 示例数据和脚本
    └── sample_analysis.py
```

## 安装

### 方法1：使用 Conda (推荐)

```bash
# 克隆项目
git clone <your-repo-url>
cd seeg_manifold_analysis

# 创建并激活环境
conda env create -f environment.yml
conda activate seeg-manifold

# 安装项目
pip install -e .
```

### 方法2：使用 pip

```bash
pip install -r requirements.txt
pip install -e .
```

## 快速开始

```python
from src.io import load_seeg_data
from src.preprocessing import preprocess_pipeline
from src.manifold import compare_reductions
from src.symmetry import detect_symmetry

# 1. 加载数据
data = load_seeg_data('your_data.mat')

# 2. 预处理
processed = preprocess_pipeline(data, 
                                 lowcut=1, 
                                 highcut=150,
                                 notch_freq=50)

# 3. 多方法降维比较
embeddings = compare_reductions(processed, 
                                 methods=['pca', 'umap', 'isomap'],
                                 n_components=3)

# 4. 对称性分析
symmetry_results = detect_symmetry(embeddings['umap'])
```

## 数据格式

支持的输入格式：
- `.mat` (MATLAB文件) - **推荐**，直接从MATLAB导出
- `.edf` (European Data Format)
- `.fif` (MNE-Python格式)

### 从MATLAB导出数据

在MATLAB中运行：

```matlab
% 假设你的数据在变量 data 中
% data 应该是 (n_channels x n_timepoints) 或 (n_epochs x n_channels x n_timepoints)

seeg_data = struct();
seeg_data.data = data;           % 必需：原始数据
seeg_data.sfreq = 1000;          % 必需：采样率
seeg_data.ch_names = ch_names;   % 推荐：通道名称 (cell array)
seeg_data.times = times;         % 可选：时间向量

save('seeg_data.mat', 'seeg_data', '-v7.3');
```

## 核心概念

### 1. 流形学习

神经数据虽然是高维的（N个通道），但有意义的变化可能只在少数几个维度上。我们使用多种方法来发现这个低维结构：

- **PCA**：线性降维，找最大方差方向
- **UMAP**：保持拓扑结构的非线性降维
- **Isomap**：保持测地距离
- **t-SNE**：保持局部邻域结构

### 2. 对称性分析

如果神经表征具有某种对称性（如旋转不变性），这可能反映了大脑的编码原理。我们可以：

- 检测数据中是否存在周期性结构（对应SO(2)或Z_n群）
- 分析不同条件/状态之间的交换对称性
- 研究动力学的时间平移不变性

### 3. 拓扑数据分析

使用持续同调(Persistent Homology)来发现数据的拓扑特征：

- 0维特征：连通分量（聚类）
- 1维特征：环/洞
- 2维特征：空腔

## 许可证

MIT License

## 作者

[Your Name]

## 引用

如果你使用了这个工具包，请引用：

```bibtex
@software{seeg_manifold_analysis,
  author = {Your Name},
  title = {SEEG Manifold & Symmetry Analysis Toolkit},
  year = {2024},
  url = {your-repo-url}
}
```

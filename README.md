# Frigate: Frugal Spatio-temporal Forecasting on Road Networks

### 简言

这个项目是SZU机器学习初步的期末大作业，要求是复现一篇近五年跟机器学习有关的论文

作者选取的是2023年一篇发表在KDD上的论文 Frigate: Frugal Spatio-temporal Forecasting on Road Networks， 这篇论文结合了LSTM与GNN的架构，具体细节见论文部分

### 复现环境
- Python: 3.12.0
- PyTorch: 2.2.2 (CUDA 11.8)
- PyTorch Geometric: 2.2.0
- Numpy: 1.26.4
- Pandas: 2.2.3
- SciPy: 1.13.1
- NetworkX: 3.4.2

### 代码结构
```bash
MyFrigate
├── data
│   ├── Beijing
│   ├── Chengdu
│   └── Harbin
├── logs
├── model
│   ├── model.py
│   ├── tester.py
│   └── trainer.py
├── outputs
│   ├── models
│   ├── predictions
│   └── tensorboard
├── run.sh
├── run_test.sh
├── test.py
├── train.py
└── utils
    ├── data_utils.py
    └── test_data_utils.py
```

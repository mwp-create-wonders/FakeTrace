from pathlib import Path
project_root = Path(__file__).parent

# 设置训练数据集的查找路径
dataset_paths = {
    # Specify where are the roots of the datasets.
    'FR'       : '/workspace/user-data/models/TruFor/dataset/FantasticReality/dataset',
    'IMD'      : '/workspace/user-data/models/TruFor/dataset/IMD',
    'CA'       : '/workspace/user-data/models/TruFor/dataset/CASIAv2',
    'tampCOCO' : '/workspace/user-data/models/TruFor/dataset/TampCOCO',
    'compRAISE': '/workspace/user-data/models/TruFor/dataset/compRAISE/compRAISE',
}

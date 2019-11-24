python-bhtsne-mod
=================

# 概要

[python-bhtsne](https://github.com/dominiek/python-bhtsne)に以下の改変を実施したもの．

- `max_iter`パラメータを指定できるようライブラリAPIを改変
- scikit-learn準拠のAPIをもつラッパークラス`bhtsne_interface.py`を作成

# インストール手順

## bh_tsne C++モジュールのビルド

```bash
cd python-bhtsne/src
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```

## python-bhtsneのインストール

```bash
cd python-bhtsne
python setup.py build
python setup.py install
```

環境によってはsudoで実行する必要あり．

## bhtsne_interfaceのインポート

```python
from bhtsne_interface import BHTSNE
```

# 使用方法

```python
model = BHTSNE(n_components=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=1000)
Y = model.fit_transform(X)
```

詳細なexampleは`test_mnist.py`を参照されたい．

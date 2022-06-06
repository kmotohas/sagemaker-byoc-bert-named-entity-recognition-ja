# sagemaker-byoc-bert-named-entity-recognition-ja

@AtsunoriFujita さんのBERTによる固有表現抽出のサンプルコードをBring Your Own Container (BYOC)としてSageMakerで動かせるようにしたものです。

![](https://user-images.githubusercontent.com/40932835/124863920-ef039080-dff2-11eb-9ac6-de26f9d6d8e5.png)

## 使い方

> **Warning**
> 基本的にはバージニア北部 (us-east-1) リージョンで作業していることを前提としているので気をつけてください。

まず、`NER_BIO.ipynb` を実行して `ner-wikipedia-dataset/ner.json` をS3にアップロードしてください。

そして、SageMakerのHuggingFace built-in containerをベースとしてDockerイメージを作成します。

```bash
bash docker/build_training_container.sh  # リージョンによって中のbase変数の値が異なります
bash docker/build_inference_container.sh  # リージョンによって中のbase変数の値が異なります
```

あとは、前処理、学習、バッチ推論の順でコードを実行してみてください。
hydraで設定ファイルを管理しているので適宜 `tests/config` 以下のファイルを書き換えてください。

```bash
pip install -r scripts/requirements.txt  # 依存関係のインストール
python tests/test_preprocess.py
python tests/test_train.py
python tests/test_inference.py
```

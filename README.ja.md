# 日本語の選択肢型意思決定モデル（Jev-inspired）

[English](README.md) | 日本語

状態・質問・選択肢を入力し、各選択肢の確率を返す日本語モデルを、JAX/Flaxで事前学習から構築する研究プロジェクトです。[TypeSafe AIのJev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)から着想を得ていますが、**Jevの公式実装や再現ではありません**。

## モデルと学習の概要

- 日本語の自己回帰言語モデルを事前学習し、その重みを選択肢型の意思決定タスクに転用します。
- アーキテクチャとしては [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245)、Sparse MoE、重みを共有するloopブロックを使用します。行列重みの更新には [Muon](https://github.com/KellerJordan/Muon)、その他にはAdamWを使用します。
- 意思決定時は、各選択肢の token representation を平均し、最後の判断 token の representation との内積でスコアを出します。選択肢ごとに固定された分類器の重みは持ちません。現在の設定では選択肢は最大6個です。
- 構造化された入力の context をそのまま用い、入力された選択肢だけを同時にスコアリングします。自由な文章生成を行わないため、候補外の出力は生成しません。

アーキテクチャの選定については作者の趣味も含まれています。

設定は [`configs/`](configs/) にあります。
既定の依存関係は CUDA 13 対応の JAX を指定しています。実行環境に合わせて `pyproject.toml` の JAX 設定を確認してください。

## 1. 事前学習

事前学習には [FineWeb2 Edu Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) の `sample_10BT` 構成と、[LLM-jp 13B v1.0](https://huggingface.co/llm-jp/llm-jp-13b-v1.0) のトークナイザーを使用します。後者は**トークナイザーのみ**を使用し、公開モデルの重みは読み込みません。

```bash
uv run pretrain.py
```

50,000ステップ終了時のログ（[`configs/pretrain.yaml`](configs/pretrain.yaml) の設定）:

| 指標 | 値 |
| --- | ---: |
| Train loss（step 49,950） | 2.7633 |
| Validation loss（step 49,999） | 3.0324 |
| Validation perplexity | 20.75 |

事前学習における train / validation loss の推移:

![Pretraining train and validation loss](figures/training_trajectories.png)

Train lossにはMoEの補助損失を含みます。Validation lossはLM lossのみです。

## 2. 合成シナリオの生成と判定

ローカルLLMで日本語の状態・質問・選択肢を生成し、別の判定パスで選択肢の目標分布を作ります。生成と判定（LLM-as-a-Judge）の既定モデルは `gemma4:26b` です。利用可能なモデル名や接続先URLなどは各YAMLを編集してください。

```bash
uv run generate_synthetic.py
uv run judge_synthetic.py
```

- 生成設定: [`configs/synthetic.yaml`](configs/synthetic.yaml)。既存の `raw.jsonl` を読み、`target_count` を合計目標件数として追加生成します。
- 判定設定: [`configs/judge_synthetic.yaml`](configs/judge_synthetic.yaml)。既に判定済みのシナリオはIDでスキップします。


## 3. 意思決定コーパスの構築

判定済み合成データに加え、[JGLUE](https://github.com/yahoojapan/JGLUE) のJNLI・JCommonsenseQA・JSTSを学習と検証に使用します。[JCoLA](https://github.com/osekilab/JCoLA) の検証データは転移評価に使用します。各データセットの説明は [JGLUE論文](https://aclanthology.org/2022.lrec-1.317/) および [JCoLA論文](https://aclanthology.org/2024.lrec-main.828/) を参照してください。

```bash
uv run build_decision_corpus.py
```

スクリプトは必要な公開リポジトリがローカルにない場合にcloneし、`data/decision_corpus/` に学習・評価用JSONLを作成します。

JGLUEとJCoLAはそれぞれ **CC BY-SA 4.0** で公開されています。元データの利用・公開時は、それぞれのライセンスを確認してください。

## 4. 意思決定モデルの学習

事前学習済みモデルに汎用の選択肢スコアリングヘッドを追加して学習します。各種実験設定は [`configs/decision.yaml`](configs/decision.yaml) で指定します。実行前に `pretrained_checkpoint` が存在することを確認してください。

```bash
uv run train_decision.py
```

検証時にチェックポイントを保存し、直近2件と、NLLが最も低い `best` を保持します。


## 5. 結果とデモ

### 評価結果

評価時の `best` は、すべての評価データをまとめた validation NLL が最小の checkpoint です。

| Checkpoint | Step | 全評価データのAccuracy |
| --- | ---: | ---: |
| `best` | 3,042 | 70.50% |

| Dataset | Examples | Accuracy |
| --- | ---: | ---: |
| JNLI | 2,434 | 81.47% |
| JCommonsenseQA | 1,119 | 69.44% |
| JSTS | 1,457 | 51.89% |
| Synthetic decisions | 792 | 56.82% |
| JCoLA | 1,550 | 78.52% |

`stream_decision_eval.py` で初回コンパイル後のモデル推論だけを測定した場合、平均 latency は約 `10ms/件`、約100件/秒です。tokenize、初回コンパイル、画面表示は含みません。

### 対話デモ

状態・質問・選択肢を入力し、各選択肢の予測確率を表示します。

```bash
uv run interactive_decision.py
```

別の保存済みモデルを使う場合は `--checkpoint <チェックポイントのパス>` を指定します。

下記は出力例です。
```
STATE
(Finish with an empty line)
> 天気予報によると雨が降ってくるようだ
>

QUESTION> 持っていくべきものはなんですか

OPTIONS
Enter one option per line.
Empty line finishes options.
1> 折りたたみ傘
2> パソコン
3> スマホ
4>

================
 > 1.  83.67% 折りたたみ傘
   2.   2.66% パソコン
   3.  13.67% スマホ
model latency: 10.14 ms

```

### 評価例と推論速度

既定ではJCommonsenseQAの検証例を20件表示します。各入力、予測確率、正解、推論時間を順に表示し、最後に正解率と平均速度を出します。

```bash
uv run stream_decision_eval.py
```

全件表示は `--limit 0`、合成データの場合は `--data data/decision_corpus/eval_synthetic.jsonl --source synthetic_decisions` を使用します。速度は固定長の1件推論のみです。

## 参考

- [TypeSafe AI: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [FineWeb2 Edu Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese)
- [LLM-jp 13B v1.0](https://huggingface.co/llm-jp/llm-jp-13b-v1.0)
- [JGLUE](https://aclanthology.org/2022.lrec-1.317/) / [JCoLA](https://aclanthology.org/2024.lrec-main.828/)
- [GQA](https://arxiv.org/abs/2305.13245) / [Muon](https://github.com/KellerJordan/Muon)
- [Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165)

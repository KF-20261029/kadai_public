# SO-101とπ0.5によるペン収納

SO-101で机上のペン３本をペン立てに収納する、ROS 2／MuJoCoのシミュレーション環境です。上方・手首カメラの画像と６関節の状態をπ0.5に入力し、予測した位置指令を30 Hzで実行します。タスクと教師データはCodexを用いて作成しました。

## 実験結果

SO-101用の `dongyoonkim/so101-pi05-base` を基に、成功デモ55試行・99,000フレームでLoRAによる模倣学習を行いました。同じ10種類の初期配置で評価した結果は次のとおりです。

| モデル | 推論中のシミュレーション | 成功数 |
| --- | --- | --- |
| ベースモデル | 一時停止 | 10回中0回成功 |
| 追加学習後 | 一時停止 | 10回中2回成功 |
| 追加学習後 | 継続 | 10回中1回成功 |

各試行はシミュレータ内で約75秒です。３本すべてを収納し、グリッパーから離れた状態を２秒維持した場合を成功としました。入出力の正規化には、両モデルとも同じタスク用統計を使いました。評価対象はこの10配置で、実機評価は行っていません。

[提出用PDF](docs/report.pdf)と[試行別の集計結果](docs/results)を収録しています。

## システム構成

物理計算にはMuJoCo、観測と指令の受け渡しにはROS 2を使います。画像・関節状態の変換、π0.5との接続、シーン・起動設定、評価処理をタスク用に実装しました。軌道制御にはROS 2の `joint_trajectory_controller`、観測の組み合わせには `message_filters` を使用しています。

| ディレクトリ | 内容 |
| --- | --- |
| `simulation/`、`description/` | ロボット形状、ペン・ペン立て、ROS 2制御インターフェース |
| `config/`、`launch/` | カメラ・関節制御・初期配置と起動設定 |
| `ros2_ws/src/so101_policy/` | 観測取得、モデル読み込み、予測、関節指令の実行 |
| `models/pen_lora/` | ペン収納用LoRAと正規化統計 |
| `scripts/` | モデル取得、試行の実行・記録、成否判定、集計 |
| `teacher/` | 逆運動学による教師データ生成・選別・形式変換 |
| `training/` | LoRAの学習環境・設定・起動スクリプト |
| `docs/` | 提出用PDF、実験結果 |

学習したLoRAと検証・評価の記録を収録しています。教師データ本体とROS bagは含めていません。

## データセット

教師データは、MuJoCo上で逆運動学による把持・収納を実行して作成しました。生成・選別・形式変換のコードは `teacher/` に収録しています。

- **学習済みのLoRAを動かす場合**：[セットアップ](#セットアップ)と[評価](#評価)を実行します。
- **再学習する場合**：[教師データの生成](teacher/README.md)、[LoRAの学習](training/README.md)の順に実行します。

### データ形式

当初はLeRobot v3形式で教師データを収集していましたが、その後、v2.1形式を前提とする学習コードを採用したため、収集済みのデータを変換しました。完成したデータの保存先は `teacher/outputs/pen_dataset/`、学習設定でのIDは `local/so101_pens` です。１試行は60秒・1,800フレーム（30 Hz）で、学習には55試行を使います。

学習用データのフィールドは次のとおりです（`meta/info.json`）。

| フィールド | 型 | 形状 | 内容 |
| --- | --- | --- | --- |
| `observation.images.base_0_rgb` | video | (480, 640, 3) | 上方カメラのRGB画像（高さ・幅・チャンネル） |
| `observation.images.left_wrist_0_rgb` | video | (480, 640, 3) | 手首カメラのRGB画像（高さ・幅・チャンネル） |
| `observation.state` | float32 | (6,) | 観測した関節位置 |
| `action` | float32 | (6,) | 実行する関節位置指令 |
| `timestamp` | float32 | (1,) | 試行開始からの時刻［秒］ |
| `frame_index` | int64 | (1,) | 試行内のフレーム番号 |
| `episode_index` | int64 | (1,) | 試行番号 |
| `index` | int64 | (1,) | データセット全体の行番号 |
| `task_index` | int64 | (1,) | タスク指示の参照番号 |

`observation.state` と `action` の要素は、次の順序です。最初の５要素は関節角度［度］、最後はグリッパーの関節可動範囲を0〜100に換算した開度［%］です。

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

画像はH.264のMP4、状態・指令・各番号はParquetに保存します。収集時の画像名 `observation.images.top` と `observation.images.wrist` は、学習モデルの入力名に合わせ、それぞれ `observation.images.base_0_rgb` と `observation.images.left_wrist_0_rgb` に変更しています。

タスク指示は `meta/tasks.jsonl` に保存し、各フレームの `task_index: 0` から次の文字列を参照します。

```text
Grab pens and place into pen holder
```

各フレームの `action` は６要素の指令１個です。学習時には連続する50個の指令をアクションチャンクとして扱い、評価時には予測した50個すべてを30 Hzで実行します。

### 訓練・検証データの分割

訓練用49試行・検証用６試行に分ける設定を作成しましたが、実際の読み込みには分割が反映されず、両方に全55試行が読み込まれました。そのため、独立した検証データによる学習時評価は得られていません。学習後の収納性能は、教師データ収集とは別のseedを使った10配置で評価しました。

## 動作環境

- Linux x86_64
- NVIDIA GPUと、CUDA 12.6版PyTorchに対応するドライバ
- Docker Engine、Docker Compose、NVIDIA Container Toolkit
- Git LFS

動作確認にはNVIDIA RTX 6000 Ada Generation（48 GB）を使用しました。コンテナの既定UID/GIDは1000で、作業ディレクトリへの読み書き権限が必要です。ベースモデルは約7.5 GBあり、別途Dockerイメージ・依存パッケージ・実行結果の保存領域も使用します。

## セットアップ

以下のホスト側コマンドは、クローンしたリポジトリの直下で実行します。

### 重みの取得

```bash
git lfs pull
```

ペン収納用LoRAはGit LFSで管理しています。取得後の `models/pen_lora/lora_adapters/adapter_model.safetensors` は約118 MiBです。

### Dockerイメージとベースモデルの準備

```bash
docker compose build
docker compose run --rm ros2 python scripts/download_model.py
```

ベースモデルとトークナイザは公開元から取得し、Composeの専用ボリュームに保存します。リビジョンとSHA-256は `config/model_assets.json` に固定しています。

### デスクトップの起動とROSパッケージのビルド

ホストで実行します。

```bash
docker compose up -d --wait
docker compose exec ros2 bash
```

続いて、入ったコンテナ内で実行します。

```bash
cd /workspace/ros2_ws
python /usr/bin/colcon build --packages-select so101_policy
source install/setup.bash
```

noVNCの接続先は次のとおりです。

```text
http://localhost:6080/vnc.html?autoconnect=1&resize=scale
```

遠隔ホストの場合は、自分の接続先に対してSSHポート転送を行います。

```bash
ssh -N -L 6080:127.0.0.1:6080 USER@HOST
```

## 実行方法

noVNC内のTerminatorでシミュレータを起動します。

```bash
source /opt/mujoco337/control/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch /workspace/launch/so101.launch.py controller:=trajectory_controller
```

別のターミナルで画像ビューアを開き、`/mujoco_camera_plugin/overview/color` を選択します。

```bash
rqt_image_view
```

評価コマンドは、シミュレータを起動したまま別のターミナルで実行します。初期配置の復元と停止・再開は評価スクリプトが行います。

## 評価

### ベースモデルと追加学習後の比較

コンテナ内で実行します。

```bash
cd /workspace
bash scripts/run_evaluation.sh base
bash scripts/run_evaluation.sh lora
python scripts/summarize_evaluation.py outputs/lora_comparison
```

両モデルを同じ10配置で評価します。各ペンの初期位置をXY方向に±12 mm、向きを±0.4 rad変化させ、MuJoCo 3.3.7で静定させた配置を使用します。初期配置と推論seed（7100〜7109）は両モデルで共有します。推論中はシミュレーションを止め、50指令すべてを30 Hzで実行するたびに再観測・再予測します。

### 推論中もシミュレーションを進める場合

同じコンテナ内で実行します。

```bash
for seed in {7100..7109}; do
  bash scripts/run_realtime_trial.sh "$seed" || break
done
python scripts/summarize_realtime.py
```

追加学習後の重みを使い、チャンク終了の0.35秒前から次の観測・推論を始めます。次の指令は前のチャンクが終了してから送ります。

### 出力ファイル

| 場所 | 内容 |
| --- | --- |
| `outputs/lora_comparison/summary.json` | ベースモデルと追加学習後の集計 |
| `outputs/realtime_comparison/summary.json` | シミュレーション停止・継続条件の集計 |
| 各出力先の `seed_*.score.json` | 試行全体の成否と、終了時の各ペンの収納・開放状態 |
| 各出力先の `seed_*` ディレクトリ | ROS bag。継続条件では `metrics.json` も保存 |
| 各出力先の `seed_*.run.log` | モデル読み込みと実行のログ |

成功判定には、ROS bagに保存したペン姿勢と関節状態を使用します。MuJoCo 3.3.7上で収納位置と指との接触を確認し、３本すべてが収納され、両側の指から離れた状態が２秒以上続いた試行を成功とします。終了時の状態も別に記録し、状態記録が不足している試行は集計を中断します。

評価スクリプトは既存の結果がある保存先では停止します。保存先は `EVAL_OUTPUT`（一時停止時）と `REALTIME_OUTPUT`（継続時）で指定します。

```bash
export EVAL_OUTPUT=/workspace/outputs/repeat_paused
export REALTIME_OUTPUT=/workspace/outputs/repeat_continuous
```

上記の評価コマンドを実行した後、指定した保存先を集計します。

```bash
python scripts/summarize_evaluation.py "$EVAL_OUTPUT"
python scripts/summarize_realtime.py --paused "$EVAL_OUTPUT/lora" --continuous "$REALTIME_OUTPUT"
```

記録したROS bagを動画で確認するには、コンテナ内で次を実行します。

```bash
python scripts/bag_to_video.py outputs/lora_comparison/lora/seed_7100 outputs/trial_review.mp4
```

## 検証記録

実験時の検査・学習・評価の記録を `docs/results/` に収録しています。

| 対象 | 記録 |
| --- | --- |
| 教師データ | [収集後の検査](docs/results/collection_validation.json)、[形式変換後の検査](docs/results/dataset_validation.json) |
| 学習 | [lossの推移と学習記録](training/README.md#学習結果) |
| データ・モデルの対応 | [学習条件・データ選別記録・重みのハッシュ](docs/results/training_run.json) |
| 収納性能 | [モデル間の比較](docs/results/lora_comparison.json)、[推論中の停止・継続の比較](docs/results/continuous_comparison.json) |

## CI

GitHub Actionsで、push・プルリクエスト時に次の検査を自動実行します（[設定](.github/workflows/ci.yml)）。

- Pythonの構文と、JSON・YAML・TOML・XMLの形式検査。
- 30 Hzの指令時刻と、評価結果の集計に関する単体テスト。
- 推論・学習・教師データ収集用のDocker Compose設定の検証。

## ライセンス

タスク用コードは [Apache-2.0](LICENSE)、モデルは [Gemma規約](MODEL_LICENSE.md) に従います。ロボット形状と公開パッケージの出典・変更内容は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に記載しています。

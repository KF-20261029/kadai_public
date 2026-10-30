# 教師データの生成

この手順では、同梱の生成コードを実行し、LoRA学習用の教師データを作成します。データ本体は配布していません。生成コードはCodexを用いて作成しました。

収集、選別・検証、形式変換の順に進めると、LeRobot v2.1形式のデータが `teacher/outputs/pen_dataset/` に保存されます。フィールドの説明は[ルートREADME](../README.md#データ形式)を参照してください。

## 生成方法

`teacher/scripts/collect_demonstrations.py` の `Teacher.solve()` は、SciPyの `least_squares` を使い、５関節の角度を求めます。目標はグリッパーの把持中心の位置・向きで、関節範囲を制約にしています。グリッパーの開閉は別に指定します。

目標位置・向きはMuJoCo内のペン姿勢とペン立て位置から計算し、接近、下降、把持、持ち上げ、向き変更、移送、挿入、開放、退避の順に動かします。移動時には関節指令を滑らかにつなぎ、30 Hzで実行します。

上方・手首カメラのRGB画像、６関節の状態、関節指令、タスク指示を保存します。物体位置は教師の指令生成に使います。学習後のπ0.5は、画像・関節状態・タスク指示を入力として動作します。

## 実験に使用したデータ

本実験では、seed 2000〜2077の78試行中56試行が成功しました。その後、手首映像が１フレーム欠けた１試行を除外し、55試行・99,000フレームを学習用としました。各試行は60秒・1,800フレームです。初期位置はXY方向に±0.012 m、向きは±0.4 rad変化させました。

実験時の収集・選別記録と学習用データのメタデータは `teacher/provenance/` に収録しています。検査結果は[収集後](../docs/results/collection_validation.json)と[形式変換後](../docs/results/dataset_validation.json)の両方で保存されています。以下では同じ条件で収集し、映像欠損のない成功試行を55試行選びます。

## スクリプト一覧

| ファイル | 処理 |
| --- | --- |
| `collect_demonstrations.py` | 逆運動学、把持・収納の段階制御、画像・状態・指令の収集 |
| `simulation.py` | MuJoCoの読み込み、初期配置の変更、関節単位変換、物理計算、カメラ描画、収納判定 |
| `curate_demonstrations.py` | カメラ区間が不完全な試行を検出し、LeRobot標準APIで除外 |
| `validate_demonstrations.py` | フレーム数・指令値・収納成否・映像の整合性を確認 |
| `export_demonstration_videos.py` | 記録の動画出力と確認用の一覧作成 |
| `prepare_pen_dataset.py` | 公開の変換器を使いLeRobot v3から学習用v2.1へ変換し、カメラ名を対応付け |
| `repair_pen_videos.py` | 元映像のフレーム区間からカメラ映像を再出力し、変換時の時刻ずれを補正 |
| `validate_pen_dataset.py` | 変換後のフレーム数・時刻・カメラ名などを確認 |

## セットアップ

収集には専用のCompose・Pixi環境を使用します。MuJoCo 3.3.7とLeRobotを含む依存は `teacher/pixi.lock` に固定しています。

以下のDockerコマンドは、すべてホスト側のリポジトリ直下で実行します。ルートREADMEのセットアップで、ROS用Dockerイメージもビルドしておいてください。

収集コードは１試行分の３カメラ画像をメモリに保持するため、60秒分の画像だけで約5 GBに加え、エンコードなどの作業領域を使います。

```bash
docker compose -f teacher/compose.yaml build
docker compose -f teacher/compose.yaml run --rm teacher pixi install --locked
```

## データ収集

出力は `teacher/outputs/demonstrations_expanded` に保存します。

```bash
docker compose -f teacher/compose.yaml run --rm teacher pixi run python scripts/collect_demonstrations.py \
  --episodes 78 --seed 2000 --position-jitter 0.012 --yaw-jitter 0.4 \
  --output outputs/demonstrations_expanded
```

## データの選別と検証

成功試行から、カメラ映像が完全な55試行を残し、フレーム数・指令値・収納成否を検証します。

```bash
docker compose -f teacher/compose.yaml run --rm teacher pixi run python scripts/curate_demonstrations.py \
  --input outputs/demonstrations_expanded --max-episodes 55

docker compose -f teacher/compose.yaml run --rm teacher pixi run python scripts/validate_demonstrations.py \
  --input outputs/demonstrations_expanded \
  --dataset-root outputs/demonstrations_expanded/training_dataset \
  --evaluation-seeds 7100 7101 7102 7103 7104 7105 7106 7107 7108 7109
```

上の `--max-episodes 55` は、完全な成功試行を元の順に55試行残します。余剰があればLeRobot標準APIで除外し、不足していれば出力を作る前に停止します。不足した場合は、試行数を増やし、別の出力先に収集し直してください。その場合、後続の入力パスも新しい出力先に合わせます。

## データ形式の変換

選別した55試行を、学習環境が使用するLeRobot v2.1形式へ変換します。`prepare_pen_dataset.py` は55試行を前提としており、異なる件数なら停止します。変換器は公開リポジトリから固定リビジョンを取得します。

次の３コマンドはホストで実行し、処理にはROS用コンテナを使います。変換、映像の再出力、検証の順に進めます。

```bash
docker compose run --rm ros2 python teacher/scripts/prepare_pen_dataset.py \
  --source /workspace/teacher/outputs/demonstrations_expanded/training_dataset

docker compose run --rm ros2 python teacher/scripts/repair_pen_videos.py \
  --source /workspace/teacher/outputs/demonstrations_expanded/training_dataset \
  --output /workspace/teacher/outputs/pen_dataset

docker compose run --rm ros2 python teacher/scripts/validate_pen_dataset.py
```

生成後と変換後の検査結果は、それぞれ `teacher/outputs/demonstrations_expanded/validation.json` と `teacher/outputs/pen_dataset/validation.json` に保存されます。

## 学習

変換・検証後は、[LoRAの学習手順](../training/README.md)に従って学習します。

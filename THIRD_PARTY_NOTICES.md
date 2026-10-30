# 出典・変更内容・ライセンス

## ロボット形状

- URDFの出典：[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/blob/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101/so101_new_calib.urdf)。リビジョンは `a758567c3978dfeefe282ede0500085a48fe8f78`。Onshape CADからonshape-to-robotで生成されたものです。
- MJCF・メッシュの出典：[MuJoCo Menagerie / robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/d7b481bac63d3d600ab404ee74f7474c5efe8a67/robotstudio_so101)。リビジョンは `d7b481bac63d3d600ab404ee74f7474c5efe8a67`。
- ライセンスは `simulation/assets/so101/LICENSE`、取得元のリビジョンは `simulation/assets/so101/UPSTREAM_COMMIT` に記録しています。実行に使用するモデルとメッシュを同梱しています。
- 本タスクでは、メッシュ参照、色、座標関係、エンドエフェクタのフレーム、ROS 2制御インターフェースを調整し、ペンとペン立てを含むシーンを追加しました。

## Docker内で取得する公開プログラム

Dockerfileは次のソースを固定したリビジョンから取得します。

| プログラム | 取得先・固定リビジョン |
| --- | --- |
| VLASH | https://github.com/mit-han-lab/vlash — `22cbabfee0f57874987c75a35a7dac129e695db0` |
| SO-101向けの修正 | https://github.com/jinnymo/so101-pi05-base — `73167f4dba88633aeec95ba7a3b3881916fa9aaa` |
| mujoco_vendor | https://github.com/pal-robotics/mujoco_vendor — `e4c679d922fdf977be8625fd770020b955d8cc62` |
| mujoco_ros2_control | https://github.com/ros-controls/mujoco_ros2_control — `d5b1f019dc3b33f3d0270450506bd6e72bea1913` |

VLASHには、SO-101向けリポジトリの `modeling_pi05.py`、`run.py`、`train.py` を適用しています。また、LeRobotの `video_utils.py` を同リポジトリの版に置き換えています。mujoco_vendorでは物理計算用のバージョンを3.3.7へ変更しています。

## 教師データ収集用の依存

逆運動学の数値最適化にはSciPyを使い、データ保存・選別には固定したLeRobotのAPIを使います。これらの依存はソースを同梱せず、`teacher/pixi.lock` で固定して取得します。形式変換器は https://github.com/jinnymo/lerobot-v3-v2-converter のリビジョン `ecfbe20d022e02631946a353142f791939c9725c` を取得します。

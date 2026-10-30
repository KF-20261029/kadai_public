# モデルの利用・配布条件

本リポジトリのLoRAは、`dongyoonkim/so101-pi05-base` を基にペン収納を学習したものです。ベースモデルの公開モデルカードはGemma規約を指定しています。そのため、コードのApache-2.0とは区別し、モデルとその派生物の利用・配布には次の条件を適用します。

- [Gemma Terms of Use](licenses/Gemma-Terms.txt)
- [Gemma Prohibited Use Policy](licenses/Gemma-Prohibited-Use-Policy.txt)
- [モデルのNotice](models/pen_lora/NOTICE)

利用者・再配布者は、Gemmaの利用制限を守り、第三者へもこれらの規約と利用制限の通知を渡してください。変更したファイルには変更を明示し、再配布物にはNoticeを添付してください。これらの条件を、モデルの利用・配布に関する条件の一部とします。

同梱の規約は2026年10月29日にGoogleの公式ページから取得した本文です。公開元の確認先は [Gemma規約](https://ai.google.dev/gemma/terms) と [利用制限](https://ai.google.dev/gemma/prohibited_use_policy) です。

独自の変更は、成功デモ55試行による30,000ステップのLoRA学習と、タスク用の入出力正規化統計です。モデル本体を統合した重みは同梱せず、ベースモデルは固定したリビジョンを公開元から取得します。公開元のモデルカードは [dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base) を参照してください。

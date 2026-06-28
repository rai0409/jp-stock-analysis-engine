# JP Stock Analysis Engine

日本株の価格データ、開示イベント、銘柄ユニバースを扱うための分析エンジンです。
J-Quants等から取得したローカルデータを前提に、価格データ整備、開示イベント抽出、対象銘柄カバレッジ検証、スクリーニング検証を行います。

## 目的

このリポジトリは、以下のような業務を想定した実装例です。

* 日本株データの自動取得・整形
* 開示イベントを使った銘柄スクリーニング
* TOPIX1000相当ユニバースのカバレッジ確認
* 分析パイプラインの再現性・検証性の担保
* CLI / テスト / サンプルデータを含む分析基盤の構築

## 主な機能

* J-Quants日次価格データの取得・保存
* 上場銘柄マスターの取得・整形
* TOPIX1000相当ユニバースの検証
* 財務・適時開示イベントの抽出
* 銘柄別・日付別のイベント集計
* 分析用CLI
* pytestによる検証コード
* 公開用の合成サンプルデータ

## 技術スタック

* Python
* pandas
* pytest
* CLIベースのデータ処理
* J-Quants API連携を想定した設計

## ディレクトリ構成

```text
src/
  jp_stock_analysis/
    cli.py
    validation/
      universe_coverage.py

scripts/
  fetch_jquants_financial_summary_daily.py

tests/
  test_universe_coverage.py

examples/
  jquants_fin_summary_sample/
```

## サンプルデータ

`examples/jquants_fin_summary_sample/` には、公開用の合成サンプルを配置しています。

実際のJ-Quantsデータ、APIレスポンス、生データ、ローカルDB、生成ログは含めていません。
ticker、開示番号、件数、日付はすべてデモ用の値です。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

必要に応じて開発・検証用パッケージを追加します。

```bash
pip install pytest pandas
```

## テスト

```bash
pytest
```

## 環境変数

APIキーやローカルデータの保存先は、環境変数で管理します。

```bash
export JQUANTS_API_KEY="your-api-key"
export JP_STOCK_ANALYSIS_PROJECT_DIR="/path/to/project"
export JP_STOCK_ANALYSIS_UNIVERSE_FILE="/path/to/topix1000_usable_tickers.csv"
export JQUANTS_EXTERNAL_STORE_DIRS="/path/to/store1:/path/to/store2"
```

`.env`、APIキー、認証情報、実データはGit管理しません。

## 公開リポジトリに含めないもの

以下は公開対象外です。

```text
.env
.env.*
secrets/
config/credentials.json
data/
logs/
artifacts/
topix_weight/
analysis/jquants_fin_summary/
*.bak
*.backup
```

## 実務上の強み

この実装では、単に分析コードを書くのではなく、以下を重視しています。

* データ取得・保存・検証の分離
* 公開可能なサンプルと非公開データの分離
* テスト可能な検証ロジック
* CLIとして再実行できる処理構成
* ローカル絶対パスや認証情報をコードに固定しない設計

## 注意事項

このリポジトリは投資助言を目的としたものではありません。
出力結果は研究・検証・開発用途のサンプルであり、実際の投資判断には利用者自身の確認と責任が必要です。

外部サービスから取得したデータには、各サービスの利用規約が適用されます。
このリポジトリには外部データの再配布権は含まれません。

## ライセンス

Copyright (c) 2026 rai0409. All rights reserved.

本リポジトリのコード、設計、ドキュメントの無断複製、再配布、商用利用、改変利用を禁止します。
利用・共同開発・業務委託での利用を希望する場合は、事前に許可を取得してください。

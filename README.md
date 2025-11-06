# detect_period (autumn_work)

このリポジトリは、Excel (.xlsx) に保存された時系列データから周期（サンプル数および秒）を推定する小さな Go コマンドラインツール `detect_period` を含みます。

主な機能
- Excel ファイルの指定列または R1C1 範囲を読み取り、信号の自動相関（FFT）を使って周期を推定します。
- ピーク検出時にパラボリック補間を行い、サブサンプル精度の周期推定を行います。
- 片側（単一列）解析の他、2 つの列間の相互相関（クロスコリレーション）も可能です。

要件
- Go 1.18+（`go.mod` を参照）
- 外部ライブラリ: `github.com/xuri/excelize/v2`（既に `go.mod` に含まれています）

ビルド

ワークツリーのルートで:

```bash
go build -o detect_period ./
```

実行例

基本的な実行（列指定）:

```bash
./detect_period -in data.xlsx -col 2 -fs 100
```

R1C1 範囲を利用して、時間列とデータ列を指定する例:

```bash
# 時間は R17C8:R9999C8、データは R17C9:R9999C9 といった範囲指定を使う
./detect_period -in data.xlsx -tRange R17C8:R9999C8 -x1Range R17C9:R9999C9 -fs 100
```

クロス相関を使って 2 列の遅延を推定する例:

```bash
./detect_period -in data.xlsx -tRange R17C8:R9999C8 -x1Range R17C9:R9999C9 -x2Range R17C11:R9999C11 -xcorr
```

主なコマンドラインオプション（抜粋）
- `-in` : 入力 `.xlsx` ファイル（必須）
- `-sheet` : シート名（省略時は先頭シート）
- `-col` : 1 ベースの列番号（`readXLSX` を使う場合）
- `-fs` : サンプリング周波数（Hz、秒換算に使用）
- `-approx` : 予想周期（サンプル数）。推定の探索範囲やダウンサンプリング自動選択に利用
- `-searchPct` : 探索幅（例: 0.2 は ±20%）
- `-dsMax` : 最大ダウンサンプリング比
- `-debug` : デバッグ情報を表示

実装ノート
- FFT はリポジトリ内で実装（radix-2 Cooley–Tukey）。実数入力は複素変換して FFT を実行します。
- 自己相関は FFT を使って効率良く計算し、ピーク位置をパラボリック補間でサブサンプル精度で推定します。
- Excel 読み取りは `excelize` を使っています。セルの値は文字列として読み、カンマ小数点（","）をピリオドに正規化して `strconv.ParseFloat` でパースします。

注意点
- `PhysicsAutumnWork.xlsx` のような大きなバイナリ Excel ファイルは Git にコミットするとリポジトリが大きくなるため、コミットしないことを推奨します（`.gitignore` に `*.xlsx` を追加するか、必要であれば Git LFS を検討してください）。
- `go mod tidy` を実行して依存を整えてください。

開発・貢献
- 簡単なバグ報告や機能追加の PR は歓迎します。Issue を立ててから PR を送るとスムーズです。

ライセンス
- ここに LICENSE ファイルを追加してください（例: MIT）。

作者
- ayataka5

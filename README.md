# text-to-speech
テキストから日本語の読み上げを生成します

## 実行コマンド

```bash
echo "こんにちは、これはテストです。" | text-to-speech -o output.mp3

cat article.txt | text-to-speech -o output.mp3

cat article.txt | text-to-speech -o result.mp3 --voice nova --speed 1.2
```

## オプション

```
オプション	短縮形	必須	説明	デフォルト値
--output	-o	Yes	出力するMP3ファイルのパス	-
--voice	-	No	声色 (alloy, echo, fable, onyx, nova, shimmer)	alloy
--speed	-	No	読み上げ速度 (0.25 ~ 4.0)	1.0
--model	-	No	使用モデル (gpt-4o-mini-tts, tts-1, tts-1-hd)	gpt-4o-mini-tts
--parallel	-p	No	並列処理数	3
--chunk	-	No	テキスト分割の目安文字数	400
--debug-dir	-	No	中間ファイルを保存するディレクトリ	なし
```

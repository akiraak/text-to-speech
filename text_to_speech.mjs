#!/usr/bin/env node

import "dotenv/config";
import fs from "fs";
import path from "path";
import os from "os";
import OpenAI from "openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import ffmpeg from "fluent-ffmpeg";
import { parseArgs } from "node:util";

// ==========================================
//  Configuration (設定)
// ==========================================
const CONFIG = {
  tts: {
    voice: "alloy", // nova, alloy, echo, fable, onyx, shimmer
    speed: 1.0,
    model: "gpt-4o-mini-tts", // コスト重視なら gpt-4o-mini-tts (旧 tts-1), 高品質なら tts-1-hd
  },
  text: {
    chunkSize: 400,
    separators: ["\n\n", "\n", "。", "、", ".", ",", " ", ""],
  },
  audio: {
    normalization: {
      targetI: -16,   // 目標ラウドネス (LUFS)
      targetTP: -1.5, // True Peak
      targetLRA: 11   // Loudness Range
    },
    paddingDuration: 1.0 // 前後の無音秒数
  },
  processing: {
    parallel: 3 // デフォルトの並列実行数
  }
};

// コマンドライン引数の解析
let args = {};
try {
  const { values, positionals } = parseArgs({
    options: {
      voice: { type: "string" },
      speed: { type: "string" },
      model: { type: "string" },
      chunk: { type: "string" },
      output: { type: "string", short: "o" },   // 出力パス (必須)
      "debug-dir": { type: "string" },          // デバッグ用出力ディレクトリ
      parallel: { type: "string", short: "p" }, // 並列数
    },
    allowPositionals: true,
  });

  // outputは必須
  if (!values.output) {
    console.error("エラー: 出力ファイルパス (--output / -o) は必須です。");
    process.exit(1);
  }

  args = { ...values, input: positionals[0] }; // input は undefined の可能性あり

  if (values["debug-dir"]) {
    args.debugDir = values["debug-dir"];
  }

  // 設定の上書き
  if (args.voice) CONFIG.tts.voice = args.voice;
  if (args.speed) CONFIG.tts.speed = parseFloat(args.speed);
  if (args.model) CONFIG.tts.model = args.model;
  if (args.chunk) CONFIG.text.chunkSize = parseInt(args.chunk, 10);
  if (args.parallel) CONFIG.processing.parallel = parseInt(args.parallel, 10);

} catch (e) {
  console.warn("引数解析エラー:", e.message);
  process.exit(1);
}

const openai = new OpenAI();

// ==========================================
//  Services (機能モジュール)
// ==========================================

/**
 * 標準入力を読み込む関数
 */
function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    const stdin = process.stdin;

    // パイプ入力がなく、かつTTY(人間が操作)の場合は案内を出す
    if (stdin.isTTY) {
      console.error("入力ファイルが指定されていません。標準入力からテキストを待機します...");
      console.error("(入力を終了するには Ctrl+D を押してください)");
    }

    stdin.setEncoding("utf8");
    stdin.on("data", chunk => data += chunk);
    stdin.on("end", () => resolve(data));
    stdin.on("error", reject);
  });
}

class FileManager {
  constructor(debugOutputDir = null) {
    this.debugOutputDir = debugOutputDir;
    this.workDir = path.join(os.tmpdir(), `tts-work-${Date.now()}`);
  }

  prepareWorkspace() {
    if (!fs.existsSync(this.workDir)) {
      fs.mkdirSync(this.workDir, { recursive: true });
    }
    return this.workDir;
  }

  getFilePath(fileName) {
    return path.join(this.workDir, fileName);
  }

  async saveText(fileName, content) {
    await fs.promises.writeFile(this.getFilePath(fileName), content);
  }

  async saveBuffer(fileName, buffer) {
    await fs.promises.writeFile(this.getFilePath(fileName), buffer);
  }

  _copyFileStream(src, dest) {
    return new Promise((resolve, reject) => {
      const readStream = fs.createReadStream(src);
      const writeStream = fs.createWriteStream(dest);
      readStream.on("error", reject);
      writeStream.on("error", reject);
      writeStream.on("finish", resolve);
      readStream.pipe(writeStream);
    });
  }

  async finalize(finalFileName, destPath) {
    const sourcePath = this.getFilePath(finalFileName);
    
    // ディレクトリが存在しない場合は作成
    const destDir = path.dirname(destPath);
    if (!fs.existsSync(destDir)) {
      await fs.promises.mkdir(destDir, { recursive: true });
    }

    console.log(`ファイルを保存中: ${destPath}`);
    await this._copyFileStream(sourcePath, destPath);

    if (this.debugOutputDir) {
      const timestamp = new Date().toISOString().replace(/[-:T.]/g, "").slice(0, 14);
      const debugDirName = `tts-debug-${timestamp}`;
      const debugPath = path.join(this.debugOutputDir, debugDirName);
      console.log(`[DEBUG] 中間ファイルを保存中: ${debugPath}`);
      await this.copyDirectory(this.workDir, debugPath);
    }

    await this.cleanup();
  }

  async cleanup() {
    if (fs.existsSync(this.workDir)) {
      try {
        await fs.promises.rm(this.workDir, { recursive: true, force: true });
      } catch (e) {
        console.warn("一時ディレクトリの削除に失敗しました:", e.message);
      }
    }
  }

  async copyDirectory(src, dest) {
    await fs.promises.mkdir(dest, { recursive: true });
    const entries = await fs.promises.readdir(src, { withFileTypes: true });

    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);
      if (entry.isDirectory()) {
        await this.copyDirectory(srcPath, destPath);
      } else {
        try {
          await this._copyFileStream(srcPath, destPath);
        } catch (e) {
          console.warn(`[DEBUG] コピー失敗: ${entry.name}`);
        }
      }
    }
  }
}

const TextProcessor = {
  async readInput(filePath) {
    if (!filePath) throw new Error("Input file path is required");
    const text = await fs.promises.readFile(filePath, "utf-8");
    if (!text.trim()) throw new Error("File content is empty");
    return text;
  },

  async splitText(text, config) {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: config.chunkSize,
      chunkOverlap: 0,
      separators: config.separators,
    });
    return await splitter.createDocuments([text]);
  }
};

const AudioGenerator = {
  async generate(text, config) {
    const mp3 = await openai.audio.speech.create({
      model: config.model,
      voice: config.voice,
      input: text,
      speed: config.speed,
    });
    return Buffer.from(await mp3.arrayBuffer());
  }
};

const AudioProcessor = {
  createSilence(referenceFile, outputFile, duration) {
    return new Promise((resolve, reject) => {
      ffmpeg(referenceFile)
        .audioFilters('volume=0')
        .duration(duration)
        .save(outputFile)
        .on('end', () => resolve())
        .on('error', reject);
    });
  },

  async mergeAndNormalize(inputFiles, outputPath, config) {
    // 1つしかない場合は結合処理をスキップしてコピー＆正規化だけでも良いが
    // 統一処理として無音付与を行う
    const silencePath = path.join(path.dirname(outputPath), "silence_padding.mp3");
    
    try {
      console.log(`\nフォーマット調整用の無音ファイルを生成中...`);
      await this.createSilence(inputFiles[0], silencePath, config.paddingDuration);

      const filesToConcat = [silencePath, ...inputFiles, silencePath];
      console.log(`結合・ノーマライズ中... (全${filesToConcat.length}ファイル)`);

      await new Promise((resolve, reject) => {
        const command = ffmpeg();
        filesToConcat.forEach(file => command.input(file));

        const filterInput = filesToConcat.map((_, i) => `[${i}:0]`).join('');
        const norm = config.normalization;
        // concat -> loudnorm
        const complexFilter = `${filterInput}concat=n=${filesToConcat.length}:v=0:a=1[cat];[cat]loudnorm=I=${norm.targetI}:TP=${norm.targetTP}:LRA=${norm.targetLRA}[out]`;

        command
          .complexFilter(complexFilter)
          .map('[out]')
          .audioCodec('libmp3lame')
          .save(outputPath)
          .on('end', resolve)
          .on('error', reject);
      });

    } finally {
      if (fs.existsSync(silencePath)) {
        try { fs.unlinkSync(silencePath); } catch(e){}
      }
    }
  }
};

// ==========================================
//  Main Flow
// ==========================================
async function main() {
  const fileManager = new FileManager(args.debugDir);

  try {
    const outputFilePath = args.output;
    
    // 1. 初期化
    const workDir = fileManager.prepareWorkspace();
    console.log(`=== 音声生成開始 ===`);
    console.log(`出力先: ${outputFilePath}`);
    console.log(`設定: Model=${CONFIG.tts.model}, Voice=${CONFIG.tts.voice}, Speed=${CONFIG.tts.speed}`);

    // 2. テキスト取得 (ファイル or 標準入力)
    let rawText = "";
    if (args.input) {
      console.log(`入力ソース: ファイル (${args.input})`);
      rawText = await TextProcessor.readInput(args.input);
    } else {
      console.log(`入力ソース: 標準入力 (パイプ)`);
      rawText = await readStdin();
    }

    if (!rawText || !rawText.trim()) {
      console.error("エラー: テキストが空です。処理を中止します。");
      await fileManager.cleanup();
      process.exit(1);
    }

    // 3. 分割
    const docs = await TextProcessor.splitText(rawText, CONFIG.text);
    console.log(`テキストを ${docs.length} チャンクに分割しました。`);

    // 4. 並列処理
    const chunkResults = new Array(docs.length).fill(null);
    const queue = docs.map((doc, index) => ({ doc, index }));
    const concurrency = CONFIG.processing.parallel;

    const processChunk = async ({ doc, index }) => {
        const partText = doc.pageContent.trim();
        if (!partText) return;

        const seqNum = String(index + 1).padStart(3, '0');
        // console.log(`[Chunk ${index + 1}] 生成中...`); // ログが多すぎる場合はコメントアウト

        const textFileName = `part_${seqNum}.txt`;
        await fileManager.saveText(textFileName, partText);

        const audioBuffer = await AudioGenerator.generate(partText, CONFIG.tts);
        const audioFileName = `part_${seqNum}.mp3`;
        await fileManager.saveBuffer(audioFileName, audioBuffer);
        
        chunkResults[index] = fileManager.getFilePath(audioFileName);
        process.stdout.write("."); // 進捗インジケータ
    };

    const workers = new Array(Math.min(concurrency, queue.length)).fill(null).map(async (_, workerId) => {
        while (queue.length > 0) {
            const item = queue.shift();
            if (item) {
                try {
                    await processChunk(item);
                } catch (e) {
                    console.error(`\n[Worker ${workerId}] Chunk ${item.index + 1} Error:`, e.message);
                    throw e; 
                }
            }
        }
    });

    console.log(`生成中 (${concurrency}並列)...`);
    await Promise.all(workers);
    console.log("\n全チャンク生成完了。");

    const audioFileNames = chunkResults.filter(p => p !== null);

    // 5. 結合・仕上げ
    if (audioFileNames.length > 0) {
      const tempFinalName = "combined_temp.mp3";
      const tempFinalPath = fileManager.getFilePath(tempFinalName);

      await AudioProcessor.mergeAndNormalize(audioFileNames, tempFinalPath, CONFIG.audio);
      await fileManager.finalize(tempFinalName, outputFilePath);
      
      console.log(`=== 完了: ${outputFilePath} ===`);
    } else {
      console.log("処理可能なテキストがありませんでした。");
      await fileManager.cleanup();
    }

  } catch (error) {
    console.error("\n致命的なエラー:", error.message);
    await fileManager.cleanup();
    process.exit(1);
  }
}

main();
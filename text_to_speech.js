#!/usr/bin/env node

require("dotenv").config();
const fs = require("fs/promises"); // fs.promises を直接 require
const { createReadStream, createWriteStream, existsSync } = require("fs"); // ストリーム系は別途必要なら
const path = require("path");
const os = require("os");
const OpenAI = require("openai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const ffmpeg = require("fluent-ffmpeg");
const { parseArgs } = require("node:util");

// ==========================================
//  Logger Utility
// ==========================================
const Logger = {
  info: (msg) => console.log(msg),
  progress: (char = ".") => process.stdout.write(char),
  warn: (msg, err) => console.warn(`\x1b[33m[WARN]\x1b[0m ${msg}`, err || ""),
  error: (msg, err) => console.error(`\x1b[31m[ERROR]\x1b[0m ${msg}`, err || ""),
};

// ==========================================
//  Configuration Management
// ==========================================
const DEFAULT_CONFIG = {
  tts: {
    voice: "alloy",
    speed: 1.0,
    model: "gpt-4o-mini-tts",
  },
  text: {
    chunkSize: 400,
    separators: ["\n\n", "\n", "。", "、", ".", ",", " ", ""],
  },
  audio: {
    normalization: { targetI: -16, targetTP: -1.5, targetLRA: 11 },
    paddingDuration: 1.0,
  },
  processing: {
    parallel: 3,
  },
};

function loadConfig() {
  try {
    const { values } = parseArgs({
      options: {
        voice: { type: "string" },
        speed: { type: "string" },
        model: { type: "string" },
        chunk: { type: "string" },
        output: { type: "string", short: "o" },
        "debug-dir": { type: "string" },
        parallel: { type: "string", short: "p" },
      },
      allowPositionals: false,
    });

    if (!values.output) {
      throw new Error("出力ファイルパス (--output / -o) は必須です。");
    }

    // 設定のマージ (ディープコピーは簡易的に実施)
    const config = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
    
    if (values.voice) config.tts.voice = values.voice;
    if (values.speed) config.tts.speed = parseFloat(values.speed);
    if (values.model) config.tts.model = values.model;
    if (values.chunk) config.text.chunkSize = parseInt(values.chunk, 10);
    if (values.parallel) config.processing.parallel = parseInt(values.parallel, 10);

    return {
      config,
      args: {
        output: values.output,
        debugDir: values["debug-dir"],
      },
    };
  } catch (e) {
    Logger.error(e.message);
    process.exit(1);
  }
}

// ==========================================
//  Services
// ==========================================

async function readStdin() {
  if (process.stdin.isTTY) {
    Logger.warn("入力テキストがありません。標準入力でテキストを待機中... (終了: Ctrl+D)");
  }
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

class FileManager {
  constructor(debugOutputDir) {
    this.debugOutputDir = debugOutputDir;
    this.workDir = path.join(os.tmpdir(), `tts-work-${Date.now()}`);
  }

  async prepareWorkspace() {
    await fs.mkdir(this.workDir, { recursive: true });
    return this.workDir;
  }

  getFilePath(fileName) {
    return path.join(this.workDir, fileName);
  }

  async saveFile(fileName, data) {
    await fs.writeFile(this.getFilePath(fileName), data);
    return this.getFilePath(fileName);
  }

  async finalize(finalTempName, destPath) {
    const sourcePath = this.getFilePath(finalTempName);
    
    // 出力先ディレクトリの作成
    await fs.mkdir(path.dirname(destPath), { recursive: true });

    Logger.info(`ファイルを保存中: ${destPath}`);
    // fs.promises.copyFile でシンプルにコピー
    await fs.copyFile(sourcePath, destPath);

    // デバッグディレクトリへのコピー (fs.cp で再帰コピーが可能)
    if (this.debugOutputDir) {
      Logger.info(`[DEBUG] 中間ファイルを保存中: ${this.debugOutputDir}`);
      await fs.cp(this.workDir, this.debugOutputDir, { recursive: true });
    }

    await this.cleanup();
  }

  async cleanup() {
    try {
      // fs.rm (recursive: true) でディレクトリごと削除
      await fs.rm(this.workDir, { recursive: true, force: true });
    } catch (e) {
      Logger.warn("一時ディレクトリ削除失敗", e);
    }
  }
}

const TextProcessor = {
  async splitText(text, config) {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: config.chunkSize,
      chunkOverlap: 0,
      separators: config.separators,
    });
    const docs = await splitter.createDocuments([text]);
    return docs.map(d => d.pageContent.trim()).filter(t => t.length > 0);
  }
};

const AudioGenerator = {
  openai: new OpenAI(),
  async generate(text, config) {
    const mp3 = await this.openai.audio.speech.create({
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
        .on('end', resolve)
        .on('error', reject);
    });
  },

  async mergeAndNormalize(inputFiles, outputPath, config) {
    const silencePath = path.join(path.dirname(outputPath), "silence_padding.mp3");
    
    try {
      Logger.info(`\nフォーマット調整用の無音ファイルを生成中...`);
      await this.createSilence(inputFiles[0], silencePath, config.paddingDuration);

      const filesToConcat = [silencePath, ...inputFiles, silencePath];
      const count = filesToConcat.length;
      Logger.info(`結合・ノーマライズ中... (全${count}ファイル)`);

      await new Promise((resolve, reject) => {
        const command = ffmpeg();
        filesToConcat.forEach(file => command.input(file));

        // フィルタ文字列生成ロジック
        const filterInput = filesToConcat.map((_, i) => `[${i}:0]`).join('');
        const { targetI, targetTP, targetLRA } = config.normalization;
        const complexFilter = `${filterInput}concat=n=${count}:v=0:a=1[cat];[cat]loudnorm=I=${targetI}:TP=${targetTP}:LRA=${targetLRA}[out]`;

        command
          .complexFilter(complexFilter)
          .map('[out]')
          .audioCodec('libmp3lame')
          .save(outputPath)
          .on('end', resolve)
          .on('error', reject);
      });
    } finally {
      // unlinkもPromise版で
      await fs.unlink(silencePath).catch(() => {});
    }
  }
};

/**
 * 並列処理ヘルパー
 * items: 処理する配列
 * concurrency: 並列数
 * taskFn: (item, index) => Promise<Result>
 */
async function runInParallel(items, concurrency, taskFn) {
  const results = new Array(items.length);
  const queue = items.map((item, index) => ({ item, index }));

  const workers = Array(Math.min(concurrency, items.length))
    .fill(null)
    .map(async () => {
      while (queue.length > 0) {
        const { item, index } = queue.shift();
        try {
          results[index] = await taskFn(item, index);
        } catch (e) {
          throw new Error(`Item ${index} failed: ${e.message}`);
        }
      }
    });

  await Promise.all(workers);
  return results;
}

// ==========================================
//  Main Flow
// ==========================================
async function main() {
  const { config, args } = loadConfig();
  const fileManager = new FileManager(args.debugDir);

  try {
    await fileManager.prepareWorkspace();
    
    Logger.info(`=== 音声生成開始 ===`);
    Logger.info(`出力先: ${args.output}`);
    Logger.info(`設定: ${JSON.stringify(config.tts)}`);

    // 1. テキスト取得
    const rawText = await readStdin();
    if (!rawText.trim()) throw new Error("テキストが空です。");

    // 2. 分割
    const textChunks = await TextProcessor.splitText(rawText, config.text);
    Logger.info(`テキストを ${textChunks.length} チャンクに分割しました。`);

    // 3. 音声生成 (並列処理)
    Logger.info(`生成中 (${config.processing.parallel}並列)...`);
    
    const audioFiles = await runInParallel(
      textChunks, 
      config.processing.parallel, 
      async (chunkText, index) => {
        const seqNum = String(index + 1).padStart(3, '0');
        
        // テキスト保存
        await fileManager.saveFile(`part_${seqNum}.txt`, chunkText);
        
        // 音声生成・保存
        const audioBuffer = await AudioGenerator.generate(chunkText, config.tts);
        const filePath = await fileManager.saveFile(`part_${seqNum}.mp3`, audioBuffer);
        
        Logger.progress();
        return filePath;
      }
    );

    Logger.info("\n全チャンク生成完了。");

    // 4. 結合・仕上げ
    if (audioFiles.length > 0) {
      const tempFinalName = "combined_temp.mp3";
      const tempFinalPath = fileManager.getFilePath(tempFinalName);

      await AudioProcessor.mergeAndNormalize(audioFiles, tempFinalPath, config.audio);
      await fileManager.finalize(tempFinalName, args.output);
      
      Logger.info(`=== 完了: ${args.output} ===`);
    }

  } catch (error) {
    Logger.error("致命的なエラー", error.message);
    await fileManager.cleanup();
    process.exit(1);
  }
}

main();
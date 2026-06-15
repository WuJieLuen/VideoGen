# VideoGen 使用文檔

VideoGen 是一個基於 Hugging Face Diffusers 與 CogVideoX 的本地文字生成影片（Text-to-Video）範例專案。專案目前使用本機已下載的 `THUDM/CogVideoX-2b` 模型快照，讀取文字提示詞後輸出 MP4 影片。

## 目錄

- [功能概覽](#功能概覽)
- [模型出處](#模型出處)
- [環境需求](#環境需求)
- [依賴模組](#依賴模組)
- [專案結構](#專案結構)
- [安裝步驟](#安裝步驟)
- [模型準備](#模型準備)
- [使用範例](#使用範例)
- [參數說明](#參數說明)
- [輸出結果](#輸出結果)
- [常見問題](#常見問題)
- [注意事項](#注意事項)

## 功能概覽

- 使用 `CogVideoXPipeline` 載入 CogVideoX 文字生成影片模型。
- 支援從本機模型快照載入權重，適合離線或受控環境執行。
- 使用 `float16` 降低顯存占用。
- 啟用 `enable_model_cpu_offload()`，在顯存不足時將部分模型模組卸載到 CPU。
- 透過 `export_to_video()` 將生成影格輸出為 MP4 檔案。

## 模型出處

本專案目前在 `main.py` 中載入的模型路徑為：

```python
./models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01
```

模型來源資訊：

| 項目 | 說明 |
| --- | --- |
| 模型名稱 | `THUDM/CogVideoX-2b` |
| 模型類型 | Text-to-Video 生成模型 |
| 使用方式 | Hugging Face Diffusers `CogVideoXPipeline` |
| 載入位置 | 本機模型快照資料夾 |
| 快照 ID | `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01` |

> 注意：本專案預設以 `local_files_only=True` 載入模型，因此執行前必須先將模型下載到上述路徑，或自行修改 `main.py` 中的模型路徑。

## 環境需求

建議環境如下：

- Python `>= 3.12`
- `uv` Python 套件管理工具
- NVIDIA GPU（建議使用支援 CUDA 12.8 的環境）
- 足夠的磁碟空間存放模型權重與輸出影片
- Linux / WSL / 具備 CUDA 支援的開發環境

若 GPU 顯存有限，專案已啟用 CPU offload，但生成速度會受到 CPU、記憶體與磁碟 I/O 影響。

## 依賴模組

本專案的依賴定義於 `pyproject.toml`：

| 模組 | 版本需求 | 用途 |
| --- | --- | --- |
| `accelerate` | `>=1.14.0` | 模型加速、裝置映射與 CPU/GPU offload 支援 |
| `diffusers` | `>=0.38.0` | 載入與執行 CogVideoX Diffusers pipeline |
| `imageio` | `>=2.37.3` | 影像與影片 I/O 支援 |
| `imageio-ffmpeg` | `>=0.6.0` | MP4 / ffmpeg 影片輸出支援 |
| `protobuf` | `>=7.35.1` | 模型與 tokenizer 相關序列化支援 |
| `sentencepiece` | `>=0.2.1` | tokenizer 支援 |
| `tiktoken` | `>=0.13.0` | tokenizer / 文字處理支援 |
| `torch` | `==2.8.0` | PyTorch 深度學習框架 |
| `torchaudio` | `==2.8.0` | PyTorch 音訊套件，與 PyTorch 版本對齊 |
| `torchvision` | `==0.23.0` | PyTorch 視覺套件，與 PyTorch 版本對齊 |
| `transformers` | `>=5.12.0` | Transformer 模型與 tokenizer 支援 |

PyTorch 相關套件使用 CUDA 12.8 wheel index：

```toml
[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu128"
```

## 專案結構

```text
VideoGen/
├── README.md          # 使用文檔
├── main.py            # 文字生成影片主程式
├── pyproject.toml     # 專案與依賴設定
├── uv.lock            # uv 鎖定檔
└── cmd.txt            # PyTorch CUDA 依賴安裝指令備忘
```

## 安裝步驟

### 1. 安裝 uv

若尚未安裝 `uv`，請先依照你的系統安裝 uv。

### 2. 建立與同步環境

在專案根目錄執行：

```bash
uv sync
```

### 3. 安裝 CUDA 12.8 版 PyTorch（如需手動安裝）

專案已在 `pyproject.toml` 設定 PyTorch CUDA 12.8 index。若需要手動加入或重新安裝，可參考 `cmd.txt`：

```bash
uv add torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index https://download.pytorch.org/whl/cu128
```

## 模型準備

因為 `main.py` 使用：

```python
local_files_only=True
```

所以程式不會在執行時自動從網路下載模型。你需要確認模型資料夾存在：

```text
./models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01
```

如果你的模型存放在其他位置，請修改 `main.py`：

```python
pipe = CogVideoXPipeline.from_pretrained(
    r"你的模型路徑",
    torch_dtype=torch.float16,
    local_files_only=True,
)
```

如果希望由 Hugging Face Hub 自動下載，可將模型來源改為模型 ID，並移除或調整 `local_files_only=True`：

```python
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16,
)
```

## 使用範例

### 基本執行

在專案根目錄執行：

```bash
uv run python main.py
```

程式會：

1. 載入本機 CogVideoX-2b 模型。
2. 使用 `prompt` 生成影片影格。
3. 將生成結果輸出為 `demo.mp4`。
4. 在終端機輸出 PyTorch 版本與完成訊息。

### 修改提示詞

打開 `main.py`，修改 `prompt`：

```python
video = pipe(
    prompt="A cinematic shot of a futuristic city at sunset, flying cars moving between skyscrapers",
    num_frames=40,
).frames[0]
```

再執行：

```bash
uv run python main.py
```

### 修改輸出檔名與 FPS

預設輸出設定如下：

```python
export_to_video(video, "demo.mp4", fps=4)
```

你可以改成：

```python
export_to_video(video, "future_city.mp4", fps=8)
```

## 參數說明

目前 `main.py` 中主要可調整參數：

| 參數 | 目前設定 | 說明 |
| --- | --- | --- |
| `prompt` | 範例提示詞 | 控制生成影片內容的文字描述 |
| `num_frames` | `40` | 生成影片影格數，越高通常耗時與顯存需求越高 |
| `fps` | `4` | 輸出影片每秒影格數 |
| `torch_dtype` | `torch.float16` | 使用半精度以降低顯存需求 |
| `local_files_only` | `True` | 僅從本機模型快取載入，不連線下載 |

## 輸出結果

預設輸出檔案：

```text
demo.mp4
```

影片長度約為：

```text
num_frames / fps = 40 / 4 = 10 秒
```

實際生成速度取決於：

- GPU 型號與顯存大小
- CPU offload 使用情況
- 系統記憶體
- 模型存放磁碟速度
- 生成影格數與影片解析度

## 常見問題

### 1. 找不到模型資料夾

請確認模型已下載到 `main.py` 指定的路徑，或修改 `from_pretrained()` 的第一個參數。

### 2. CUDA / PyTorch 版本不相容

請確認你的 NVIDIA 驅動、CUDA 執行環境與 PyTorch wheel 相容。本專案目前指定 PyTorch `2.8.0`、TorchVision `0.23.0`、TorchAudio `2.8.0`，並使用 CUDA 12.8 wheel index。

### 3. 顯存不足

可嘗試：

- 降低 `num_frames`
- 關閉其他 GPU 程式
- 確認 `pipe.enable_model_cpu_offload()` 已啟用
- 使用顯存更大的 GPU

### 4. 影片輸出失敗

請確認 `imageio` 與 `imageio-ffmpeg` 已正確安裝，並使用 `uv sync` 重新同步依賴。

## 注意事項

- 生成式模型可能產生不準確、不穩定或不符合預期的內容，請在發布或商用前進行人工審查。
- 使用模型時請遵守模型授權條款、資料來源規範與所在地法律法規。
- 若生成內容涉及人物、品牌、敏感場景或受保護素材，請確認具備必要權利與使用許可。
- 不建議生成或散布暴力、仇恨、色情、詐欺或其他違法有害內容。

# 训练内核 —— 源码 + 重打说明

> SVN 位置：`作者/DemoText客户端svn版本/大模型/`
>
> 这个目录是"打包前"的东西：训练内核的 **Python 源码** + **重打包所需的全部信息**；
> 打包后的成品在本目录的 `内核成品\` 下，OSS 上另有一份，见第四节。

## 一、目录内容

上级目录 `大模型\`：

| 文件 | 说明 |
|---|---|
| `LargeModelDemoText.py` | **内核源码**（30,703 字节）。训练 / 评估 / 预测 / 命令行参数全在这里 |
| `Word2Vec.py` | 词向量相关（历史文件） |
| `_打包文档_正文.txt`、`DemoText大模型打包.docx` | 早期的打包记录 |

本目录 `构建脚本\`：

| 文件 | 说明 |
|---|---|
| `LargeModelDemoText.spec` | PyInstaller 配置（`phase6` 会生成/使用） |
| `_kernel_requirements.txt` | **61 个依赖的精确版本**，与 `phase2_deps.py` 完全一致 |
| `phase1_install_python.py` | 装 Python 3.11.6 到构建目录 |
| `phase2_deps.py` | 装 cu128 的 torch 三件套 + 其余 58 个依赖（版本钉死） |
| `phase3_cuda_smoke.py` | CUDA 自检：矩阵乘 / cuDNN 卷积 / 反向传播 / 交叉熵 |
| `phase5_run_kernel.py` | 用客户端**同款参数**跑内核（判据：报告文件有没有生成） |
| `phase6_build_kernel.py` | **PyInstaller 打包 + 用打出来的 exe 跑一遍** |
| `_peek_oss_zip.py` | 用 HTTP Range 看 OSS 上某个 zip 里是什么（不用下整包） |
| `_check_oss_now.py` | 查 OSS 上几个内核对象的体积 / 时间 |
| `README_未打包版_怎么重打.md` | 本文件 |

## 二、本版（cu128）相对上一版（cu118）改了什么

1. **内核源码补 `labels=`**（`LargeModelDemoText.py`）：

   `classification_report(..., labels=list(label_encoder.classes_), zero_division=0, digits=4)`

   —— 修"评估指标显示为空"：测试集里只剩一个类别时 sklearn 会抛
   `ValueError: Number of classes, 1, does not match size of target_names, 2. Try specifying the labels parameter`，
   异常被调用方 `except` 吞掉 → **报告文件根本不会生成** → 客户端只能显示空指标。

2. **打包依赖 torch 2.7.1+cu118 → 2.7.1+cu128**（torchvision 0.22.1+cu128、torchaudio 2.7.1+cu128）

   —— cu118 里**没有 sm_120 的 kernel**，RTX 50 系（Blackwell）一碰 CUDA 就报
   `no kernel image is available` 或直接卡死。
   cu128 的 `torch.cuda.get_arch_list()` 实测为：`sm_50 / 60 / 61 / 70 / 75 / 80 / 86 / 90 / 100 / 120`
   （30 系 = sm_86 有专用 kernel；40 系 = sm_89 靠同大版本二进制兼容跑 sm_86，与历史做法一致）。

**驱动要求**：cu118 要 ≥452，**cu128 要 ≥570**（30/40/50 系通用同一个驱动包）。
对应的客户端闸门是 `DiskUtils.CheckGpuUsableByDriver()`（NVIDIA 卡 + `nvidia-smi` 驱动主版本 ≥ 570），
**不需要装 CUDA Toolkit**；驱动不够新时客户端会禁用 GPU 并提示升级。

## 三、怎么重打包

脚本里的构建目录写死为 `D:\aitext_kernel_build`（`phase1/2/5/6` 的 `BUILD` 常量），
`phase5/phase6` 还写死了源码路径 `...\DemoText客户端svn版本\大模型\LargeModelDemoText.py`。
**换一台机器重打，先把这几处改成自己的路径。**

```powershell
# 1) 装 Python 3.11.6 到 <BUILD>\py311
python phase1_install_python.py

# 2) 装依赖（torch 走 download.pytorch.org/whl/cu128，其余走 PyPI 清华源）
<BUILD>\py311\python.exe phase2_deps.py

# 3) CUDA 自检（应输出各步骤全部通过）
<BUILD>\py311\python.exe phase3_cuda_smoke.py

# 4) 打包 + 自测（PyInstaller onedir → <BUILD>\dist\LargeModelDemoText\，然后用它跑一遍出报告）
<BUILD>\py311\python.exe phase6_build_kernel.py
```

打包要点（`phase6` 里已固化）：

- **onedir + `_internal`**（不是 onefile），**不要 `--windowed`**（客户端要读它的 stdout，必须有控制台）
- `--collect-all` 的包：`torch torchvision torchaudio transformers tokenizers sklearn scipy numpy pandas joblib psutil safetensors huggingface_hub regex tqdm datasets pyarrow`
- 产物参考：**22,316 个文件 / 5.90 GB**，`LargeModelDemoText.exe` **91,212,186 字节**
- 打成 zip（deflate）后：**3,510,656,757 字节**

打包后校验（确认 labels= 修复真的生效）：

```powershell
# 报告必须能生成，且含「整体准确率accuracy:」
<BUILD>\py311\python.exe phase5_run_kernel.py after
```

## 四、成品包在哪、客户端从哪下

| | |
|---|---|
| 本仓库里 | `大模型\内核成品\LargeModelDemoText_cu128.zip` |
| 大小 | 3,510,656,757 字节 |
| SHA256 | `3D3B14A41A197D445C492EDDE24699EB48CC2D1C92F379F8E833229D80B6879F` |
| OSS 上同一份 | `https://bytelove-picture.oss-cn-shanghai.aliyuncs.com/PythonExe/LargeModelDemoText_cu128.zip` |

OSS 上现有三个内核对象（实测：HEAD 取体积 + HTTP Range 读 zip 中央目录）：

| OSS 对象 | 体积 | 最后修改 | 包里是 |
|---|---|---|---|
| `aitextclient/2026124/LargeModelDemoText.zip` | 3,173,877,471 B | 2026-01-24 | **cu118**：`cublas64_11` / `cudart64_110` / `nvrtc64_112_0`，exe 68,917,594 字节 |
| `PythonExe/LargeModelDemoText.zip` | 842,193,258 B | 2025-12-24 | 更老的 Paddle 包，`BigModelUtils.loadModelMarching` 在用（4 个 `*_Txt` 页面） |
| `PythonExe/LargeModelDemoText_cu128.zip` | 3,510,656,757 B | 2026-10-12 | **本版 cu128**：`cublas64_12` / `cudart64_12` / `nvrtc64_120`，exe 91,212,186 字节 |

> **注意（重要）**：客户端源码里 5 处下载地址现在指向的是 2026-01-24 那个 **cu118** 老对象 ——
> `LargeLanguageModel.xaml.cs`（4 处）+ `LargeModelHomePage.xaml.cs`（1 处）用的都是
> `.../aitextclient%2F2026124%2FLargeModelDemoText.zip`。
>
> 也就是说：**只把新内核放到 OSS 上还不够，客户端自动下到的仍是 cu118 内核，50 系显卡依然不可用。**
> 要真正生效，下面两条选一条：
>
> 1. 把这 5 处地址改成 `.../PythonExe/LargeModelDemoText_cu128.zip`（改客户端代码 → 重新出包），或
> 2. 把 cu128 的 zip 覆盖上传到老对象 `aitextclient/2026124/LargeModelDemoText.zip` 上。
>
> 另外，**已经下过旧内核的机器**还需要后端把「LargeModelDemoText 最新版本号」+1 才会自动重下
> （客户端判定逻辑：`getTheLatest(model)` 与 `getUserDownLoadModel(user,mac,model)` 是否相等）。

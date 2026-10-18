# Strata 在 Windows / Ryzen AI Max+ 395（Strix Halo，gfx1151）上的部署指南

> 实测环境：AMD Ryzen AI Max+ 395（Radeon 8060S，gfx1151），128 GB LPDDR5X，Windows 11 x64。
> 引擎：Strata 预编译 Windows HIP 版（`strata-windows-x64-hip.zip`），后端 `hip`；调优数据取自 **0.1.41**，上下文与版本数据取自 **0.1.45**，表格逐行标注。
> 上下文：**256K（262144）** —— 模型原生训练长度，无需 rope 缩放。
> 结论一句话：Strata 官方文档把 Windows + Strix Halo 标注为"尚未测试"；本文记录了实际跑通的全过程、实测性能与需要绕开的坑。
>
> 📁 **路径约定**：本文用两个占位符，请先替换成你自己的路径——
> `<STRATA>` = Strata 安装目录（举例：`D:\Strata`）；`<MODELS>` = 模型根目录（举例：`D:\Models`）。
> JSON / bat 片段里保持反斜杠写法即可。

---

## 一、结果先看（实测性能）

模型 IQ3_S（GSQ-RCO）+ **256K（262144）原生上下文** + int8 KV，单次请求固定输出长度。
第 1~3 步在 0.1.41 上测，第 4、5 步在升到 0.1.45 后测。

| 阶段 | 配置动作 | 中文解码 tok/s | 长 prompt 解码 | 长 prompt prefill |
|---|---|---|---|---|
| 起点 | 默认（auto 专家缓存，仅 320 槽，命中率 33%） | 28.0 | — | — |
| 第 1 步 | **`--mmap-experts`**（专家缓存涨到 24576 槽，全常驻显存） | 37.7 | 43.0 | 372 |
| 第 2 步 | **官方第 5 节 fast 开关**（会改变数值，见下文） | 38.1 | 36.1 | **522 ~ 532** |
| 第 3 步 | **`STRATA_SH_STREAM=0`** | **39.6** | — | — |
| 第 4 步 | **上下文 128K → 256K**（`--max-context 262144`） | 32.6 | — | — |
| 第 5 步 | **引擎 0.1.43 → 0.1.45**（256K，参数不变） | 34.0 | — | — |

- 同配置下换成英文 prompt：**42.3 tok/s**（草稿接受率 71%）。
- 官方文档 STRIX_HALO.md 第 6 节的 Linux 同型号参考值：**IQ3_S@128K = 42.8 tok/s**。
  本机（Windows）已非常接近这一水平，剩余差距主要来自文本语言与 Windows 开销。
- **上下文从 128K 提到 256K，只多花约 1.8 GiB 显存，速度基本不变。**
  262144 正是模型的**原生训练长度**，不涉及 rope 缩放。模型是 GDN 混合架构，**只有 QSA 层的 KV 随上下文增长**（约 12.4 KB/token，256K 约 3.1 GiB）；
  专家缓存仍稳住 **24576 槽**，空闲显存 45907 MiB（128K 时为 47759 MiB）。
  同机背靠背 A/B：**256K = 32.6 tok/s vs 128K = 30.8 tok/s**，差值属噪声——与官方"GDN 每 token O(1)、解码对上下文不敏感"一致。
- **0.1.43 → 0.1.45 升级在本机不提速**：实测 34.0 对 32.6，同处噪声区间。
  官方明确 Flash-Next 输出在这些版本间**逐字节相同**，且 0.1.45 的提速工作**"未在 Windows 硬件上运行过"**；
  0.1.42 / 0.1.43 的默认提速只覆盖 CUDA 与 RDNA4（gfx1200 / gfx1201），gfx1151 保持原行为。
  变的是默认开关数量：启动日志从 **17 个增到 18 个**，新增 `STRATA_TOPK_REG_MIN_BLOCKS`（16K cell 起的寄存器 top-k；官方 PR #2016，仅 gfx1151 实测）。
- **只在同一时段做背靠背对比。** 同一套配置在不同时段分别测到 38.3 和 32.6 tok/s，那是机器状态（页缓存、温度、后台负载）差异，不是调参效果。
- 网上常见的"50+ tok/s"读数，多数来自**更小的量化**（官方同型号表：IQ3_XXS@128K = 50.9、UD-IQ4_XS@128K = 51.4）。
  想在 128K 下上 50+，换 IQ3_XXS 或 UD-IQ4_XS 是更直接的路子；IQ3_S 在本机的实测天花板在 42 tok/s 附近。

**净效果：28.0 → 39.6 tok/s（+41%），prefill 提升更大。** 下面是一条条怎么来的。

---

## 二、前置准备

### 2.1 BIOS / 显存划分

- BIOS 中给 iGPU 分配 **32 GB 固定 carve-out**；其余走共享内存。
- 本机 HIP 最终报告可用显存 **104.5 GiB**（32 GB carve-out + 约 76.5 GB 共享）。
- 无需修改内核参数（Windows 没有这一项）；官方 STRIX_HALO.md 也说明它不改任何 BIOS 或内核设置。

### 2.2 软件

1. 下载 Strata 并解压（Windows HIP 预编译包 `strata-windows-x64-hip.zip`）到 **纯 ASCII、尽量少空格**的目录，例如 `<STRATA>`。
2. 运行 `SETUP.bat`（内部调用 `setup.py`）拉取模型与派生数据。
3. 模型与派生数据的默认落点：

```
<MODELS>\
├── ISTA-DASLab\IQ3_S\                 # 原版 IQ3_S（77.87 GiB）
│   ├── ...-IQ3_S-00001-of-00002.gguf  # 主模型 51.05 GiB
│   └── ...-IQ3_S-00002-of-00002.gguf  # 纯 PLE 表 26.82 GiB
└── Strata-data\                       # Strata 派生数据（pack / MTP 草稿层）
    ├── packs\iq3_s\
    └── mtp\rt\
```

> **坑**：模型一旦从 `Strata-data\models\` 移走，重跑 `setup.py` 会**重新下载**（`.done` 标记随目录一起移走了）。
> 本机把模型按发布者整理到 `ISTA-DASLab\` 后，必须在 config 里同步 `--native` / `--ple-gguf` 路径。

### 2.3 虚拟化

不需要。Strata 走 HIP 直通 iGPU，本机全程在裸机 Windows 上跑。

---

## 三、配置（关键部分）

配置文件 `strata-iq3_s.json`（引擎实际启动参数，路径请替换）：

```json
{
 "exe": "<STRATA>\\engine\\strata.exe",
 "args": [
  "--pack",          "<MODELS>\\Strata-data\\packs\\iq3_s",
  "--native",        "<MODELS>\\ISTA-DASLab\\IQ3_S\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
  "--ple-gguf",      "<MODELS>\\ISTA-DASLab\\IQ3_S\\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
  "--expert-profile","<STRATA>\\data\\expert-profile.bin",
  "--expert-cache",  "auto",
  "--prefill",       "16384",
  "--spec",          "4",
  "--spec-min-p",    "0.5",
  "--lookup-chain",  "3",
  "--mtp-q4",        "all",
  "--mtp-window",    "8192",
  "--mtp",           "<MODELS>\\Strata-data\\mtp\\rt",
  "--max-context",   "262144",
  "--kv",            "int8",
  "--mmap-experts"
 ],
 "backend": "hip",
 "env": {
  "STRATA_HIPBLASLT_TUNING": "<STRATA>\\tools\\hip\\gfx1151-hipblaslt-100500.txt",
  "STRATA_PF_FUSED": "1", "STRATA_PF_GEMM": "1", "STRATA_HC_UPMIX": "1",
  "STRATA_PA_FAST": "1", "STRATA_HIP_WMMA": "1", "STRATA_SELECT_WMMA": "1",
  "STRATA_HC_Q8": "1", "STRATA_PF_SWITCH_MIN_T": "4096",
  "STRATA_PREFILL_STREAM_MIN": "128",
  "STRATA_SH_STREAM": "0"
 }
}
```

### 3.1 `--mmap-experts` —— 本机收益最大的一项

**不加它，其余调优的收益会被大幅抵消。** 默认的"resident expert arena"会把约 46.84 GiB 专家 pin 进主机内存，Windows 把这笔内存算进本进程的 **GPU 显存预算（WDDM）**，结果 `device_free_bytes()` 只剩 2.25 GiB，专家缓存被压到 320 槽（命中率 33%，34% 的专家走 PCIe 读取），解码仅 ~28 tok/s。

实测证据：引擎进程 GPU 计数器 `Shared Usage = 48,723 MiB`，而 `Dedicated Usage` 只有 8,961 MiB（32 GB carve-out 里还空着约 22.6 GiB）。

加上 `--mmap-experts`（放弃 pinned arena，改用 GGUF 分片 mmap）后：缓存涨到 **24,576 槽**（全部专家常驻显存），空闲显存 47,760 MiB，解码 **37.7 tok/s（+35%）**，PCIe 专家读取归零。

代价：冷启动（重启后首次）要读 51 GiB 分片，缓存预填约 68 s；页缓存热了以后约 19 s。

### 3.2 官方第 5 节 fast 开关（会改变输出数值，官方已用 KL 门控）

见官方 `docs/STRIX_HALO.md` 第 5 节，本机全部开启：`STRATA_PF_FUSED / PF_GEMM / HC_UPMIX / PA_FAST / HIP_WMMA / SELECT_WMMA / HC_Q8`，
配 `STRATA_PF_SWITCH_MIN_T=4096`（短 prompt 保持默认数值）与 `STRATA_PREFILL_STREAM_MIN=128`。
效果：**prefill +42%（372 → 531 tok/s），两次复测一致**；解码基本不变。

### 3.3 `STRATA_SH_STREAM=0` —— Windows 上建议关掉

官方第 4 节说 `STRATA_SH_STREAM`（共享专家走第二条流）在 **gfx1151 上默认开启**，它在 Linux 上确实有收益。
但本机实测：默认（=1）中文解码 34.1 tok/s → 设为 **0** 后 **39.6 tok/s（+13%）**，干净的 A/B 远超噪声。
这与官方 issue #816 的记录一致（RX 6800 在 Windows 上开它慢 12–22%）。**Windows + Strix Halo 建议显式关掉。**

### 3.4 官方解码参数（照抄即可）

`--lookup-chain 3`、`--mtp-q4 all`、`--mtp-window 8192`、`--prefill 16384` —— 直接采用官方 fast 配置。
`--expert-cache auto` 在 Windows 上默认预留 2560 MiB（修复 #1376 的换页问题）。

### 3.5 官方有、但本机尚未验证的三条线索

以下三项官方文档给了正面数据，本机**没有确认**。前两项的首次 A/B 不是"没效果"而是**作废**了：
三次引擎重启分别耗时 57 s / 70 s / 141 s 完成专家缓存预填，说明**页缓存在逐渐变冷**，测出的 prompt 读取速度不可比。
需要把两组实验**交错重跑**才能定论。

- **`STRATA_PF_F16=1`**（gfx11 专用）—— 把被覆盖的 prompt 专家改走 **F16 矩阵核**而非 int8（源码 `src/prefill/moe_f16_kq.cu`，需在 `STRATA_PF_FUSED=1` 之上生效）。
  官方 `STRIX_HALO.md` 第 5 节给出单层 16384 token 的微基准：IQ3_S / IQ4_NL 的 gate/up 由 **11.65 → 9.87 ms**、down 由 **6.11 → 4.71 ms**。它会改变数值四舍五入，所以是 opt-in。
- **`STRATA_REFILL_PARTS=1`** —— 就地读 GGUF 时，用映射里的片段回填被借出的专家缓存槽位（源码 `src/program/generate.cpp`）。
  官方称它 **"exact"（逐字节不变）**，并给出 4K–8K prompt **+2.3% ~ +3.5%**；正好适用于我们这种 `--mmap-experts` 配置。
- **其余 16 个 gfx1151 默认开关**。源码 `src/core/arch_defaults.cpp` 注明它们是 **"Exact (bitwise) on Aurora"** —— 即**在 Linux 上调的**。
  `STRATA_SH_STREAM` 已被我们证明在 Windows 上反转（见 3.3），其余从没在 Windows 上验证过，值得逐个做背靠背 A/B；`STRATA_GFX1151_DEFAULTS=0` 可一键全关。

---

## 四、启动

`.bat` 启动脚本（关键是**先把 TEMP/TMP 指到纯 ASCII 路径**）：

```bat
@echo off
title Strata IQ3_S
cd /d "<STRATA>"
if not exist "<MODELS>\Strata-data\temp" mkdir "<MODELS>\Strata-data\temp"
set "TEMP=<MODELS>\Strata-data\temp"
set "TMP=<MODELS>\Strata-data\temp"
"<STRATA>\.venv\Scripts\python.exe" "<STRATA>\serve\server.py" ^
  "--engine" "strata" "--config" "<STRATA>\strata-iq3_s.json" "--port" "8080" "--open"
if errorlevel 1 pause
```

启动后浏览器打开 `http://127.0.0.1:8080/`，同时提供 **OpenAI 兼容**与 **Anthropic 兼容**接口。

**先把 TEMP 指走的原因**：server 以 UTF-8 传临时文件路径，引擎按 ANSI 打开；若 Windows 用户名含中文，TEMP 路径会让**视觉编码器**打不开文件，因此要改成纯 ASCII 路径。

### 验证是否成功

```powershell
curl http://127.0.0.1:8080/health
# {"status":"ok","max_context":262144,"model":"...","images":true,"loaded":true,"service":"strata"}
```

`loaded: true` 即就绪。`/metrics` 可看实时状态、tok/s、显存空闲、专家缓存槽位。

---

## 五、Windows 专属踩坑清单

> 这些都是代码和文档里读不出来、只有踩过才知道的。按此顺序排查能省掉大量时间。

1. **pinned 专家 arena 吃掉 WDDM 显存预算** —— 本机影响最大的一个坑。默认把 46.84 GiB 专家 pin 进主机内存，Windows 计入本进程 GPU 预算，显存只剩 2.25 GiB，专家缓存被压到 320 槽。**解法：`--mmap-experts`（+35%）。**

2. **`--kv-streaming on` 是 `setup` 的参数，不是引擎参数。** 引擎只认 `--kv-resident N`（最小 20480）。把 `--kv-streaming` 写进 config 的 `args` 会让引擎打印帮助并退出（exit 1）。

3. **本机 128K 下 KV streaming 不可用。** `--kv-resident 32768` 会把专家缓存压到 0 槽，引擎报 `no VRAM is left for the expert cache` 并拒绝服务。用了 `--mmap-experts` 后就不需要它了。

4. **`STRATA_WDDM_BUDGET=0` 在 0.1.41 无效果**：实测 320 槽 / 1116 MiB 空闲，与不设置完全一致；日志里也没有文档描述的 `Windows budgets N of this card's M MiB` 行。

5. **`--expert-cache auto` 在 Windows 上默认预留 2560 MiB**；显存不足时引擎会自动缩小预留以保住至少 250 槽。所以"0 MiB free 告警"只在显式设很大的 `--expert-cache N` 时才会出现。

6. **GGUF 分片语义因发布者而异，不能想当然合并。** 实测（`python tools/strata_inspect.py <分片路径>`）：
   - 原版 ISTA-DASLab 的**分片 2 是刻意独立的 PLE 表**（100% `n-gram/per-layer embeddings`，28.80 GB 单一张量）——**不能合并**，会破坏引擎把 PLE 单独分页的设计；
   - 另一类 IQ3_XXS 发布者是常规按大小切分（分片 1 = PLE + 部分专家，分片 2 = 其余专家）——这种情况可以合并。
   - 判断技巧：`strata_inspect.py` 报 `architecture None` 的即非首分片。
   - **`--ple-gguf` 必须指向含 PLE 的那个分片**（原版 = 分片 2；按大小切分的 = 分片 1）。

7. **模型移出 `Strata-data\models\` 后，重跑 `setup.py` 会重新下载**（`.done` 标记随目录移走）。迁移后要同步改 config 里的 `--native` / `--ple-gguf`。

8. **Windows + AMD 读不到 GPU 负载 / 显存 / 温度。** `/metrics` 的 `hardware_static.gpu_note` 原文："no GPU load or VRAM readings for AMD cards on Windows yet"。要显示显存请用引擎自己的 `engine.vram_free_mib`、`engine.expert_cache_mib`、`engine.expert_slots`（这三个是真数据）。

9. **`/metrics` 容量字段单位不统一，不能共用一个换算函数。** `hardware.ram_used/ram_total` 是**字节**（÷2³⁰）；`engine.vram_free_mib` / `expert_cache_mib` 是 **MiB**（÷2¹⁰）。官方 `web/app.js` 的 `gb()` 一律 ÷2³⁰，所以只对字节字段成立。

10. **Windows 用户名含中文时，TEMP 路径会让视觉编码器打不开文件**（见第四节：启动时把 TEMP/TMP 指到纯 ASCII 路径）。

11. **PowerShell 5 会按 GBK 解析无 BOM 的 `.ps1`**，脚本里写中文会把引号解析坏。基准脚本请保持纯 ASCII，中文 prompt 放独立 UTF-8 文件读取。

12. **`Start-Process -ArgumentList` 不会自动给含空格的路径加引号**，路径必须手动加双引号。

---

## 六、可选：视觉支持

引擎配置里加 `vision` 段即可（启动时自动附加 `--vision`）：

```json
"vision": {
  "exe": "<STRATA>\\engine\\strata-vision.exe",
  "mmproj": "<MODELS>\\unsloth\\Qwen3.8-Flash-Next-GGUF\\mmproj-F16.gguf",
  "model": "<含 PLE 的主分片路径>",
  "gpu": false, "max_tokens": 300, "threads": 16
}
```

要点：
- `strata-vision.exe` 是 **CPU 视觉编码器**，需要从源码自行编译；
- `mmproj` 要选**与底座同源**的投影文件（本机用的是同底座发布的 `mmproj-F16.gguf`）；
- 就绪后 `/health` 会返回 `"images": true`。

---

## 七、调优记录：哪些有效、哪些没有收益

以下全部为 3 次取中位数的干净 A/B。

**有效：**
- `--mmap-experts`：+35%（28.0 → 37.7）
- `STRATA_SH_STREAM=0`：+13%（34.1 → 39.6）
- 官方 fast 开关：prefill +42%

**本次实测没有收益或更差：**
- 缩小上下文：ctx 8K（38.3）、ctx 32K（38.2）——与 128K 无差；**放大到 256K 同样无差**（背靠背 A/B：256K 32.6 vs 128K 30.8），且专家缓存仍稳在 24576 槽
- 引擎 **0.1.43 → 0.1.45**：34.0 对 32.6，同处噪声区间（官方保证 Flash-Next 输出在这些版本间逐字节相同）
- `--kv-grow` 38.3、`--vram-reserve-mib 512` 38.0、pool tasks 12 38.5
- `--host-core last` 34.5（更差）、fast 开关全关 35.3
- `--spec 2` 33.5、`--spec 8` 30.6、`--spec-min-p 0.2` 31.1
- `--kv q4` 35.4、`--mtp-q4 proj` 36.6
- 各类 head/norm/io 变体（`--mtp-q4 head` 42.5、`--mtp-hnorm stream` 42.9、`--ple-io ram` 41.4 且启动 106 s、`--kv fp16` 42.8）——全部落在噪声内

**会让引擎启动失败：**
- `--no-spec` 和 `--no-pool` —— IQ 量化模型必须 `--spec>=2`；禁用 CPU 池后专家算力超出显存。

---

## 八、诊断手段（值得复用的）

- `STRATA_DECODE_TIMING=1` + `STRATA_VERIFY_PROFILE=1`：打印每个窗口的耗时分解。
  本机实测（**0.1.41**，中文 300 tok 请求）：**50.67 ms/窗口、1.91 tok/窗口 = 37.7 tok/s**；
  verify 44.31 ms（87%，其中 GPU-reach wait 0.00、CPU experts 0.00 → 专家全在显存，CPU 完全不参与）、draft 6.22 ms、commit 0.14 ms。
  GPU 各阶段没有单一瓶颈，是均衡的带宽受限负载。
- 原始校验吞吐 ≈ 55.5 tok/s，实际 37.7 是 **68% 效率**，损失来自草稿接受率。

---

## 九、参考

- Strata 仓库与文档：https://github.com/Niko1221/Strata
  - `docs/STRIX_HALO.md`（Linux 上的 Strix Halo 编译与调优，第 4/5/6 节）
  - `docs/COMMUNITY_BENCHMARKS.md`（社区跑分报告的格式与提交方式）
  - `docs/AMD_HIP.md`、`docs/DETAILS.md`、`docs/TROUBLESHOOTING.md`
- Strix Halo 社区：Strix Halo HomeLab wiki（`strixhalo-homelab.d7.wtf`）及其 Discord
- hal0.dev 论坛（Profiles & Benchmarks 版块）

---

*本指南所有数字均来自本机实测，标注"官方"的为 Strata 文档中的数值。未实测项已明确标注。*
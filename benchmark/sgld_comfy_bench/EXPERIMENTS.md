# SGLD × ComfyUI / vllm-omni 性能测试：实验清单（第 2 版，待 review）

> 计时采样修正（2026-10-13）：性能运行默认只采集 GPU 指标。4090 上 `/proc/<pid>/smaps_rollup` 的主机内存采样会阻塞 CUDA IPC 引用计数共享内存的映射/释放，产生周期性慢步。主机 PSS/RSS 需另跑显式 `H3_BENCH_SAMPLE_HOST_MEMORY=1` 的内存测量；这轮延迟不用于速度比较。未采样的主机内存留空，不能填 0。

日期：2026-10-12。第 2 版已按你的 10 条回复修改，变更见第 11 节。
状态：拍板后再合并 bench 分支、写交接文档和脚本。

模型：MiniMax-H3（重点）、LTX-2.5、Qwen-Image 2.1
卡：RTX 4090（24 GB，sm89）、RTX 5090（32 GB，sm120）、H200（141 GB，sm90）、B200（180 GB，sm100）。每台机器 2 张卡，独占，不用 docker。每种卡跑 1 卡和 2 卡两种拓扑。

---

## 0. 两组实验分别回答什么

| 组 | 对比 | 回答的问题 | 怎么保证公平 |
|---|---|---|---|
| **A 组** | SGLD 集成模式 vs 原版 ComfyUI | 功能是否完整、提速多少、多卡、显存和内存占用、每秒视频成本 | ComfyUI 版本、workflow、单文件权重、seed 全部相同，**只换 loader**（`UNETLoader` → `SGLDUNETLoader` + `SGLDOptions`） |
| **B 组** | SGLang 原生 serve vs vllm-omni serve | 同参数下的延迟差 | HF diffusers 权重、分辨率、帧数、步数、seed、并行拓扑相同；都走 OpenAI 兼容 API |

---

## 1. 代码版本（所有机器一致）

| 组件 | 版本 | 说明 |
|---|---|---|
| SGLang | fork `niehen6174/sglang` 新建分支 `bench/comfyui-h3-ltx-qwen` | main 合并三个 PR 头：H3 `c5ea93bc07`、LTX `d6e7ef8799`、Qwen `952b1c5f3e`。push 前在 H200 上跑单测和三模型冒烟 |
| ComfyUI | v0.39.0（最新 release，2026-10-05） | e2e 是在 v0.38.0（`6b747c04`）上验证的。冒烟时先试 v0.39.0，不通过就退回 v0.38.0 |
| vllm-omni | main `156ddb3f4a`（2026-10-12，当前最新） | 最新 release 是 v0.30.0，预发布版是 v0.31.0rc1。按"用最新的"固定到这个 commit，所有机器统一，单独 venv |
| comfy-kitchen、FA、Sage | 按卡编译，版本写进 `env.json` | |

---

## 2. 负载定义

**步数**：H3 和 Qwen 都改成 **20 步**；LTX 用蒸馏模型的固定 sigma，不变。

| ID | 模型 | 参数 | 视频长度 | 用途 |
|---|---|---|---|---|
| **W-H3-768** | H3 T2V | 1344×768，124 帧，24 fps，20 步，video shift 12 / audio shift 3 | 5.17 s | A、B 组主负载，所有卡都跑 |
| **W-H3-480** | H3 T2V | 864×480，121 帧，24 fps，20 步 | 5.04 s | 4090 / 5090 补充；和之前的 artifact 同分辨率同步数，可以直接对照 |
| W-H3-576 | H3 T2V | 1024×576，121 帧，20 步 | 5.04 s | 只在 B 组 4090 上用，对应 vllm-omni 的 rtx4090 profile |
| **W-H3-ref4** | H3 Ref2VA | 1344×768，124 帧，20 步，**4 张参考图**（`ref_image_size=match`），prompt 里用 `<Picture 1>`…`<Picture 4>` | 5.17 s | A 组 |
| **W-H3-ref8** | H3 Ref2VA | 同上，**8 张参考图** | 5.17 s | B 组 |
| W-H3-sweep | H3 T2V | 864×480，33 帧，8 步 | — | 只用于 P0 注意力扫描 |
| **W-LTX-2s** | LTX-2.5 T2V，官方 ComfyUI 模板 | 一阶段 640×360×121，8 个 sigma；x2 latent 上采样；二阶段 1280×720，3 个 sigma；24 fps，distilled，cfg 1 | 5.04 s | A 组 |
| **W-LTX-1s** | LTX-2.5 T2V | 960×544，121 帧，distilled 8 步 | 5.04 s | B 组 |
| W-LTX-2s-1088 | LTX-2.5 T2V | 二阶段到 1920×1088，121 帧 | 5.04 s | B 组，H200 / B200 |
| **W-QW-cfg1** | Qwen-Image 2.1 T2I | 1024×1024，20 步，cfg 1 | — | A、B 组 |
| **W-QW-cfg4** | Qwen-Image 2.1 T2I | 1024×1024，20 步，cfg 4 | — | 覆盖 CFG 路径（每步两次前向） |
| W-QW-edit | Qwen-Image 2.1 编辑 | 1024×1024，1 张参考图，20 步 | — | 功能 + 计时 |

每个负载固定一条 prompt，写进 `prompts.json`。质量对照另用 3 条 prompt。

---

## 3. 各卡可行性和权重选择

| | 4090 24G | 5090 32G | H200 141G | B200 180G |
|---|---|---|---|---|
| **H3 DiT** | pruned INT8 21 GB，必须开 layerwise offload | pruned INT8 21 GB，先试不开 offload | BF16 66 GB（无损基线），另测 INT8 34 GB | 同 H200 |
| H3 TE | INT8 27 GB（NVFP4 只能在 Blackwell 上跑），offload | NVFP4-AWQ 15.7 GB | NVFP4-AWQ | NVFP4-AWQ |
| H3 INT8 kernel | comfy-kitchen（sm89 没有 JIT） | JIT | JIT | JIT |
| **FastH3 / VDN** | BF16 66 GB → 在线 INT8（约 34 GB）+ offload；1 卡和 2 卡 TP2 都试 | 同左 | BF16，另测在线 INT8 | 同 H200 |
| **LTX-2.5 DiT** | INT8 ConvRot 21.5 GB + offload | INT8，DiT 和 TE 都 layerwise offload | INT8；有 BF16 单文件就加测 BF16 | 同 H200 |
| LTX TE | Gemma-4 12B INT8 15.4 GB，offload | 同左 | 常驻 | 常驻 |
| **Qwen-2.1 DiT** | INT8 7.3 GB；BF16 14.2 GB 加 TE offload | BF16 | BF16 + INT8 | BF16 + INT8 |
| 主机内存 | offload 每卡钉住约 55 GB；FastH3 / VDN 在线量化加载时还要约 70 GB | 同左 | — | — |

**FastH3 / VDN 在小卡上的风险**：这两个模型只有 66 GB 的 BF16 权重，没有 pruned / INT8 版本。小卡上必须同时开"在线 INT8 + layerwise offload"，这个组合还没验证过。

- 我会在交接前，在 H200 上把 resident 层数压到 5090 / 4090 的水平，先验证一次。
- 小卡跑不通的话，就记为"不可行"并写明原因，不算失败。
- vllm-omni 的 4090 / 5090 recipe 也是 2 卡 TP2 + offload。

---

## 4. P0：每张卡先做注意力扫描（不进报告）

只在 SGLD 一侧跑，用 W-H3-sweep，每个配置新起服务，1 次预热 + 3 次计时。

| ID | 后端 | 卡 |
|---|---|---|
| P0-1 | `fa`（4090 上是 FA2，H200 上是 FA3，5090 / B200 上可用就用 FA4） | 全部 |
| P0-2 | `torch_sdpa` | 全部 |
| P0-3 | cuDNN SDPA | 全部 |
| P0-4 | `sage_attn`（Sage2，有损） | 全部 |
| P0-5 | `sage_attn_3`（Sage3，有损） | 5090 / B200 |

**产出**：
- 每张卡的 **best-lossless** 后端，A、B 两组都用它。预期 H200 是 FA3，B200 是 cuDNN。
- 确认 Sage 能正常编译和加载。

---

## 5. A 组：SGLD vs 原版 ComfyUI

### 5.1 H3（重点）

**负载**：所有卡都跑 W-H3-768。4090 / 5090 再补一组 W-H3-480，覆盖下表打 ★ 的配置。4090 上 768p 如果 OOM，就以 480p 为主，并记录 OOM。

**精度基线**：H200 / B200 用 BF16 文件；4090 / 5090 用 pruned INT8，没有 BF16 参照。

**原版 ComfyUI**

| ID | 配置 | 卡 | 说明 |
|---|---|---|---|
| A-H3-01 ★ | 默认（PyTorch SDPA，smart memory） | 全部 | 主基线 |
| A-H3-02 | `--use-sage-attention` | 全部 | 和 A-H3-20 成对，同为有损 |
| A-H3-03 | 加 EasyCache 节点 | H200、5090 | 和 A-H3-22 成对 |
| A-H3-04 | INT8 文件（kitchen） | H200 / B200 | 和 A-H3-23 成对 |

**SGLD，1 卡，无损**

| ID | 配置 | 卡 | 说明 |
|---|---|---|---|
| A-H3-10 | SGLDOptions 全部默认 | 全部 | 直接替换 loader 的开箱效果 |
| A-H3-11 ★ | best-lossless 注意力（来自 P0） | 全部 | 无损最好成绩 |

**SGLD，1 卡，有损**（都在 A-H3-11 的基础上叠加）

| ID | 配置 | 卡 | 说明 |
|---|---|---|---|
| A-H3-20 | Sage2 | 全部 | |
| A-H3-21 | Sage3 | 5090 / B200 | |
| A-H3-22 | cache-dit（cookbook 默认 FN / BN / warmup / RDT / MC） | 全部 | |
| A-H3-23 | int8linear：INT8 ConvRot 文件 + JIT kernel | H200 / B200 | 小卡的基线本来就是 INT8 |
| A-H3-24 | int8linear，强制用 comfy_kitchen | H200 / B200 | 拆出 JIT kernel 本身的收益 |
| A-H3-25 ★ | FastH3（8 步蒸馏，`runtime_model_path` + `SGLDH3DistilledSigmas`） | 全部；小卡用在线 INT8 + offload | 原版 ComfyUI 跑不了 |
| A-H3-26 ★ | VDN（8 nfe，hybrid window attention） | 同上 | 原版 ComfyUI 跑不了 |
| A-H3-27 ★ | 叠加：int8 + Sage + cache-dit | 全部 | 20 步下的最快配置 |
| A-H3-28 | 叠加：FastH3 + int8 + Sage | 全部 | 整体最快配置 |

**SGLD，2 卡**

| ID | 配置 | 卡 | 说明 |
|---|---|---|---|
| A-H3-30 ★ | Ulysses 2 + best-lossless | 全部 | 4090 加 offload |
| A-H3-31 | TP 2 | 全部 | 4090 / 5090 用 TP2 + offload |
| A-H3-32 | Ulysses 2 + A-H3-27 的叠加 | 全部 | |
| A-H3-33 | 2 卡 + FastH3（H200 / B200 用 Ulysses2，小卡用 TP2） | 全部 | |
| A-H3-34 | 2 卡 + VDN（TP2） | 全部 | |

原版 ComfyUI 没有单请求多卡并行。2 卡的对照就是 A-H3-01，按"两个原版实例各占一张卡"折算：吞吐翻倍，单条成本和延迟都不变。

**Ref2VA，4 张参考图（W-H3-ref4）**

DiT 换成 ref2va 分区（H200 / B200 用 INT8 34 GB；小卡用 pruned INT8 21 GB）。参考图由 Qwen-Image 2.1 预先生成 8 张，放进套件，A、B 组共用。

| ID | 配置 | 卡 |
|---|---|---|
| A-H3-40 | 原版默认 | 全部 |
| A-H3-41 | SGLD best-lossless | 全部 |
| A-H3-42 | SGLD 叠加：int8 + Sage + cache-dit | 全部 |
| A-H3-43 | SGLD 2 卡（H200 / B200 用 Ulysses2，小卡用 TP2 + offload） | 全部 |

### 5.2 LTX-2.5

**负载**：W-LTX-2s（官方两阶段模板）。DiT 用 INT8 ConvRot；有 BF16 单文件的话，H200 / B200 加测 BF16。**cache-dit 这次不测。**

| ID | 配置 | 卡 |
|---|---|---|
| A-LTX-01 | 原版默认 | 全部 |
| A-LTX-02 | 原版 `--use-sage-attention` | 全部 |
| A-LTX-10 | SGLD 默认 | 全部 |
| A-LTX-11 | SGLD best-lossless | 全部 |
| A-LTX-20 | SGLD Sage2 | 全部 |
| A-LTX-21 | SGLD Sage3 | 5090 / B200 |
| A-LTX-23 | int8linear：BF16 文件 vs INT8 文件 | H200 / B200，需要有 BF16 文件 |
| A-LTX-30 | SGLD 2 卡 Ulysses 2 | 全部 |
| A-LTX-31 | SGLD 2 卡 Ulysses 2 + Sage | 全部 |

distilled 路径的 cfg 是 1，不适用 CFG 并行。

### 5.3 Qwen-Image 2.1

**负载**：W-QW-cfg1 和 W-QW-cfg4。**cache-dit 这次不测。**

| ID | 配置 | 卡 |
|---|---|---|
| A-QW-01 | 原版默认（BF16；4090 用 INT8） | 全部 |
| A-QW-02 | 原版 Sage | 全部 |
| A-QW-10 | SGLD 默认 | 全部 |
| A-QW-11 | SGLD best-lossless | 全部 |
| A-QW-20 | SGLD Sage2 | 全部 |
| A-QW-21 | SGLD Sage3 | 5090 / B200 |
| A-QW-23 | int8linear：INT8 ConvRot 文件 | 全部 |
| A-QW-30 | SGLD 2 卡 Ulysses 2（cfg1 和 cfg4 都跑） | 全部 |
| A-QW-31 | SGLD 2 卡 TP 2 | 全部 |

CFG 由 ComfyUI 负责，`enable_cfg_parallel` 会被拒绝，所以 A 组没有 CFG 并行。

### 5.4 功能完整性矩阵（只看能不能跑，不计时）

在 H200 和 5090 上跑。原版和 SGLD 各跑一次，记录 pass / fail / 不支持。

| ID | 用例 |
|---|---|
| F-01 | H3 T2V / FL2VA（首尾帧）/ Ref2VA |
| F-02 | H3 + turbo LoRA |
| F-03 | H3 奇数尺寸（如 1000×560） |
| F-04 | H3 FastH3 / VDN（原版：不支持） |
| F-05 | H3 2 卡 Ulysses / TP（原版：不支持） |
| F-06 | LTX-2.5 T2V 两阶段 / I2V |
| F-07 | LTX-2.3 FP8 all-in-one + distilled LoRA |
| F-08 | LTX 2 卡 SP |
| F-09 | Qwen T2I bf16 / INT8，编辑 1 张参考 / 2 张参考，奇数尺寸 |
| F-10 | Qwen 2 卡 |
| F-11 | 同一个 ComfyUI 进程里切换模型（H3 → Qwen → H3），显存能否释放 |

---

## 6. B 组：SGLang 原生 vs vllm-omni

两边都用 `serve`，同一个客户端脚本发请求，记录客户端 E2E 和服务端日志里的各阶段耗时。

**三类对比**
- **matched**：参数逐项相同，精度 BF16，不开有损选项，注意力同类（Hopper 上都用 FA3；做不到时两边都用 SDPA / cuDNN，并在表里注明）。
- **同类有损**：两边开同一类有损特性。实现方式不同（见 6.4），只能算"同类"，不算严格对齐。
- **recipe**：各自 cookbook 推荐的配置，代表用户实际能拿到的速度。

### 6.1 H3（权重 `MiniMaxAI/MiniMax-H3`）

| ID | 负载 | 拓扑 / 选项 | 卡 | 类型 |
|---|---|---|---|---|
| B-H3-01 | W-H3-480 | 1 卡 | H200、B200 | matched |
| B-H3-02 | W-H3-768 | 1 卡 | H200、B200 | matched |
| B-H3-03 | W-H3-768 | 2 卡 Ulysses 2 | H200、B200 | matched |
| B-H3-04 | W-H3-768 | 2 卡 TP 2 | H200、B200 | matched |
| B-H3-05 | W-H3-576 | 2 卡 TP 2 + layerwise offload，cuDNN | 4090 | matched（vllm-omni rtx4090 profile） |
| B-H3-06 | W-H3-768 | 2 卡 TP 2 + layerwise offload，cuDNN | 5090 | matched（vllm-omni rtx5090 profile 就是 1344×768、20 步） |
| B-H3-10 | W-H3-768 | 1 卡，在线 FP8（DiT） | H200、B200 | 同类有损 |
| B-H3-11 | W-H3-768 | 1 卡，在线 INT8（DiT） | H200、B200 | 同类有损 |
| B-H3-12 | W-H3-768 | 1 卡，Sage | H200、B200 | 同类有损 |
| B-H3-13 | W-H3-768 | 1 卡，cache-dit | H200、B200 | 同类有损 |
| B-H3-14 | W-H3-768 | 2 卡 TP 2，VDN（8 步） | H200、B200 | 同类有损；需确认两边用的是同一份 OpenVDN 权重 |
| B-H3-15 | W-H3-768 | 2 卡 TP 2 + offload，在线 FP8 | 4090、5090 | 同类有损 |
| B-H3-20 | W-H3-768 | 各自 recipe | 全部 | recipe |
| B-H3-30 | W-H3-ref8 | 1 卡 | H200、B200 | matched |
| B-H3-31 | W-H3-ref8 | 2 卡 Ulysses 2 | H200、B200 | matched |
| B-H3-32 | W-H3-ref8 | 2 卡 TP 2 + offload | 4090、5090 | matched |

两边对 Ref2VA 的上限相同：最多 9 张图，参考总数不超过 12，所以 8 张图在两边都合法。

**FastH3 不放进 B 组的对比**：vllm-omni 的 FastH3 是 FastVideo 的 4 步 preview LoRA，我们用的是 FastH3-8step-V2 合并权重，两者不是同一个模型。两边可以各跑一次，只作为参考数字列出。

### 6.2 LTX-2.5（`Lightricks/LTX-2.5-Diffusers`）

| ID | 负载 | 拓扑 / 选项 | 卡 | 类型 |
|---|---|---|---|---|
| B-LTX-01 | W-LTX-1s | 1 卡 | 全部（小卡按 recipe 开 offload） | matched |
| B-LTX-02 | W-LTX-1s | 2 卡 Ulysses 2 | 全部 | matched |
| B-LTX-03 | W-LTX-2s-1088 | 1 卡 | H200、B200 | matched |
| B-LTX-04 | W-LTX-2s-1088 | 2 卡 Ulysses 2 | H200、B200 | matched |
| B-LTX-10 | W-LTX-1s | 1 卡，在线 FP8 | 全部 | 同类有损（两边 cookbook 都有 FP8 recipe） |
| B-LTX-11 | W-LTX-1s | 1 卡，Sage | 全部 | 同类有损 |
| B-LTX-20 | W-LTX-1s | 各自 recipe | 全部 | recipe |

### 6.3 Qwen-Image 2.1（`Qwen/Qwen-Image-2.1`）

| ID | 负载 | 拓扑 / 选项 | 卡 | 类型 |
|---|---|---|---|---|
| B-QW-01 | W-QW-cfg1 | 1 卡 | 全部（4090 按 recipe 开 offload） | matched |
| B-QW-02 | W-QW-cfg4 | 1 卡 | 全部 | matched |
| B-QW-03 | W-QW-cfg4 | 2 卡 CFG 并行 | 全部 | matched |
| B-QW-04 | W-QW-cfg1 | 2 卡 Ulysses 2 | 全部 | matched |
| B-QW-05 | W-QW-edit | 1 卡 | H200、5090 | matched |
| B-QW-10 | W-QW-cfg1 | 1 卡，在线 FP8 | 全部 | 同类有损 |
| B-QW-11 | W-QW-cfg1 | 1 卡，在线 INT8 | 全部 | 同类有损 |
| B-QW-12 | W-QW-cfg1 | 1 卡，Sage | 全部 | 同类有损 |
| B-QW-20 | W-QW-cfg1 | 各自 recipe | 全部 | recipe |
| B-QW-21 | W-QW-cfg1，并发 1 / 2 / 4 | 1 卡 | H200 | 吞吐（可选） |

### 6.4 FP8 / INT8 的说明（报告里也会写）

两边都提供在线 FP8 和在线 INT8，权重都是盘上的 BF16，加载时量化。但量化方案不同，所以这些对比只能算"同类有损"：

| | SGLang | vllm-omni |
|---|---|---|
| FP8 | `--quantization fp8`：在线 FP8 W8A8 | `--quantization fp8`：在线 FP8 W8A8，走 vLLM 的 FP8 linear；H3 可以只量化 DiT 或只量化 TE |
| INT8 | `--quantization convrot_int8`：先做 ConvRot 旋转，再做 per-channel W8A8；sm90 / 100 / 120 用 JIT kernel，sm89 用 comfy-kitchen | `--quantization int8`：在线 W8A8 Int8，不做旋转 |
| 哪些层被量化 | 以日志为准 | 以日志为准 |

**报告写法**
- 每条有损结果都和**同一引擎自己的 BF16 结果**比，列出速度提升和质量变化（PSNR / SSIM，加帧拼图）。
- 两个引擎之间只比"同类有损配置下的绝对延迟"，并注明方案不同。
- 两边实际量化的层范围写进结果，对 H3 的 TE 是否量化也分开标注。

---

## 7. 指标和测量方法

### 7.1 运行规则
- **每个配置新起一个服务**，配置之间不复用进程。
- **预热**：1 次，分辨率和帧数与正式负载相同，步数减到 **3 步**。shape 一致，kernel 和显存分配就都已预热，耗时只有正式跑的 1/7 左右。FastH3 / VDN 的 sigma 表是固定的，预热跑完整的 8 步。
- **计时**：单次 < 60 s 的跑 5 次，≥ 60 s 的跑 3 次。取中位数，同时记 min / max。
- **避开 ComfyUI 的缓存**：A 组每次计时都把 noise seed 换成 seed + run_idx，否则 ComfyUI 会直接复用上次的输出。TE 输出在两边都会被缓存，所以热态 E2E 不含 TE；TE 耗时从冷启动那次单独取。
- **两张卡怎么用**：
  - 1 卡配置在 H200 / B200 上可以两张卡各跑一条，并行省时间。
  - 小卡上的 offload 配置要串行跑，避免两条共享 PCIe 和主机内存带宽，互相影响。
  - 2 卡配置独占整台机器。
- **端口**：每个服务用不同的端口和 master_port。

### 7.2 机器和资源使用记录

- **开跑前记录一次机器信息**（`machine.json`）：
  - CPU 型号和核数、内存总量、磁盘剩余空间
  - GPU 型号、驱动、CUDA、功耗上限
  - `nvidia-smi topo -m`（PCIe 还是 NVLink）
  - 各组件的 commit 和库版本
- **每条运行期间每 200 ms 采样一次**（`usage.csv`）：
  - 每卡：显存、利用率、功耗、SM 频率
  - 主机：进程树 RSS 总和、CPU 占用
  - 报告里取峰值和均值。

### 7.3 指标

| 指标 | 定义 | A 组 | B 组 |
|---|---|---|---|
| E2E | A 组：从提交 prompt 到视频保存完（不含加载）。B 组：从发请求到收到响应 | ✓ | ✓ |
| 去噪时间 | A 组取 SGLD worker / 原版采样节点的耗时；B 组取服务端日志 | ✓ | ✓ |
| 冷启动 | 从启动服务到第一条结果完成 | ✓ | ✓ |
| 峰值显存 / 平均利用率 / 平均功耗 | 每卡，取自 usage.csv | ✓ | ✓ |
| 峰值主机内存 | 进程树 RSS 总和的峰值 | ✓ | ✓ |
| GPU-s / 视频秒 | E2E × 卡数 ÷ 视频秒数（Qwen 用 GPU-s / 张） | ✓ | ✓ |
| $ / 视频秒 | GPU-s / 视频秒 × 单价 ÷ 3600（Qwen 用 $ / 1000 张） | ✓ | ✓ |
| 一致性 | 同 seed 下 SGLD 和原版的 latent 余弦相似度 | ✓ | 可选 |
| 有损质量 | 和同卡、同引擎的无损基线比：解码帧 PSNR / SSIM，加 3 条 prompt 的帧拼图供人工看。FastH3 / VDN 是蒸馏模型，不做像素级比较，只看拼图 | ✓ | ✓ |

### 7.4 结果格式
- 每次计时写一行 JSON 到 `<card>/<track>/results.jsonl`。
- 字段：`exp_id, card, n_gpu, engine, commit, workload, config, run_idx, e2e_s, denoise_s, cold_s, peak_vram_gb[], mean_util[], mean_power_w[], peak_rss_gb, status, error`。
- 视频和帧拼图保存在 `<card>/<track>/media/`。

---

## 8. GPU 单价

| 卡 | RunPod Community | **RunPod Secure（主口径）** | GetDeploying 按需中位数（交叉核对） |
|---|---|---|---|
| RTX 4090 | $0.34/h | **$0.74/h** | $0.47/h（最低 $0.33，2026-10-12） |
| RTX 5090 | $0.69/h | **$0.99/h** | $0.70/h（最低 $0.35） |
| H200 SXM | $3.59/h | **$4.59/h** | ≈ $4.29/h（2026-08-26） |
| B200 180G | $5.98/h | **$6.79/h** | $7.24/h（最低 $3.75，38 家，2026-09-14） |

来源：RunPod 定价页（2026-09-27 更新）；getdeploying.com 各卡页面。报告主要用 GPU-s，换价格时乘一下即可。

---

## 9. 工作量粗估

改成 20 步、预热只跑 3 步后，比第 1 版少了大约 60%。

| 部分 | 每张卡耗时 |
|---|---|
| A 组 H3：H200 / B200 约 20 个配置，每条约 6–8 分钟（含起服务） | 3–4 h |
| A 组 H3：4090 / 5090 加上 480p 的补充 | 4–6 h |
| A 组 LTX + Qwen | 约 2 h |
| B 组（含搭 vllm-omni 环境） | 3–4 h |
| **合计** | **每张卡约 12–16 h，四张卡可以并行** |

---

## 10. 交接内容（拍板后由我准备，给另一个 agent）

那个 agent 在你的 Mac 上运行，通过 ssh 连各台机器（包括本机 H200）。

1. **代码**：第 1 节的 bench 分支，以及 ComfyUI 和 vllm-omni 的固定 commit。
2. **环境**：每种卡一份安装步骤。
   - SGLang venv；FA / Sage / kitchen 按 `TORCH_CUDA_ARCH_LIST` 编译；vllm-omni 用独立 venv。
   - `check_env.py` 检查 commit、kernel 是否可用、磁盘、内存、卡数，并生成 `machine.json`。
3. **磁盘**：
   - H200 / B200 默认用 `/personal`，4090 / 5090 默认用 `/scratch`。
   - 脚本先用 `df` 比较两者剩余空间，选大的。
   - 需要的空间：小卡约 850 GB，H200 / B200 约 1.1 TB（A 组单文件 + FastH3 / VDN 的 BF16 和 VDN 预融合 + B 组 HF 权重 + 缓存和结果）。
4. **HF token**：
   - agent 从操作机上的 token 文件读取（路径单独提供，不写进仓库）。
   - 用 scp 拷到远端 `$HF_HOME/token`，权限设为 600。
   - 不打印、不写进日志、不提交。
5. **模型下载**：`download.py --card <card> --tracks A,B` 下载并链接到 ComfyUI/models。
6. **跑测脚本**：
   - `run_track_a.py` / `run_track_b.py` 按实验 ID 跑，支持断点续跑，在远端用 tmux / nohup 执行。
   - agent 定期查看进度。
   - `report.py` 负责汇总。
7. **结果回传**：每张卡的结果拉回操作机上的结果目录（路径单独提供）。之后你把整个目录给我，我汇总成报告。

---

## 11. 第 2 版改动（对应你的回复）

| # | 你的决定 | 文档里的改动 |
|---|---|---|
| 1 | LTX / Qwen 的 cache-dit 先不测 | A 组删掉 A-LTX-22 / A-QW-22；B 组 LTX / Qwen 也不测 cache-dit |
| 2 | 用 20 步 | H3、Qwen 改为 20 步，LTX 不变；4090 也以 768p 为主 |
| 3 | 预热步数少一些 | 预热 1 次，同 shape，3 步 |
| 4 | FastH3 / VDN 在 5090 / 4090 上也要跑 | 小卡用"在线 INT8 + offload"，1 卡和 TP2 都试；交接前我先在 H200 上模拟验证 |
| 5 | FP8 / INT8 也试，并说明 | B 组新增 FP8 / INT8 / Sage / cache-dit / VDN 等同类有损对比，6.4 节说明方案差异 |
| 6 | 用最新的 | vllm-omni 用 main `156ddb3f4a`，ComfyUI 用 v0.39.0 |
| 7 | 每台都是 2 卡、独占、不用 docker，记录资源使用 | 新增 7.2 节：`machine.json` 和 `usage.csv` |
| 8 | 结果放 Mac 本地目录 | 写进第 10 节 |
| 9 | 其余按建议 | 单价用 RunPod Secure；EasyCache 只测 H3；质量用 PSNR / SSIM 加拼图；LTX-2.3 和 Qwen 编辑只进功能矩阵 |
| 10 | HF token 路径 | 写进第 10 节 |
| 追加 | H3 Ref2VA：A 组 4 张参考图，B 组 8 张参考图 | 新增负载 W-H3-ref4 / W-H3-ref8 和实验 A-H3-40～43、B-H3-30～32 |

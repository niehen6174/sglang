# 5090 重跑说明：ComfyUI 显存修复后的 A 组

## 为什么重跑

上一轮 5090 上，SGLD 的 H3 和 LTX 无损配置比原版 ComfyUI 慢 2～10%。原因是：

1. ComfyUI 做完文本编码后，文本编码器还留在 GPU 上（H3 的 Qwen3-VL 约 14 GB）。
2. SGLD 的 worker 是另一个进程，ComfyUI 不知道它要加载 21 GB 的 DiT，所以不会先腾地方。
3. worker 加载到一半 OOM，套件只好自动开 layerwise offload。之后每一步都要把全部权重经 PCIe 搬一遍。

这次插件修好了：worker 加载 DiT 前、以及每次采样前，SGLD 会先让 ComfyUI 把自己的模型挪出显存（日志里会出现 `SGLD: moved ... off cuda:0`）。另外，套件在小卡上跑 TP2 时不再强制开 offload。

目标是确认：H3、LTX 不再需要 offload，速度至少和原版持平。

## 1. 更新代码

在之前的工作区里（`$WS` 是上次用的目录，比如 `/scratch/sgld_bench`）：

```bash
cd $WS/sglang
git fetch -q origin bench/comfyui-h3-ltx-qwen
git checkout -q origin/bench/comfyui-h3-ltx-qwen
git log --oneline -3      # 第一行应是 bench kit 的提交，往下能看到 "free ComfyUI's VRAM before the worker loads and samples"
```

插件是软链接到这个目录的，不需要重装。之前的 `kit.env` 继续用，`CARD=5090`、`LANES=1` 不变。

## 2. 用新的结果目录

不要覆盖上一轮的结果。新开一个目录，并沿用上一轮 P0 选出的注意力后端：

```bash
cd $WS/sglang/benchmark/sgld_comfy_bench
export SUITE=$WS/results/5090-vramfix
mkdir -p $SUITE/A
cp $WS/results/5090/A/best_attention.json $SUITE/A/
cp $WS/results/5090/machine.json $SUITE/ 2>/dev/null || ./run.sh check
```

## 3. 跑这些实验

原版 ComfyUI 的基线也一起重跑，保证新旧数字在同一台机器、同一时间段测出来。

```bash
# 1 卡（约 2.5 小时）
./run.sh a --gpus 1 --only A-H3-01,A-H3-01-480,A-H3-10,A-H3-11,A-H3-11-480,A-H3-20,A-H3-22,A-H3-27,A-H3-27-480,A-H3-40,A-H3-41,A-H3-42,A-LTX-01,A-LTX-10,A-LTX-11,A-LTX-20,A-QW-01-cfg1,A-QW-11-cfg1,A-QW-01-cfg4,A-QW-11-cfg4 > $WS/rerun-a1.log 2>&1

# 2 卡（约 1 小时）
./run.sh a --gpus 2 --only A-H3-30,A-H3-30-480,A-H3-31,A-H3-32,A-H3-43,A-LTX-30,A-LTX-31,A-QW-30-cfg1,A-QW-31-cfg1,A-QW-30-cfg4,A-QW-31-cfg4 > $WS/rerun-a2.log 2>&1

# 功能矩阵（上一轮 H3、LTX 的用例都是 OOM，约 40 分钟）
./run.sh functional > $WS/rerun-f.log 2>&1

./run.sh quality
./run.sh report
```

FastH3 / VDN（A-H3-25 / 26 / 28 / 33 / 34）这次不用重跑：它们只有 66 GB 的 BF16 权重，在 5090 上本来就必须开 offload。

## 4. 跑完检查

```bash
# 应该能看到 SGLD 把文本编码器挪走的日志
grep -h "SGLD: moved" $SUITE/A/runs/*/server.log | sort | uniq -c

# 这几项的 offload_fallback 应该是 false，峰值显存应该在 25～30 GB（上一轮开了 offload 只有 11 GB）
python3 - <<'EOF'
import json, os
for l in open(os.environ["SUITE"] + "/A/results.jsonl"):
    r = json.loads(l)
    if r.get("phase") == "summary" and r["exp_id"] in ("A-H3-11", "A-H3-41", "A-LTX-11", "A-H3-31"):
        print(r["exp_id"], r["status"], r.get("e2e_s", {}).get("median"), "offload_fallback=", r.get("offload_fallback"), r.get("peak_vram_gib"))
EOF
```

如果 `A-H3-11` 仍然出现 `offload_fallback=true`，说明 32 GB 依然放不下，把这一项 `runs/A-H3-11__w/server.log` 里 OOM 那几行一起带回来。

## 5. 回传

把 `$SUITE`（即 `$WS/results/5090-vramfix/`）整个目录拉回操作机上约定的结果目录，命名为 `5090-vramfix`，和上一轮的 `5090` 放在一起。日志 `$WS/rerun-*.log` 也一起带回。

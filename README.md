# DeepSeek-V4.1-Flash on 4×CMP 170HX — PP4 + shadow KV + EXL3 2bpw + DSpark

在四张 CMP 170HX（sm80、PCIe Gen2 x4、无 P2P、每卡 64GB）上跑满血 764B 的
DeepSeek-V4.1-Flash：**PP4 流水线 + shadow KV + EXL3 2bpw 量化 + DSpark 投机解码 +
512k 上下文 + 原生视觉**，一个引擎同时伺候文本和图像。

## 结果（2026-09-14 实测，干净引擎）

| 维度 | 数字 |
|---|---|
| 质量 | 7/7（计数 400/400、陷阱题、推理、代码结构） |
| 视觉 | 2/2（形状/颜色/文字 OCR、多图联合、发票金额精确读出） |
| 单流 decode | 计数 66.7 / 散文 25.0 / **思考 79.0** / 代码 44.3 tok/s |
| 并发 decode-only | C4 76 → C24 204 → C32 219 tok/s（流水线饱和平台 ~220） |
| prefill（nonce 口径） | 2k 1367 / 8k 1453 / 16k 2119 tok/s（37% MFU） |
| KV 池 | **590 万 token**（524k 窗口 11.25 路并发） |
| 启动 | ~7.5 min（TileLang 持久缓存后） |

单卡 64GB×4 = 256GB 装不下 510GB 的原版 —— 路线是 **EXL3 2bpw（334GB）+
engram 表钉 189GiB 宿主内存（UVA）+ 压缩 KV fp8**。

## 硬件要求

- 4× GPU ≥ 64GB，sm80 可用（本项目在解锁 8GB→64GB 的 CMP 170HX 上验证）
- 宿主 RAM ≥ 373GB（engram pin 189GiB + 余量 ≥ 24GiB 是硬线）
- 磁盘 ~700GB（EXL3 包 334GB + 原版 476GB 如需重转换）

## 组装（从零）

```bash
# 1) 官方镜像 (自带 nvcc, Triton, lmcache)
docker pull vllm/vllm-openai:deepseekv41-flash-0909

# 2) dsv41reap overlay: shadow KV 的核心 (Apache-2.0)
git clone https://github.com/zebgop-ops/dsv41reap-pp.git
#    把 overlay/vllm/ 下全部 54 个 .py 拷进镜像的
#    /usr/local/lib/python3.12/dist-packages/vllm/ 对应路径
#    把 overlay/hybrid/ 拷进 site-packages/ (PYTHONPATH 不进 worker 进程, 必须落包)
#    还需要 v19-lineage 的 vllm/models/deepseek_v4_1/ampere/ (SM8x sparse attention)

# 3) 4 个补丁 (本 repo patches/, 全部幂等, 按序打):
python3 patches/pp_relay_img_ids.py  <vllm>/models/deepseek_v4_1/nvidia/model.py
python3 patches/engram_sm80_fp8.py   <vllm>/models/deepseek_v4_1/common/engram.py
python3 patches/moe_cand_key_fix.py  <vllm>/v1/worker/gpu/model_runner.py
python3 patches/plugin_v150_shim.py  <vllm_exl3>/exl3.py        # exllamav3 v1.5.0 用

# 4) 插件 + 内核: vllm-exl3 (AGPL-3.0) + exllamav3 v1.5.0 ext
#    exllamav3_ext 用容器内 nvcc 编 (TORCH_CUDA_ARCH_LIST=8.0);
#    cusparse.h 缺头 → pip install nvidia-cusparse-cu13 等, 头拷进 /usr/local/cuda/include

# 5) 权重: EXL3 2bpw 转换 (exllamav3 convert.py), docker commit 成自己的镜像

# 6) 启动
bash serve/launch-pp4.sh
```

## 基准

```bash
cd bench && python3 accept.py        # 质量/视觉/单流/并发/prefill 一键全套
```

尾行输出 `REPORT_JSON`，做历史对比。**测量纪律见 bench/README 与 FINDINGS——
违反必出假数（本项目四次翻车实录）**。

## 为什么是 PP4 而不是 TP

Gen2 x4 无 P2P 的 TP allreduce 税 ≈ TP1 全 6 激活专家的翻倍计算，decode 打平；
PP4 赢在 prefill、KV 池、以及 kv-sharing 组跨 rank 的 shadow KV 重算。详见 FINDINGS。

## 致谢

- [zebgop-ops/dsv41reap-pp](https://github.com/zebgop-ops/dsv41reap-pp) — shadow KV overlay（Apache-2.0）
- [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) — EXL3 vLLM 插件（AGPL-3.0）
- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — EXL3 量化与内核
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — sm80 sparse-MLA 路线的源头

License: Apache-2.0（本 repo 只分发我们自己的补丁脚本/配置/基准；
运行时依赖的上游组件遵循各自许可，vllm-exl3 插件为 AGPL-3.0）

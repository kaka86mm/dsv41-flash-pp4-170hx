# FINDINGS — 移植全记录与踩坑档案

从 "NotImplementedError: Compressed-KV source not found" 到生产，共 6 层问题。
以及四次性能测量翻车实录。按时间顺序，全部实测。

## 1. 移植六层坑

1. **hybrid 模块必须落 site-packages** — `PYTHONPATH=/overlay` 不进 worker 子进程。
   `from hybrid.pp_shadow import ...` 在 rank>0 的 worker 里 ImportError。
2. **ampere 模块缺失** — overlay 的 model.py 引
   `vllm.models.deepseek_v4_1.ampere.ampere_sparse`（SM8x sparse attention，
   vllm-backport 血统），官方镜像没有。从 backport 镜像整目录拷入。
3. **exllamav3_ext 缺失** — EXL3 CUDA 扩展整包（exllamav3/ + 顶层 .so）拷入。
   裸 import 报 libc10.so 找不到 = 没先 import torch，不是 ABI 问题。
4. **dsv4_img_ids PP relay** — 官方镜像的 vision 版 FFN 在 bias_vl 存在时每个 MoE
   层都要 input_ids；PP 下游 rank 拿不到（model_runner 显式置 None）。backport 的
   解法 = IntermediateTensors 里接力 `dsv4_img_ids`。三处修改：
   make_empty_intermediate_tensors 声明 / forward 收 / ship 发
   （`patches/pp_relay_img_ids.py`）。上游 vllm-backport #76 后续做了同款修复。
5. **engram Triton kernel sm80 fp8** — `tl.load` fp8e4nv 指针在 sm80 编不过
   （Triton 只认 fp8e4b15/fp8e5）。backport 的解法 = uint8 view + `_decode_fp8_f32`
   ALU 解码（`patches/engram_sm80_fp8.py`）。
6. **dsv41_cand 键不对称** — ship_cand 的 rank 在自己的 dummy spec 里声明
   `dsv41_cand`（它生产这个键），但 model_runner 的拷贝循环把 spec 所有键都当
   "上游发来的" 读 → KeyError。跳过缺失键（`patches/moe_cand_key_fix.py`）。

## 2. 关键开关

- **`VLLM_USE_V2_MODEL_RUNNER=0` 必须** — dspark 移植 + engram piecewise 静态 meta
  全在 V1 runner；V2 直接 raise "dspark with pipeline parallel is not supported"。
- DSpark 配置：`{"method":"dspark","num_speculative_tokens":5}`；
  `use_local_argmax_reduction` 不支持（overlay 版 proposer 无 get_top_tokens）。
- **分区 10,10,11,9**：rank3 = 9 层 + 草稿模型，KV 池从 292 万 → 455 万（+56%）。
  shadow 机制允许任意内切，10/21/30 的切法实测与 10,10,10,10 单流无差。

## 3. 已试并否决（别再踩）

| 尝试 | 结果 | 原因 |
|---|---|---|
| ngram 投机 | **7.6 t/s 灾难** | V1 runner 无图路径 + 每步流水线气泡 |
| DSV41_DSPARK_CONF=0.7（草稿截断） | 散文 -35% 代码 -30% C16 -31% | 他们的 CPU 专家架构验证成本随 token 走；我们全 GPU 带宽型每步固定开销主导，截断只减产出 |
| FULL_DECODE_ONLY 图 | 与 PIECEWISE 持平 | 单流瓶颈在流水线串行，不在图模式 |
| seqs 64 | C64 引擎聚合 ~190 < C32 219 | 平台在 ~220，加并发纯浪费 |

## 4. 测量翻车档案（四次，全部被用户当场识破）

1. **单流 15.4 t/s 假数** — 首发请求吃冷启 JIT（Triton/graph 编译）+ t/s 分母混入
   TTFT。教训：流式分离 TTFT，热身 2 轮取第 2。
2. **"C16 聚合 273，v19 的 2.9 倍"假数** — 纯计数 prompt 的投机验收率 100%，
   聚合虚高；且与 v19 的对比口径不同。教训：并发必须混合场景，对比必须同脚本同口径。
3. **prefill 42k tok/s 假数** — nonce 放在提示词**尾部**，前缀缓存按最长公共前缀
   全命中，只算了尾部几十个 token。物理检验：超 4×170HX FLOPs 上限 7 倍必假。
   教训：nonce 必须在**头部**；拿 `2×激活参数×token数÷tok/s` 对比硬件峰值自检。
4. **"PP 比 TP 慢"误判** — 上述三条叠加出的错误结论。同口径 A/B 后：decode 打平
   （TP1 全专家 ≈ 省下的 allreduce），PP4 真赢面在 prefill/KV/代码栈新旧。

## 5. 运维

- **僵尸请求**：客户端断流可留 Running 态请求（~25 万 KV token），引擎周期性
  0 tok/s 停顿，decode 全线腰斩且冷却不恢复（易误判热降频/配置回归）。检测：
  空闲时日志 `Running>0 且 generation 0.0 tok/s`。修复：重启引擎。
- **docker rm -f 杀不死**：容器卡 shutdown 时 for 循环重试。
- **启动 9→7.5 min**：TileLang JIT 持久化（`-v ~/tilelang-cache:/root/.tilelang`）。
  剩余 = engram 189GiB 页锁定 ~3min + 分片/arena 2.6min + graphs 1.9min（结构性）。

## 6. 性能天花板（结构性，说透）

- 单流思考 ~80 t/s = 4 级流水串行 + 每 rank 全 6 激活专家。Gen2 无 P2P 拓扑决定。
- 并发平台 ~220 tok/s = 流水线饱和。下一个杠杆是 exllamav3 v1.5.0 的
  `exl3_moe_coop` 两段式内核（插件仍在旧入口 + 尾参 shim，等上游移植）。
- prefill 16k=2119 tok/s @ 37% MFU，EXL3 反量化路径的合理值。

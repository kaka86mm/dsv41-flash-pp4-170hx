# bench — 验收基准套件

DeepSeek-V4.1-Flash PP4 栈的一键验收 (质量/视觉/单流/并发/prefill)。

## 验收套件 (纯标准库, 宿主 python3 直接跑)

```bash
python3 accept.py                                    # 全套
python3 accept.py --skip vision                      # 跳视觉
python3 accept.py --concs 4,16                       # 只测两档并发
python3 accept.py --base http://localhost:8095     # 换地址
```

输出末尾 `REPORT_JSON {...}` 可 grep 出来做历史对比。

## 测量纪律 (三次翻车换来的, 违者出假数)

1. **流式分离 TTFT** — t/s 混入 TTFT 的单流数字可偏离 3 倍
2. **热身 2 轮取第 2** — 首发请求吃冷启 JIT (尤其 graph/Triton)
3. **prefill nonce 必须在头部** — 前缀缓存按最长公共前缀匹配, 尾部 nonce 无效 (曾假出 42k tok/s, 超硬件 FLOPs 上限 7 倍)
4. **并发必须混合场景** — 纯计数 prompt 投机验收率 100%, 聚合虚高
5. **pybind 探测别用参数名** — doc 是 arg0/arg1 编号

## 基线 (2026-09-14 03:25, v13 = PP4+DSpark+10,10,11,9+seqs32+512k, 干净引擎)

- 质量: **7/7** 视觉: **2/2**
- 单流 decode: 计数 66.7 / 散文 25.0 / 思考 79.0 / 代码 44.3 tok/s
- 并发 decode-only: C4 76 / C8 103 / C16 164 / C24 204 / C32 219 (平台 ~220)
- prefill (头nonce): 2k=1367 / 8k=1453 / 16k=2119 tok/s (16k 新提示词 TTFT≈7.5s; 重复内容走前缀缓存另算)
- KV 池: 590 万 token (524k×11.25 路)

## 运维警示

**僵尸请求**: 客户端断流可能留下 Running 状态的请求 (占 ~25万 KV token), 引擎周期性 0 tok/s 停顿, 全部 decode 腰斩 (思考 79→33). 
检测: 空闲时 `docker logs | grep 'generation throughput'` 看 Running>0 且 0.0 tok/s. 修复: 重启引擎.

## 引擎


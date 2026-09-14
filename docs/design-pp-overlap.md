# 设计：消除 PP4 流水线空闲（草稿重叠 / 双 microbatch）— 未部署

## 问题（2026-09-15 实测）

DSpark 起草每步 **16-18ms ≈ 单流步时 35%**（inputs+meta 5.6 / forward 6.7-8.4 /
ctx_insert 1.7 / sample 2.0），全部发生在 rank3 且**完全串行**——此时 rank0-2 空闲。
叠加 4 级流水本身的气泡，单流 ~47ms/步里大量时间不在算。

## Plan A：启用官方 async scheduling（推荐先试，中等工作量）

官方镜像自带 async scheduling，但被保守禁用：

```
vllm/config/vllm.py:1321  # dsv41 dspark: async scheduling under PP needs
                          # [num_reqs, 1] sampled ids; the drafted
                          # [num_reqs, K+1] path is handled synchronously
:1326  self.scheduler_config.async_scheduling = False
```

**工作项**：
1. 读 vllm/v1/core/sched 的 async 路径 + spec manager，确认采样 id 契约
2. DSparkProposer 产出 [B, K+1] → 适配 async scheduler 期望的 [B,1] 交接
   （或让 scheduler 接受 drafted ids）
3. 验证：质量 7/7 + 验收率不变 + PIECEWISE 图正常

**预期收益**：host 侧调度与当前步 GPU 执行重叠，估 +5-10% 单流。
**风险**：中——动 scheduler/proposer 契约边界，spec 状态机是雷区。

## Plan B：真·双 microbatch 流水（大收益，暂不做）

两个请求组同时在飞：rank0-2 处理 B 组时 rank3 起草 A 组。需要：
- scheduler 发两个交错的 execute_model 流
- multiproc executor 脱离 lockstep 广播
- spec 状态机按 microbatch 分离

vLLM V1 是 lockstep 架构，这是 scheduler 重写（估 2-4 周，高风险）。
**决策：不自建，盯上游**（vLLM main 的 PP 连续批处理进展）。

## Plan C（已就绪）：压草稿自身成本

`patch_dspark_meta_cache.py`（overlay 目录，未应用）——inputs+meta 的 5.6ms
python 开销缓存化，已过独立代码审查。部署窗口跑一次，配
DSV41_DEBUG_DSPARK=1 前后对比计时。

## 收益地图（实测锚点）

| 手段 | 预期单流 | 状态 |
|---|---|---|
| 现状 v16b | 85 t/s | 生产 |
| + Plan C（草稿缓存化） | ~89-90 | 补丁就绪待窗口 |
| + Plan A（async sched） | +5-10% 再叠 | 设计定案 |
| + Plan B（双 microbatch） | 理论 ~120+（填满 3/4 空闲 rank） | 不做，盯上游 |

#!/usr/bin/env python3
"""一键验收: warmup → 质量 → 视觉 → 单流 decode → 并发 → prefill.
用法: python3 accept.py [--base http://localhost:8095] [--model deepseek-v4.1-flash]
教训固化: ①流式分 TTFT (t/s 混 TTFT = 假数) ②热身2轮取第2 (首发吃冷启JIT) ③prefill 必须 nonce (固定文本=缓存命中) ④并发场景必须混合 (纯计数投机100%虚高)"""
import argparse, json, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import stream_chat, SCEN, run_single, run_conc, run_prefill
import quality, vision

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8095")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--skip", default="", help="逗号分隔跳过: quality,vision,single,conc,prefill")
    ap.add_argument("--concs", default="4,8,16,24,32")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    t0 = time.time()
    report = {"base": a.base, "model": a.model, "ts": time.strftime("%F %T")}

    print(f"== 验收 {a.base} model={a.model} ==")
    print("-- warmup (每场景一发, 不计分) --")
    for name, msg, mt, th in SCEN:
        stream_chat(a.base, a.model, msg, mt, th)
    print("   done")

    if "quality" not in skip:
        p, t = quality.run(a.base, a.model)
        report["quality"] = f"{p}/{t}"
    if "vision" not in skip:
        p, t = vision.run(a.base, a.model)
        if t: report["vision"] = f"{p}/{t}"
    if "single" not in skip:
        print("=== 单流 decode (热身后第2轮) ===")
        r = run_single(a.base, a.model)
        report["single"] = {}
        for name, x in r.items():
            print(f"  {name}单流: TTFT {x['ttft']:.2f}s | {x['ct']} tok | decode {x['decode']:.1f} tok/s")
            report["single"][name] = round(x["decode"], 1)
    if "conc" not in skip:
        print("=== 并发 (混合场景, decode-only) ===")
        report["conc"] = {}
        for C in [int(x) for x in a.concs.split(",")]:
            x = run_conc(a.base, a.model, C)
            print(f"  C{C:<2d}: {x['tok']} tok | 墙钟 {x['wall']:.1f}s | 墙钟 {x['wall_tps']:.1f} | decode-only {x['decode_tps']:.1f} tok/s")
            report["conc"][f"C{C}"] = round(x["decode_tps"], 1)
    if "prefill" not in skip:
        print("=== prefill (nonce, 3发取最大) ===")
        r = run_prefill(a.base, a.model)
        report["prefill"] = {}
        for t, v in r.items():
            print(f"  prefill {t:>6d} tok: {v:8.0f} tok/s")
            report["prefill"][str(t)] = round(v)

    print(f"-- 总耗时 {time.time()-t0:.0f}s --")
    print("REPORT_JSON " + json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()

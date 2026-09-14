"""共享请求层: 流式 chat (分 TTFT), 场景定义. 只用标准库."""
import json, time, urllib.request, threading, random

def stream_chat(base, model, msg, max_tokens, thinking=False, timeout=3600, image=None, extra_msgs=None):
    """返回 {ttft, ct, total, decode}. image=b64(png). TTFT=首个内容 token."""
    content = []
    if image:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}})
    content.append({"type": "text", "text": msg})
    body = {"model": model, "messages": [{"role": "user", "content": content if image else msg}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": thinking}}
    if extra_msgs: body["messages"] = extra_msgs
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; ct = 0; text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: ") or line == b"data: [DONE]\n": continue
            d = json.loads(line[6:])
            if d.get("usage"): ct = d["usage"]["completion_tokens"]
            if d.get("choices"):
                delta = d["choices"][0].get("delta", {})
                c = delta.get("content")
                if c:
                    if ttft is None: ttft = time.time() - t0
                    text.append(c)
    total = time.time() - t0
    if ttft is None: ttft = total
    dec = (ct - 1) / max(total - ttft, 1e-9) if ct > 1 else 0.0
    return {"ttft": ttft, "ct": ct, "total": total, "decode": dec, "text": "".join(text)}

# 场景: (名字, prompt, max_tokens, thinking)
SCEN = [
    ("计数", "从1数到400，每行一个数字，不要任何其他文字。", 500, False),
    ("散文", "写一篇500字左右的中文散文，主题：深夜的机场。不要分点。", 700, False),
    ("思考", "一个笼子里有鸡和兔共35个头94只脚，鸡兔各几只？给出完整推理过程。", 700, True),
    ("代码", "用Python写一个LRU缓存类，带完整注释和使用示例。", 700, False),
]

BASE_ZH = ("大模型的推理性能取决于内存带宽与调度策略，流水线并行场景下激活值通过互联传递，"
           "长上下文检索需要稀疏注意力机制支撑，工程上还要考虑显存碎片与内核调度的相互影响，"
           "以及多级缓存的一致性维护成本。")

def nonce_prefill_prompt(target):
    """nonce 必须在头部 — 前缀缓存按最长公共前缀匹配, 尾部 nonce 无效
    (曾实测尾部 nonce 假出 42k tok/s, 超硬件 FLOPs 上限 7 倍)."""
    s = f"(编号{random.random()}): " + BASE_ZH
    while len(s) < target / 0.62: s += BASE_ZH
    return s[:int(target / 0.62)] + "\n\n只回复：好"

def run_conc(base, model, C, rounds=2):
    """混合场景并发, 第二轮为准. decode-only = 总token / (墙钟 - 最晚首token)."""
    for i in range(rounds):
        res, lock = [], threading.Lock()
        def w(k):
            name, msg, mt, th = SCEN[k % 4]
            r = stream_chat(base, model, msg + f" (变体{k})", mt, th)
            with lock: res.append(r)
        t0 = time.time()
        ts = [threading.Thread(target=w, args=(k,)) for k in range(C)]
        [t.start() for t in ts]; [t.join() for t in ts]
        wall = time.time() - t0
        tot = sum(r["ct"] for r in res)
        last_ttft = max(r["ttft"] for r in res)
        if i == rounds - 1:
            return {"C": C, "tok": tot, "wall": wall,
                    "wall_tps": tot / wall, "decode_tps": tot / (wall - last_ttft)}
    return None

def run_single(base, model, rounds=2):
    out = {}
    for name, msg, mt, th in SCEN:
        for i in range(rounds):
            r = stream_chat(base, model, msg, mt, th)
            if i == rounds - 1:
                out[name] = r
    return out

def run_prefill(base, model, targets=(2000, 8000, 16000), reps=3):
    out = {}
    for t in targets:
        rates = []
        for _ in range(reps):
            r = stream_chat(base, model, nonce_prefill_prompt(t), 5)
            rates.append(t / 0.62 / r["ttft"])
        out[t] = max(rates)
    return out

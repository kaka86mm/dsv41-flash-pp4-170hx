"""质量验收: 计数精确性 / 陷阱题 / 推理 / 代码结构 / 中文常识. 全部自动判分."""
import re
from common import stream_chat

def check_counting(text):
    nums = re.findall(r"(?m)^\s*(\d+)\s*$", text)
    seq = [int(n) for n in nums]
    expect = list(range(1, 401))
    if seq == expect[:len(seq)] and len(seq) >= 395:
        return True, f"序列完整 {len(seq)}/400"
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(seq, expect)) if a != b][:3]
    return False, f"长度{len(seq)} 首错{bad if bad else '(尾部截断?)'}"

def check_contains(text, *keywords):
    miss = [k for k in keywords if k.lower() not in text.lower()]
    return (not miss), ("全中" if not miss else f"缺 {miss}")

def check_code(text):
    ok1, _ = check_contains(text, "class", "def get", "def put")
    ok2 = ("OrderedDict" in text) or ("dict" in text) or ("heapq" in text)
    return ok1 and ok2, f"结构{'✓' if ok1 else '✗'} 数据结构{'✓' if ok2 else '✗'}"

CASES = [
    ("陷阱-9.11vs9.8", "9.11和9.8哪个大？一句话。", 150, False,
     lambda t: check_contains(t, "9.8")),
    ("推理-鸡兔", "笼子里有鸡和兔共35个头94只脚，鸡兔各几只？只给答案。", 900, True,
     lambda t: check_contains(t, "23", "12")),
    ("常识-烤鸭", "北京正经烤鸭一只大概多少钱？一句话。", 150, False,
     lambda t: check_contains(t, "元")),
    ("数学-百分数", "一件衣服先涨价20%再降价20%，最终比原价贵还是便宜？一句话。", 150, False,
     lambda t: check_contains(t, "便宜")),
    ("日期-星期", "2026年9月14日是星期几？只回答星期X。", 900, True,
     lambda t: check_contains(t, "星期一")),
]

def run(base, model):
    print("=== 质量 ===")
    passed = total = 0
    # 计数
    r = stream_chat(base, model, "从1数到400，每行一个数字，不要任何其他文字。", 1400)
    ok, detail = check_counting(r["text"]); total += 1; passed += ok
    print(f"  计数1-400      {'✓' if ok else '✗'} {detail}")
    for name, msg, mt, th, fn in CASES:
        r = stream_chat(base, model, msg, mt, thinking=th)
        ok, detail = fn(r["text"]); total += 1; passed += ok
        ans = r["text"].strip().replace("\n", " ")[:60]
        print(f"  {name:14s} {'✓' if ok else '✗'} {detail} | {ans}")
    # 代码
    r = stream_chat(base, model, "用Python写一个LRU缓存类，带注释和使用示例。", 700)
    ok, detail = check_code(r["text"]); total += 1; passed += ok
    print(f"  {'代码-LRU':14s} {'✓' if ok else '✗'} {detail}")
    return passed, total

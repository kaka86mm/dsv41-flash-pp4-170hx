"""视觉验收: 形状/颜色/文字 OCR/多图联合. PIL 缺席则跳过 (1.28 容器外宿主有 python3-PIL? 自动探测)."""
import base64, io, json

def _mk_images():
    from PIL import Image, ImageDraw
    out = {}
    img = Image.new("RGB", (320, 240), "white"); d = ImageDraw.Draw(img)
    d.text((50, 20), "TEST 123", fill="black")
    d.rectangle([40, 60, 160, 200], fill="#d32f2f")
    d.ellipse([190, 60, 300, 200], fill="#1976d2")
    b = io.BytesIO(); img.save(b, "PNG"); out["shapes"] = base64.b64encode(b.getvalue()).decode()
    img2 = Image.new("RGB", (400, 120), "white"); d2 = ImageDraw.Draw(img2)
    for i, s in enumerate([" invoice #2026-0913", " TOTAL: $1,337.42", " VAT: 8%"]):
        d2.text((15, 15 + i*35), s, fill="black")
    b2 = io.BytesIO(); img2.save(b2, "PNG"); out["ocr"] = base64.b64encode(b2.getvalue()).decode()
    img3 = Image.new("RGB", (400, 280), "#87ceeb"); d3 = ImageDraw.Draw(img3)
    d3.polygon([(60,220),(200,80),(340,220)], fill="#4a7c3a")
    d3.rectangle([0,220,400,280], fill="#3a7ca5")
    d3.ellipse([300,30,360,90], fill="#ffd700")
    b3 = io.BytesIO(); img3.save(b3, "PNG"); out["scene"] = base64.b64encode(b3.getvalue()).decode()
    return out

def run(base, model):
    print("=== 视觉 ===")
    try: imgs = _mk_images()
    except ImportError:
        print("  [跳过] PIL 不可用"); return 0, 0
    from common import stream_chat
    passed = total = 0
    r = stream_chat(base, model, "描述图中所有元素，一句话。", 200, image=imgs["shapes"])
    ok = all(k in r["text"] for k in ("红", "蓝")) and "123" in r["text"]
    total += 1; passed += ok
    print(f"  形状+文字OCR     {'✓' if ok else '✗'} | {r['text'].strip()[:70]}")
    # 多图
    body_imgs = [imgs["ocr"], imgs["scene"]]
    import urllib.request
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + body_imgs[0]}},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + body_imgs[1]}},
               {"type": "text", "text": "第一张图发票总额是多少？第二张图有几个几何元素？"}]
    r2 = stream_chat(base, model, "", 300, extra_msgs=[{"role": "user", "content": content}])
    ok2 = "1,337.42" in r2["text"] and ("3" in r2["text"])
    total += 1; passed += ok2
    print(f"  多图+小字OCR     {'✓' if ok2 else '✗'} | {r2['text'].strip()[:90]}")
    return passed, total

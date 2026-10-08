"""ScreenSpot-Pro grounding evaluation for UI-TARS served by vLLM (OpenAI-compatible API).

Measures zero-shot GUI grounding on high-resolution professional software before any
Archicad training: given a screenshot and an instruction, can the model click inside the
target element? Uses the same prompt and coordinate convention as the rollouter
(dart_rollouter/src/core/prompts.py, src/utils/ui_tars_utils.py): UI-TARS-1.5 is a Qwen2.5-VL
model that answers in pixels of the smart_resized image, which are mapped back to the original.

Data (MIT licence, about 3.4 GB; not downloaded automatically unless --download is given):
    https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
    <data-dir>/annotations/*.json   list of {img_filename, img_size [w,h], bbox [x1,y1,x2,y2] px,
                                      instruction, application, platform, ui_type, group?}
    <data-dir>/images/...

Examples:
    python scripts/screenspot_pro_eval.py --download --data-dir data/ScreenSpot-Pro
    python scripts/screenspot_pro_eval.py --data-dir data/ScreenSpot-Pro --limit 20          # quick check
    python scripts/screenspot_pro_eval.py --data-dir data/ScreenSpot-Pro --group all         # full set
    python scripts/screenspot_pro_eval.py --data-dir data/ScreenSpot-Pro --max-pixels 3000000  # small context

Context length: a full-size screenshot becomes up to 16,384 image tokens (the default
--max-pixels). Start vLLM with --max-model-len of at least 18000 for that, or lower
--max-pixels (this lowers accuracy on tiny targets; record the value you used).

Needs: Pillow. `--download` also needs huggingface_hub. Everything else is the standard library.
Writes <out>/results.jsonl (one line per sample) and <out>/summary.json. Exit 1 on a fatal error.
"""
import argparse
import base64
import collections
import concurrent.futures as futures
import io
import json
import math
import os
import re
import sys
import time
import urllib.request

from PIL import Image

DATASET = "likaixin/ScreenSpot-Pro"
FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
DEFAULT_MAX_PIXELS = 16384 * 28 * 28

# Same text as COMPUTER_USE_PROMPT in dart_rollouter/src/core/prompts.py.
PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


# ----------------------------------------------------------- resize (matches the repo)
def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


# ------------------------------------------------------------------- output parsing
_NUM = r"-?\d+(?:\.\d+)?"


def parse_click(text):
    """Return (x, y) in the model's pixel space from the first click-like action, else None.

    Accepts click/left_double/right_single(start_box='<|box_start|>(x,y)<|box_end|>'), a
    4-number box (centre is used), and the <point>x y</point> form.
    """
    action = text.split("Action:")[-1]
    m = re.search(r"(?:click|left_double|right_single)\s*\(\s*(?:start_box|point)\s*=\s*['\"]?(.*?)['\"]?\s*\)\s*$",
                  action.strip().split("\n")[0], re.S)
    body = m.group(1) if m else action
    if not m and not re.search(r"(click|left_double|right_single)\s*\(", action):
        return None
    nums = [float(n) for n in re.findall(_NUM, body)]
    if len(nums) == 2:
        return nums[0], nums[1]
    if len(nums) >= 4:
        return (nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2
    return None


# ---------------------------------------------------------------------------- data
def download(data_dir):
    from huggingface_hub import snapshot_download  # imported lazily: only needed here

    print("downloading %s (about 3.4 GB) to %s" % (DATASET, data_dir), flush=True)
    snapshot_download(repo_id=DATASET, repo_type="dataset", local_dir=data_dir,
                      allow_patterns=["annotations/*", "images/*", "README.md"])


def load_samples(data_dir, group, app_filter):
    ann_dir = os.path.join(data_dir, "annotations")
    if not os.path.isdir(ann_dir):
        raise SystemExit("no annotations/ folder in %s (use --download, or check --data-dir)" % data_dir)
    samples = []
    for name in sorted(os.listdir(ann_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(ann_dir, name), encoding="utf-8") as f:
            for s in json.load(f):
                s["_task"] = name[:-5]
                samples.append(s)
    if group != "all":
        samples = [s for s in samples if _group_of(s) == group]
    if app_filter:
        samples = [s for s in samples if app_filter.lower() in str(s.get("application", "")).lower()]
    return samples


def _group_of(sample):
    """The `group` field is optional in the annotations; fall back to the file name."""
    g = sample.get("group")
    if g:
        return str(g).lower()
    cad_apps = ("autocad", "solidworks", "inventor", "quartus", "vivado", "fusion", "freecad", "revit")
    task = sample["_task"].lower()
    return "cad" if any(a in task for a in cad_apps) else "other"


# --------------------------------------------------------------------------- model
def encode_image(path, max_pixels):
    """Return (data_url, sent_height, sent_width, orig_height, orig_width)."""
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    sh, sw = smart_resize(oh, ow, max_pixels=max_pixels)
    if (sh, sw) != (oh, ow):
        img = img.resize((sw, sh), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return url, sh, sw, oh, ow


def ask(base_url, model, url, instruction, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(instruction=instruction)},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    reply = json.load(urllib.request.urlopen(req, timeout=timeout))
    return reply["choices"][0]["message"]["content"], reply.get("usage", {})


def evaluate_one(args, model, sample):
    path = os.path.join(args.data_dir, "images", sample["img_filename"])
    row = {"task": sample["_task"], "application": sample.get("application"), "ui_type": sample.get("ui_type"),
           "instruction": sample["instruction"], "bbox": sample["bbox"], "img": sample["img_filename"]}
    t0 = time.time()
    try:
        url, sh, sw, oh, ow = encode_image(path, args.max_pixels)
        text, usage = ask(args.model_url, model, url, sample["instruction"], args.max_tokens, args.timeout)
    except Exception as e:  # a failed request counts as an error, not a wrong answer
        row.update(result="error", error="%s: %s" % (type(e).__name__, e), seconds=round(time.time() - t0, 2))
        return row
    point = parse_click(text)
    row.update(output=text, seconds=round(time.time() - t0, 2), prompt_tokens=usage.get("prompt_tokens"),
               sent_size=[sw, sh], orig_size=[ow, oh])
    if point is None:
        row["result"] = "wrong_format"
        return row
    # model pixels (sent image) -> original pixels
    x, y = point[0] / sw * ow, point[1] / sh * oh
    x1, y1, x2, y2 = sample["bbox"]
    row.update(point=[round(x, 1), round(y, 1)], result="correct" if (x1 <= x <= x2 and y1 <= y <= y2) else "wrong")
    return row


# ------------------------------------------------------------------------- summary
def acc(rows):
    scored = [r for r in rows if r["result"] != "error"]
    n = len(scored)
    return {"n": n, "correct": sum(r["result"] == "correct" for r in scored),
            "accuracy": round(sum(r["result"] == "correct" for r in scored) / n, 4) if n else None,
            "wrong_format": sum(r["result"] == "wrong_format" for r in scored),
            "errors": len(rows) - n}


def summarize(rows):
    out = {"overall": acc(rows)}
    for key in ("application", "ui_type"):
        buckets = collections.defaultdict(list)
        for r in rows:
            buckets[str(r.get(key))].append(r)
        out["by_" + key] = {k: acc(v) for k, v in sorted(buckets.items())}
    tokens = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    secs = [r["seconds"] for r in rows if "seconds" in r]
    if tokens:
        out["prompt_tokens"] = {"mean": round(sum(tokens) / len(tokens)), "max": max(tokens)}
    if secs:
        out["seconds_per_sample_mean"] = round(sum(secs) / len(secs), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/ScreenSpot-Pro")
    ap.add_argument("--download", action="store_true", help="fetch the dataset from Hugging Face first (about 3.4 GB)")
    ap.add_argument("--model-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="model id (default: first listed by the server)")
    ap.add_argument("--group", default="cad", help="cad (default), dev, creative, scientific, office, os, or all")
    ap.add_argument("--app", default="", help="only applications whose name contains this text")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N samples (quick check)")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests; keep low on a 24 GB GPU")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                    help="downscale screenshots above this many pixels (default 16384*28*28, about 16k image tokens)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="screenspot_pro_out")
    args = ap.parse_args()

    if args.download:
        download(args.data_dir)
    samples = load_samples(args.data_dir, args.group.lower(), args.app)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise SystemExit("no samples matched --group %s --app '%s'" % (args.group, args.app))

    model = args.model
    if not model:
        listing = json.load(urllib.request.urlopen(args.model_url.rstrip("/") + "/models", timeout=30))
        model = listing["data"][0]["id"]
    os.makedirs(args.out, exist_ok=True)
    print("model %s | %d samples | group=%s | max_pixels=%d | workers=%d"
          % (model, len(samples), args.group, args.max_pixels, args.workers), flush=True)

    rows = []
    with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as sink, \
            futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(lambda s: evaluate_one(args, model, s), samples), 1):
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if i % 10 == 0 or i == len(samples):
                a = acc(rows)
                print("%d/%d  accuracy %s (%d correct, %d wrong_format, %d errors)"
                      % (i, len(samples), a["accuracy"], a["correct"], a["wrong_format"], a["errors"]), flush=True)

    summary = summarize(rows)
    summary.update(model=model, group=args.group, max_pixels=args.max_pixels, samples=len(samples),
                   date=time.strftime("%Y-%m-%d"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"overall": summary["overall"], "by_application": summary["by_application"]}, indent=2))
    if summary["overall"]["errors"] == len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Day-one smoke test for the DART-GUI + Archicad setup. Standard library only.

Stages (run one or `all`):
  host      GPU, driver, KVM, Docker, RAM, disk, Python packages
  model     vLLM server answers a screenshot prompt; latency and GPU memory
  env       environment server boots a VM; screenshot and command execution work
  archicad  Archicad answers its JSON API (inside the VM with --vm, or locally with --local)

Examples:
  python scripts/smoke_test.py host
  python scripts/smoke_test.py model --model-url http://localhost:8000/v1
  python scripts/smoke_test.py env --env-url http://localhost:50003 --token dart --archicad
  python scripts/smoke_test.py archicad --local
  python scripts/smoke_test.py all --env-url http://localhost:50003 --token dart

Each check prints PASS / WARN / FAIL / SKIP. A JSON report is written to --out.
Exit code is 1 if any check FAILed. Written before the hardware arrived: untested on it.
"""
import argparse
import base64
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from urllib.parse import urlparse

RESULTS = []
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(REPO, "dart_rollouter", "evaluation_examples", "examples_archicad", "vm_tools", "archicad_check.py")


def record(stage, name, status, detail=""):
    RESULTS.append({"stage": stage, "check": name, "status": status, "detail": detail})
    print("[%-4s] %-8s %-34s %s" % (status, stage, name, detail))


def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 127, str(e)


def http(method, url, body=None, headers=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def test_png(width=1280, height=720):
    """A plain gradient PNG, enough to exercise the image path without any dependency."""
    rows = b"".join(
        bytes([0]) + bytes(c for x in range(width) for c in (x * 255 // width, y * 255 // height, 128))
        for y in range(height)
    )

    def chunk(tag, payload):
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def gpu_memory_used_mib():
    code, out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


# ------------------------------------------------------------------- stages
def stage_host(args):
    s = "host"
    record(s, "os", "PASS" if platform.system() == "Linux" else "FAIL",
           "%s %s (the environment server needs Linux)" % (platform.system(), platform.release()))
    record(s, "python", "PASS" if sys.version_info >= (3, 10) else "WARN", platform.python_version())

    if shutil.which("nvidia-smi"):
        code, out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        record(s, "gpu", "PASS" if code == 0 and out.strip() else "FAIL", out.strip().replace("\n", " | ") or "no output")
        if code == 0:
            for line in out.strip().splitlines():
                try:
                    mib = int(line.split(",")[1].strip().split()[0])
                    record(s, "gpu memory", "PASS" if mib >= 22000 else "FAIL", "%d MiB (7B model in bf16 needs ~17 GB + context)" % mib)
                except Exception:
                    pass
    else:
        record(s, "gpu", "FAIL", "nvidia-smi not found")

    kvm = "/dev/kvm"
    record(s, "kvm", "PASS" if os.path.exists(kvm) and os.access(kvm, os.R_OK | os.W_OK) else "FAIL",
           "%s present and accessible" % kvm if os.path.exists(kvm) else "%s missing: enable virtualization in BIOS / load kvm modules" % kvm)

    code, out = run(["docker", "ps"])
    record(s, "docker", "PASS" if code == 0 else "FAIL", "daemon reachable" if code == 0 else out.strip()[:120])

    try:
        meminfo = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        gib = int(meminfo["MemTotal"].split()[0]) / 1024 / 1024
        record(s, "ram", "PASS" if gib >= 120 else "WARN", "%.0f GiB total (pilot: 128 GiB workstation, 256+ GiB env server)" % gib)
    except Exception:
        record(s, "ram", "SKIP", "could not read /proc/meminfo")

    free = shutil.disk_usage(args.disk_path).free / 1024 ** 3
    record(s, "disk", "PASS" if free >= 200 else "WARN", "%.0f GiB free at %s (images, models, trajectories)" % (free, args.disk_path))

    for mod in ("torch", "vllm"):
        code, out = run([sys.executable, "-c", "import %s;print(%s.__version__)" % (mod, mod)], timeout=120)
        record(s, "python:" + mod, "PASS" if code == 0 else "WARN", out.strip().splitlines()[-1] if out.strip() else "not installed")


def stage_model(args):
    s = "model"
    base = args.model_url.rstrip("/")
    try:
        _, body = http("GET", base + "/models", timeout=10)
        models = [m["id"] for m in json.loads(body)["data"]]
        record(s, "server up", "PASS", ", ".join(models))
    except Exception as e:
        record(s, "server up", "FAIL", "%s (start vLLM first; see docs/dart_local)" % e)
        return
    model = args.model or models[0]
    image = "data:image/png;base64," + base64.b64encode(test_png()).decode()

    def ask(n_images):
        content = [{"type": "image_url", "image_url": {"url": image}} for _ in range(n_images)]
        content.append({"type": "text", "text": "Describe the colours you see in one short sentence."})
        t = time.time()
        _, body = http("POST", base + "/chat/completions",
                       {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 48, "temperature": 0},
                       timeout=300)
        return time.time() - t, json.loads(body)

    before = gpu_memory_used_mib()
    for n in (1, 5):  # the rollouter keeps up to 5 recent screenshots in context
        try:
            seconds, reply = ask(n)
            used = reply.get("usage", {})
            record(s, "%d-image prompt" % n, "PASS",
                   "%.1fs, prompt %s tok, completion %s tok" % (seconds, used.get("prompt_tokens"), used.get("completion_tokens")))
        except urllib.error.HTTPError as e:
            record(s, "%d-image prompt" % n, "FAIL", "HTTP %s: %s" % (e.code, e.read()[:200]))
        except Exception as e:
            record(s, "%d-image prompt" % n, "FAIL", str(e))
    after = gpu_memory_used_mib()
    if after:
        record(s, "gpu memory in use", "PASS" if max(after) < 23500 else "WARN",
               "%s MiB (before %s); near 24 GB means lower --max-model-len" % (after, before))
    try:
        times = [ask(1)[0] for _ in range(3)]
        record(s, "latency (1 image)", "PASS", "median %.1fs of 3" % sorted(times)[1])
    except Exception as e:
        record(s, "latency (1 image)", "FAIL", str(e))


def _vm_exec(host, port, command, timeout=60):
    _, body = http("POST", "http://%s:%s/execute" % (host, port), {"command": command, "shell": False}, timeout=timeout)
    return json.loads(body)


def stage_env(args):
    s = "env"
    base = args.env_url.rstrip("/")
    host = urlparse(base).hostname
    headers = {"Authorization": "Bearer " + args.token} if args.token else {}
    try:
        _, body = http("GET", base + "/status", headers=headers, timeout=15)
        record(s, "server up", "PASS", body.decode()[:160])
    except Exception as e:
        record(s, "server up", "FAIL", str(e))
        return

    emulator = None
    t0 = time.time()
    try:
        _, body = http("POST", base + "/start_emulator", {"token": args.token} if args.token else {}, headers, timeout=900)
        reply = json.loads(body)
        if reply.get("code") != 0:
            record(s, "start vm", "FAIL", json.dumps(reply)[:200])
            return
        emulator = reply["data"]
        record(s, "start vm", "PASS", "emulator %s, %.0fs" % (emulator["emulator_id"], time.time() - t0))
        port = emulator["server_port"]

        deadline = time.time() + args.boot_timeout
        shot = None
        while time.time() < deadline:
            try:
                status, shot = http("GET", "http://%s:%s/screenshot" % (host, port), timeout=15)
                if status == 200 and shot[:4] == b"\x89PNG":
                    break
            except Exception:
                pass
            shot = None
            time.sleep(5)
        if shot:
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, "vm_screenshot.png")
            open(path, "wb").write(shot)
            w, h = struct.unpack(">II", shot[16:24])
            record(s, "screenshot", "PASS", "%dx%d in %.0fs after start, saved %s" % (w, h, time.time() - t0, path))
        else:
            record(s, "screenshot", "FAIL", "no screenshot within %ds (is the guest helper running in the image?)" % args.boot_timeout)
            return

        try:
            out = _vm_exec(host, port, ["python", "--version"])
            text = (out.get("output") or out.get("error") or "").strip()
            record(s, "execute in vm", "PASS" if out.get("status") != "error" and text else "WARN", text or json.dumps(out)[:120])
        except Exception as e:
            record(s, "execute in vm", "FAIL", str(e))

        if args.archicad:
            try:
                out = _vm_exec(host, port, ["python", args.vm_checker, "probe"], timeout=120)
                record(s, "archicad in vm", "PASS" if '"ok": true' in (out.get("output") or "") else "FAIL",
                       (out.get("output") or out.get("error") or "")[:160].strip())
            except Exception as e:
                record(s, "archicad in vm", "FAIL", str(e))
    except Exception as e:
        record(s, "start vm", "FAIL", str(e))
    finally:
        if emulator:
            try:
                http("DELETE", "%s/emulators/%s" % (base, emulator["emulator_id"]), headers=headers, timeout=60)
                record(s, "stop vm", "PASS", "")
            except Exception as e:
                record(s, "stop vm", "WARN", "could not stop %s: %s" % (emulator["emulator_id"], e))


def stage_archicad(args):
    s = "archicad"
    if not args.local:
        record(s, "local probe", "SKIP", "use --local on a machine running Archicad, or `env --archicad` for the VM")
        return
    code, out = run([sys.executable, CHECKER, "probe"], timeout=60)
    ok = code == 0 and '"ok": true' in out
    record(s, "api reachable", "PASS" if ok else "FAIL", out.strip()[:200])
    if ok:
        snap_dir = os.path.join(args.out, "archicad_state")
        env = dict(os.environ, ARCHICAD_PILOT_DIR=snap_dir)
        t = time.time()
        p = subprocess.run([sys.executable, CHECKER, "snapshot"], capture_output=True, text=True, env=env, timeout=600)
        record(s, "snapshot (read-only)", "PASS" if '"ok": true' in p.stdout else "FAIL",
               "%.1fs: %s" % (time.time() - t, (p.stdout or p.stderr).strip()[:160]))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["host", "model", "env", "archicad", "all"])
    ap.add_argument("--model-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="model id (default: first listed by the server)")
    ap.add_argument("--env-url", default="http://localhost:50003")
    ap.add_argument("--token", default=os.environ.get("OSWORLD_TOKEN", ""))
    ap.add_argument("--archicad", action="store_true", help="env stage: also probe Archicad inside the VM")
    ap.add_argument("--vm-checker", default=r"C:\pilot\archicad_check.py")
    ap.add_argument("--local", action="store_true", help="archicad stage: probe an Archicad on this machine")
    ap.add_argument("--boot-timeout", type=int, default=600)
    ap.add_argument("--disk-path", default=".")
    ap.add_argument("--out", default="smoke_test_out")
    args = ap.parse_args()

    stages = {"host": stage_host, "model": stage_model, "env": stage_env, "archicad": stage_archicad}
    for name in (stages if args.stage == "all" else [args.stage]):
        stages[name](args)

    os.makedirs(args.out, exist_ok=True)
    report = os.path.join(args.out, "report.json")
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": RESULTS}, open(report, "w"), indent=2)
    counts = {k: sum(r["status"] == k for r in RESULTS) for k in ("PASS", "WARN", "FAIL", "SKIP")}
    print("\n%(PASS)d passed, %(WARN)d warnings, %(FAIL)d failed, %(SKIP)d skipped. Report: " % counts + report)
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()

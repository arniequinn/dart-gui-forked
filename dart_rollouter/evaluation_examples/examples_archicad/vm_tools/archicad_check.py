"""Archicad state checker. Runs INSIDE the Windows VM (Python 3.8+, standard library only).

Talks to Archicad's official JSON API on localhost (ports 19723-19743), so it needs
no add-on. Verified read-only against Archicad 26 build 3001 on 2026-10-08:
IsAlive, GetProductInfo, GetAllElements, GetElementsByType, GetTypesOfElements,
GetAttributesByType, GetLayerAttributes, GetPropertyIds, GetPropertyValuesOfElements.
The `check` logic itself has NOT been run end to end against the tasks yet.

Usage (state dir defaults to C:\\pilot, override with ARCHICAD_PILOT_DIR):
    python archicad_check.py probe
    python archicad_check.py snapshot          # once, at image build: writes baseline.json
    python archicad_check.py mark              # at episode start: records the start time
    python archicad_check.py check <task> [--arg key=value ...]

`check` prints one JSON line: {"passed": bool, "task": ..., "details": {...}}
"""
import json
import os
import re
import sys
import time
import urllib.request

PORTS = range(19723, 19744)
STATE_DIR = os.environ.get("ARCHICAD_PILOT_DIR", r"C:\pilot" if os.name == "nt" else os.path.expanduser("~/pilot"))
BASELINE = os.path.join(STATE_DIR, "baseline.json")
MARK = os.path.join(STATE_DIR, "mark.json")
CHUNK = 500
Z_TOL = 0.01  # metres


# ---------------------------------------------------------------- API client
def find_port():
    for port in PORTS:
        try:
            if call("API.IsAlive", port=port, timeout=2).get("result", {}).get("isAlive"):
                return port
        except Exception:
            continue
    raise RuntimeError("no running Archicad found on ports 19723-19743")


def call(command, params=None, port=None, timeout=60):
    body = {"command": command}
    if params is not None:
        body["parameters"] = params
    req = urllib.request.Request(
        "http://127.0.0.1:%d" % (port or find_port()),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    text = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    # Archicad sometimes emits `"value":}` for a non-finite number; read it as null.
    reply = json.loads(re.sub(r'"value":\s*(?=[}\]])', '"value":null', text))
    if not reply.get("succeeded"):
        raise RuntimeError("%s failed: %s" % (command, reply.get("error")))
    return reply


def _chunks(items):
    for i in range(0, len(items), CHUNK):
        yield items[i:i + CHUNK]


def _prop_ids(names):
    reply = call("API.GetPropertyIds", {"properties": [{"type": "BuiltIn", "nonLocalizedName": n} for n in names]})
    return [p["propertyId"] for p in reply["result"]["properties"]]


def _value(pv):
    pv = pv.get("propertyValue", {})
    return pv.get("value") if pv.get("status") == "normal" else None


# ------------------------------------------------------------------ snapshot
def take_snapshot():
    """elements: guid -> type, layer, z (bottom elevation to project zero, metres), id."""
    elements = call("API.GetAllElements")["result"]["elements"]
    ids = _prop_ids(["ModelView_LayerName", "General_BottomElevationToProjectZero", "General_ElementID"])
    table = {}
    for chunk in _chunks(elements):
        types = call("API.GetTypesOfElements", {"elements": chunk})["result"]["typesOfElements"]
        values = call(
            "API.GetPropertyValuesOfElements",
            {"elements": chunk, "properties": [{"propertyId": i} for i in ids]},
        )["result"]["propertyValuesForElements"]
        for t, v in zip(types, values):
            guid = t["typeOfElement"]["elementId"]["guid"]
            layer, z, eid = [_value(x) for x in v["propertyValues"]]
            table[guid] = {"t": t["typeOfElement"]["elementType"], "layer": layer, "z": z, "id": eid}

    attr_ids = call("API.GetAttributesByType", {"attributeType": "Layer"})["result"]["attributeIds"]
    layers = {}
    for chunk in _chunks(attr_ids):
        for a in call("API.GetLayerAttributes", {"attributeIds": chunk})["result"]["attributes"]:
            la = a["layerAttribute"]
            layers[la["name"]] = {"hidden": la["isHidden"], "locked": la["isLocked"]}
    version = call("API.GetProductInfo")["result"]["version"]
    return {"archicad": version, "taken": time.time(), "elements": table, "layers": layers}


def load_baseline():
    with open(BASELINE) as f:
        return json.load(f)


def diff(base, now):
    """What changed on elements that existed in the baseline, plus what is new."""
    b, n = base["elements"], now["elements"]
    removed = [g for g in b if g not in n]
    added = [g for g in n if g not in b]
    layer_changed = [g for g in b if g in n and b[g]["layer"] != n[g]["layer"]]
    moved = [g for g in b if g in n and not _same_z(b[g]["z"], n[g]["z"])]
    return {"removed": removed, "added": added, "layer_changed": layer_changed, "moved": moved}


def _same_z(a, b):
    return a == b or (a is not None and b is not None and abs(a - b) <= Z_TOL)


def _short(items, n=5):
    return {"count": len(items), "sample": items[:n]}


# -------------------------------------------------------------------- checks
def check_layer_create(base, now, name):
    d = diff(base, now)
    ok = (
        name in now["layers"]
        and name not in base["layers"]
        and len(now["layers"]) == len(base["layers"]) + 1
        and not d["removed"] and not d["added"] and not d["layer_changed"]
    )
    return ok, {"layer_present": name in now["layers"], "layers_before": len(base["layers"]),
                "layers_after": len(now["layers"]), "unexpected_element_changes": _short(d["removed"] + d["added"] + d["layer_changed"])}


def check_layer_assign(base, now, layer, types):
    types = set(types.split(","))
    d = diff(base, now)
    targets = [g for g, e in base["elements"].items() if e["t"] in types]
    wrong = [g for g in targets if g in now["elements"] and now["elements"][g]["layer"] != layer]
    collateral = [g for g in d["layer_changed"] if base["elements"][g]["t"] not in types]
    ok = bool(targets) and not wrong and not collateral and not d["removed"] and not d["added"]
    return ok, {"target_count": len(targets), "not_on_layer": _short(wrong), "collateral_layer_changes": _short(collateral),
                "removed": len(d["removed"]), "added": len(d["added"])}


def check_layer_flags(base, now, hide, lock):
    hide = [x for x in hide.split(",") if x]
    lock = [x for x in lock.split(",") if x]
    problems = []
    for name, flag, key in [(n, True, "hidden") for n in hide] + [(n, True, "locked") for n in lock]:
        if now["layers"].get(name, {}).get(key) is not flag:
            problems.append("%s should be %s" % (name, key))
    for name, state in base["layers"].items():
        if name in hide or name in lock or name not in now["layers"]:
            continue
        if now["layers"][name] != state:
            problems.append("%s changed unexpectedly" % name)
    d = diff(base, now)
    ok = not problems and not d["removed"] and not d["added"]
    return ok, {"problems": problems, "removed": len(d["removed"]), "added": len(d["added"])}


def check_file_saved(path, min_bytes="1000"):
    started = json.load(open(MARK))["time"] if os.path.exists(MARK) else 0
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    fresh = exists and os.path.getmtime(path) >= started
    return exists and fresh and size >= int(min_bytes), {"exists": exists, "bytes": size, "saved_after_start": fresh}


def check_copy_to_story(base, now, source_z="", target_z="", types=""):
    if source_z in ("", "null", None) or target_z in ("", "null", None):
        return False, {"error": "not configured: set source_z and target_z (metres) for the chosen pilot project"}
    sz, tz = float(source_z), float(target_z)
    wanted = set(t for t in types.split(",") if t)
    source = [g for g, e in base["elements"].items()
              if e["z"] is not None and abs(e["z"] - sz) <= Z_TOL and (not wanted or e["t"] in wanted)]
    d = diff(base, now)
    new = [now["elements"][g] for g in d["added"]]
    want_types = sorted(base["elements"][g]["t"] for g in source)
    got_types = sorted(e["t"] for e in new)
    at_target = all(e["z"] is not None and abs(e["z"] - tz) <= Z_TOL for e in new)
    ok = bool(source) and want_types == got_types and at_target and not d["removed"] and not d["moved"] and not d["layer_changed"]
    return ok, {"source_count": len(source), "new_count": len(new), "types_match": want_types == got_types,
                "all_new_at_target": at_target, "removed": len(d["removed"]), "moved": len(d["moved"])}


# ----------------------------------------------------------------------- CLI
def _args(argv):
    out = {}
    i = 0
    while i < len(argv):
        if argv[i] == "--arg" and i + 1 < len(argv):
            k, _, v = argv[i + 1].partition("=")
            out[k] = v
            i += 2
        else:
            i += 1
    return out


TASKS = {
    "layer_create": (check_layer_create, ["name"]),
    "layer_assign": (check_layer_assign, ["layer", "types"]),
    "layer_flags": (check_layer_flags, ["hide", "lock"]),
    "copy_to_story": (check_copy_to_story, ["source_z", "target_z"]),
}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "probe"
    os.makedirs(STATE_DIR, exist_ok=True)
    if cmd == "probe":
        port = find_port()
        info = call("API.GetProductInfo", port=port)["result"]
        count = len(call("API.GetAllElements", port=port)["result"]["elements"])
        print(json.dumps({"ok": True, "port": port, "archicad": info, "elements": count,
                          "baseline_present": os.path.exists(BASELINE)}))
    elif cmd == "snapshot":
        snap = take_snapshot()
        with open(BASELINE, "w") as f:
            json.dump(snap, f)
        print(json.dumps({"ok": True, "elements": len(snap["elements"]), "layers": len(snap["layers"]), "path": BASELINE}))
    elif cmd == "mark":
        with open(MARK, "w") as f:
            json.dump({"time": time.time()}, f)
        print(json.dumps({"ok": True}))
    elif cmd == "check":
        task = argv[2]
        kw = _args(argv[3:])
        try:
            if task == "file_saved":
                passed, details = check_file_saved(**kw)
            elif task in TASKS:
                fn, _ = TASKS[task]
                passed, details = fn(load_baseline(), take_snapshot(), **kw)
            else:
                raise ValueError("unknown check '%s'" % task)
            print(json.dumps({"passed": bool(passed), "task": task, "details": details}))
        except Exception as e:  # report, never crash the evaluator
            print(json.dumps({"passed": False, "task": task, "error": str(e)}))
    else:
        print(json.dumps({"error": "unknown command '%s'" % cmd}))


if __name__ == "__main__":
    main(sys.argv)

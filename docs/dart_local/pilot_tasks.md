# Archicad pilot: task definitions and smoke test

## What is tested vs not

| Part | Status |
| --- | --- |
| `archicad_check.py`: snapshot and the "nothing changed" check | Run read-only against a live Archicad 26 (build 3001, 1,600 elements): snapshot in ~2.6 s, results as expected |
| `archicad_check.py`: the four task checks with real edits | NOT run. Only their failing paths were seen (nothing changed yet) |
| Evaluator getter/metric (`archicad_check`, `archicad_check_passed`) | Syntax-checked only; not run through the environment |
| Task JSON setup steps (launch Archicad, sleep, mark) | NOT run; Archicad install path and the 120 s wait are assumptions |
| `scripts/smoke_test.py` host / model / archicad stages | Run here (model stage against a fake server) |
| `scripts/smoke_test.py` env stage (VM boot, screenshot, execute) | NOT run: needs the environment server |

## Pilot tasks

All five live in `dart_rollouter/evaluation_examples/examples_archicad/archicad/`. The task list for the rollouter is `evaluation_examples/archicad_pilot.json` (tasks 1-4). Point `task.task_file` and `task.osworld_root` in the rollouter config at them.

| # | Task | Tier | Checked by |
| --- | --- | --- | --- |
| 1 | Create layer `PILOT-NEW` | 1 | Layer exists, exactly one layer added, no element changed |
| 2 | Move every column to layer `S-COL-TYPICAL` | 2 | All columns on that layer, no other element's layer changed, none added or removed |
| 3 | Hide `PILOT-HIDE`, lock `PILOT-LOCK` | 1 | Both flags set, no other layer's flags changed |
| 4 | Save As `C:\pilot\out\pilot_saved.pln` | 1 | File exists, written after episode start, over 1,000 bytes |
| 5 | Copy lowest story's elements to the story above | 3 | New elements match the source by type, sit at the target elevation, nothing else changed. DISABLED until `source_z` / `target_z` are set |

The checks use only Archicad's official JSON API (layer attributes, element types, and the built-in properties `ModelView_LayerName`, `General_BottomElevationToProjectZero`, `General_ElementID`). The Tapir add-on is not needed. It is not loaded in the Archicad instance tested here.

## Pilot project requirements

`C:\pilot\pilot_base.pln` in the Windows image must have:
- columns (task 2) and at least two stories with elements on them (task 5);
- layers `S-COL-TYPICAL`, `PILOT-HIDE`, `PILOT-LOCK` already created, with the last two visible and unlocked;
- no layer named `PILOT-NEW`.

A small excerpt of a real multistorey project works well. Keep it small: the checker reads every element on each check.

## Image build steps for the pilot

1. Python 3 in the VM; the OSWorld guest helper (port 5000) running at boot.
2. Copy `vm_tools/archicad_check.py` to `C:\pilot\`.
3. Open `pilot_base.pln` in Archicad, then run `python C:\pilot\archicad_check.py snapshot` once. This writes `C:\pilot\baseline.json`, the "before" state every check compares against.
4. Confirm the Archicad path used by the task setup (`C:\Program Files\GRAPHISOFT\Archicad 26\Archicad 26.exe`) and tune the 120 s sleep.
5. Shut down and keep the image read-only. Each episode then starts from the same clean state.

## Smoke test on arrival

```bash
python scripts/smoke_test.py host
python scripts/smoke_test.py model --model-url http://localhost:8000/v1
python scripts/smoke_test.py env --env-url http://localhost:50003 --token dart --archicad
python scripts/smoke_test.py all --env-url http://localhost:50003 --token dart
```

Start vLLM first for the `model` stage, starting with a short context (see risk 5 in `preflight_risk_register.md`). The host stage is meant to FAIL on a non-Linux or GPU-less machine.

## Measuring the pilot

For each task, run the untouched model at least 20 times and record success rate, steps used and time. Agree the pass targets with the design lead before starting rather than after seeing results. Review every output by hand during the pilot.

## Zero-shot grounding check (ScreenSpot-Pro)

Run this first on the new box, before the Archicad pilot. It scores UI-TARS on professional high-resolution software, including a CAD group, and tells us whether the model can find small buttons at all.

```bash
pip install pillow huggingface_hub
python scripts/screenspot_pro_eval.py --download --data-dir data/ScreenSpot-Pro   # about 3.4 GB, MIT licence
python scripts/screenspot_pro_eval.py --data-dir data/ScreenSpot-Pro --limit 20   # quick check
python scripts/screenspot_pro_eval.py --data-dir data/ScreenSpot-Pro              # CAD group
```

- Uses the rollouter's own prompt and coordinate convention (smart_resize pixels mapped back to the original).
- A full-size screenshot is up to 16,384 image tokens. Start vLLM with `--max-model-len` of 18000 or more, or lower `--max-pixels` (less accuracy on tiny targets; record the value).
- Tested only against a fake server so far: prompt, downscale, coordinate mapping, scoring and summaries. NOT tested: the real dataset's field names beyond what its published eval script reads, the `group` value used for `--group cad`, or the real model's output format.
- Writes `screenspot_pro_out/results.jsonl` and `summary.json`.

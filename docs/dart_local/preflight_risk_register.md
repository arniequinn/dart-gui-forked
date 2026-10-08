# DART-GUI + Archicad: pre-flight risk register

Status: research only, nothing run yet. Based on reading the repo (fork + submodules) and web research on 2026-10-08.
"Verified" = read in code or a source. "Unverified" = my estimate; test when hardware arrives.

## Target architecture

- Linux GPU workstation (RTX 4090 24 GB): serves UI-TARS-1.5-7B with vLLM.
- Linux environment server (KVM): runs Windows 10 VMs in Docker/QEMU, each with Archicad 26 + the OSWorld guest helper.
- Archicad JSON/Tapir API reads model state back, used as the pass/fail check.
- Training (later): rented 8x80 GB node. Its VMs must sit next to it.

## Risks, ranked

| # | Risk | Evidence | Mitigation | Test on arrival |
|---|------|----------|-----------|-----------------|
| 1 | Archicad 26 may not run well, or at all, in a VM without a GPU | Graphisoft lists DirectX 11 GPUs at every tier ([requirements](https://graphisoft.com/resources-and-support/system-requirements-26)); no source mentions VM support. Unverified. | Ask Graphisoft support whether VMs are supported. Try software rendering first. Fallback: pass the 4090 through to ONE Windows VM (needs a second GPU for the host, or run the host headless). | Install Archicad in a test Windows VM; open a project; open 2D plan and 3D view; note crashes and frame rate. |
| 2 | Env code only ships an Ubuntu image path | `docker/provider.py` takes `os_type` but I found no Windows-specific handling; default `RAM_SIZE=2G`, `CPU_CORES=2` is far too small for Archicad. Verified. | Build a Windows qcow2, raise RAM/CPU/disk in the provider, install the OSWorld guest server in Windows, snapshot a clean state. Budget days to weeks. | `env_test.py` against the Windows VM: screenshot, click, type. |
| 3 | Reset between attempts | RL needs a clean state each episode; Archicad projects are large files. | Revert to snapshot or copy the project file fresh each episode. Measure reset time. | Time a full reset cycle. |
| 4 | Capacity of the environment server | Archicad VM likely 12-16 GB RAM each (unverified). | Plan 12-15 concurrent VMs on 256 GB; size RAM after measuring one VM. | Run 1, 4, 8 VMs; record RAM/CPU per VM. |
| 5 | 24 GB GPU is tight for inference | Weights are 16.58 GB in BF16 ([model listing](https://llmrun.dev/model/bytedance-seed-ui-tars-1-5-7b)); I found no tested 4090 config. Unverified. | Start vLLM with `--max-model-len` 8k-16k, `--limit-mm-per-prompt` set low, `--gpu-memory-utilization` ~0.9; set `replicas: 1`. Quantised build as fallback. | Serve the model, send 5-screenshot prompts, watch for OOM. |
| 6 | Zero-shot accuracy on Archicad is unknown | No Archicad data exists. | Define 2-3 narrow tasks with API-checkable results; keep a human reviewing all output; stop at the month-2 gate if targets fail. | Run each pilot task 20+ times; record success rate. |
| 7 | Rewards need automatic checkers | Repo evaluators target OSWorld Linux apps only. Verified. | Write checkers that query the Archicad API for expected elements/properties. | One checker per pilot task, validated against hand-made correct and wrong models. |
| 8 | Training needs far more GPU than the workstation | Trainer script: full FSDP, 32k prompt, no offload. Verified. | Rent 8x80 GB for training only; do not attempt on the 4090. | n/a until Phase 2. |
| 9 | Rented GPUs and Lahore VMs are far apart | Trainer, rollouter and VMs exchange screenshots constantly. | Run VMs at the same provider as the GPUs (CPU instances), or keep training small. Check licence terms for cloud VMs. | Measure screenshot size per step and round-trip latency before committing to a rental. |
| 10 | Repo is not runnable as cloned | Submodules were empty (initialised locally); `MODEL_PATH` is a placeholder; Docker image comes from an Alibaba Beijing registry. Verified. | Pin submodule commits in our fork; mirror the Docker image; fill in paths. | Pull image and model on the new box; check download size/time on our connection. |
| 11 | Leaked credentials in config | `rollouter/config/config.yaml` has a third-party server URL and token. Verified. | Do not use them. Remove from our fork and git history; use our own env server and token. | Grep the fork for tokens before first run. |
| 12 | Licences | Archicad and Windows copies in VMs need valid licences, and concurrent copies need terms that allow it. Unverified. | Confirm with Graphisoft and Microsoft before building images; get written answers. | n/a |
| 13 | Power and connectivity in Lahore | Grid outages; industrial tariff about Rs 26/unit ([source](https://www.thenews.pk/print/1399652-pakistan-cuts-industrial-power-tariffs-by-up-to-rs5-per-unit)). | UPS sized for the workstation; env server on UPS too; save checkpoints often. | Pull the plug test on UPS runtime. |

## Order of work when the hardware arrives

1. Install Ubuntu, NVIDIA driver, Docker, check `/dev/kvm`.
2. Smoke test: serve UI-TARS on the 4090; run the Ubuntu environment from the repo (risks 5, 10).
3. Build the Windows + Archicad image (risks 1, 2, 3, 4).
4. Write checkers and run zero-shot pilots (risks 6, 7).
5. Only then decide on rented training (risks 8, 9).

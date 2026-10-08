# Windows + Archicad environment (plan)

Goal: let the agent drive Archicad 26 inside a Windows VM, using the same Docker/QEMU environment server as the Ubuntu tasks.

Status: configuration hooks are in place; the Windows image has NOT been built or tested.

## Steps

1. Build a Windows qcow2 on the Linux environment host (KVM required), install Archicad 26, and install the OSWorld guest helper (Flask server on port 5000) so the agent can take screenshots and send input.
2. Set the image and sizing in `GUI-Docker-Env/configs/config.yaml`:
   `path_to_vm`, `vm_ram_size` (start at 16G), `vm_cpu_cores`, `vm_disk_size`.
   Extra QEMU container variables for Windows (boot mode, etc.) go in `vm_extra_env`; the right values are unverified.
3. Run `python env_test.py` against the new image: screenshot, click, type.
4. Measure RAM/CPU per VM before deciding concurrency.
5. Reward checks: query the Archicad API (Tapir) after each episode for the expected elements and properties.

## Open questions (test on the real hardware)

- Does the base image boot an arbitrary Windows qcow2 unchanged, or does it need Windows-specific boot settings?
- Archicad 3D view without a GPU: frame rate and stability.
- Episode reset: snapshot revert vs fresh project copy; time per reset.

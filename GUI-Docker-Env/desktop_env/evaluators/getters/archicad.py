import json
import logging
from typing import Any, Dict

import requests

logger = logging.getLogger("desktopenv.getters.archicad")

DEFAULT_SCRIPT = r"C:\pilot\archicad_check.py"


def get_archicad_check(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Archicad state checker inside the VM and return its JSON verdict.

    Config:
        check (str): checker name, e.g. "layer_assign"
        args (dict): optional key/value arguments passed as --arg key=value
        script (str): optional path of archicad_check.py inside the VM
        python (str): optional interpreter command inside the VM (default "python")
    """
    command = [config.get("python", "python"), config.get("script", DEFAULT_SCRIPT), "check", config["check"]]
    for key, value in (config.get("args") or {}).items():
        command += ["--arg", "%s=%s" % (key, value)]
    try:
        response = requests.post(
            "http://%s:%s/execute" % (env.vm_ip, env.server_port),
            json={"command": command, "shell": False},
            timeout=120,
        )
        output = response.json().get("output", "") if response.status_code == 200 else ""
        # the checker prints a single JSON line; ignore any other output
        for line in reversed(output.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        logger.error("archicad checker gave no JSON (status %s): %r", response.status_code, output[:300])
    except Exception as e:
        logger.error("archicad checker failed: %s", e)
    return {"passed": False, "error": "checker did not return a result"}

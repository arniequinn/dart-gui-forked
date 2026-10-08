from typing import Any, Dict


def archicad_check_passed(result: Dict[str, Any], *args, **options) -> float:
    """1.0 when the in-VM Archicad checker reported passed, else 0.0."""
    return 1.0 if isinstance(result, dict) and result.get("passed") is True else 0.0

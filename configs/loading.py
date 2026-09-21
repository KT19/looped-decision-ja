from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml_sections(path: str | Path) -> dict[str, Any]:
    """Flatten a feature-grouped YAML file into dataclass keyword arguments."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        sections = yaml.safe_load(file) or {}

    if not isinstance(sections, Mapping):
        raise TypeError("Configuration root must be a mapping.")

    values: dict[str, Any] = {}
    for section_name, section_values in sections.items():
        if not isinstance(section_values, Mapping):
            raise TypeError(f"Configuration section '{section_name}' must be a mapping.")
        for key, value in section_values.items():
            if key in values:
                raise ValueError(f"Duplicate configuration key: {key}")
            values[str(key)] = value

    return values

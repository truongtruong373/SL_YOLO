from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Đọc YAML và trả về config cùng thư mục gốc của file."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Nội dung config.yaml phải là một mapping YAML.")

    return config, path.parent


def get_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(
            f"config.yaml phải chứa section '{section_name}'."
        )
    return section


def resolve_config_path(
    value: Any,
    config_dir: Path,
    field_name: str,
    allow_none: bool = False,
) -> Path | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Config '{field_name}' không được để trống.")
    if not isinstance(value, (str, Path)):
        raise TypeError(f"Config '{field_name}' phải là một đường dẫn.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()

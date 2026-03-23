# Import Libraries
import yaml
from pathlib import Path

def load_config(
    path: str | None = None
) -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    path: str or None, optional
        Path to the YAML configuration file. If None, loads `config.yaml`
        from the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    if path is None:
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)
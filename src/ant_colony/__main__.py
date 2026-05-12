"""
Entry point: python -m ant_colony [options]

CLI for running ant colony simulations in pygame (interactive) or
headless (batch) mode.
"""

import argparse
import os
import sys

from ant_colony import __version__
from ant_colony.simulation import AntColonySimulation


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict so both flat and dotted keys are available."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        items[k] = v  # keep the bare key too
    return items


def _load_config(path: str | None) -> dict:
    """Load config from a YAML file, returning a flattened dict."""
    # Default stock config
    cfg = {
        "width": 1200,
        "height": 800,
        "title": "Ant Colony Simulation",
        "fps": 60,
        "num_steps": 3600,
        "output_path": "output.mp4",
        "output_fps": 30,
        "scale": 1.0,
        "n_colonies": 2,
        "num_agents": 50,
        "obstacle_count": 12,
        "initial_food": 10,
        "evap_rate": 0.002,
        "pheromone_cell_size": 4.0,
        "log_interval": 50,
    }

    if path is not None:
        try:
            import yaml
            with open(path) as f:
                user_cfg = yaml.safe_load(f) or {}
                # Flatten nested keys so both 'width' and 'world.width' work
                flat = _flatten_dict(user_cfg)
                cfg.update(flat)
        except ImportError:
            print(f"[config] PyYAML not available, trying simple parse of {path}",
                  file=sys.stderr)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    try:
                        cfg[key] = float(val) if "." in val else int(val)
                    except (ValueError, TypeError):
                        cfg[key] = val
        except FileNotFoundError:
            print(f"[config] config file not found: {path}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"[config] error loading {path}: {exc}", file=sys.stderr)
            sys.exit(1)

    return cfg


def _resolve_config_path(path: str | None) -> str | None:
    """Resolve config path relative to the package, a custom path, or
    the default location."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(os.path.dirname(pkg_dir))  # src/
    project_dir = os.path.dirname(repo_dir)  # ant-colony-sim/

    if path:
        # Try as-is
        if os.path.exists(path):
            return path
        # Try relative to cwd
        cwd_path = os.path.join(os.getcwd(), path)
        if os.path.exists(cwd_path):
            return cwd_path
        # Try relative to project root
        proj_path = os.path.join(project_dir, path)
        if os.path.exists(proj_path):
            return proj_path
        # Try relative to package dir (e.g. configs/cfg.yaml)
        pkg_path = os.path.join(pkg_dir, path)
        if os.path.exists(pkg_path):
            return pkg_path
        print(f"[config] config not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Default: check several common locations
    candidates = [
        os.path.join(project_dir, "configs", "default.yaml"),
        os.path.join(os.getcwd(), "configs", "default.yaml"),
        os.path.join(pkg_dir, "config", "config.yaml"),
        os.path.join(project_dir, "config", "config.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ant_colony",
        description="Ant colony simulation with pheromone trails and emergent behaviour",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["pygame", "headless"],
        default=None,
        help="Run mode (default: pygame when display is available)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for recording (only in headless mode)",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=None,
        help="Number of steps for headless mode (default: from config, 3600)",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"ant-colony-sim v{__version__}",
    )
    return parser


def _detect_mode() -> str:
    """Auto-detect available mode: pygame if display is available, else headless."""
    try:
        import pygame
        import os as _os
        disp = _os.environ.get("DISPLAY", "")
        if disp:
            return "pygame"
    except ImportError:
        pass
    return "headless"


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve mode
    mode = args.mode
    if mode is None:
        mode = _detect_mode()

    # Load config
    config_path = _resolve_config_path(args.config)
    config = _load_config(config_path)

    # CLI overrides
    if args.output is not None:
        config["output_path"] = args.output
    if args.steps is not None:
        config["num_steps"] = args.steps

    # Construct and run simulation
    sim = AntColonySimulation(config)

    if mode == "pygame":
        sim.run_pygame()
    else:
        sim.run_headless(steps=config.get("num_steps"))

    sys.exit(0)


if __name__ == "__main__":
    main()

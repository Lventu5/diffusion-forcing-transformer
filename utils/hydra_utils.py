from typing import Optional
import json
import os
from omegaconf import OmegaConf


def _value_to_cli(value) -> str:
    """Serialize a Python value into a Hydra CLI override value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _append_cli_overrides(cli: list[str], key: str, value) -> None:
    """Flatten nested dictionaries into dotted Hydra overrides."""
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _append_cli_overrides(cli, f"{key}.{child_key}", child_value)
        return
    cli.append(f"++{key}={_value_to_cli(value)}")


def _yaml_to_cli(
    yaml_path: str,
    prefix: Optional[str] = None,
) -> list[str]:
    """
    Convert a yaml file to a list of command line arguments.
    """
    cfg = OmegaConf.load(yaml_path)
    cli = []
    for key, value in OmegaConf.to_container(cfg).items():
        full_key = f"{prefix}.{key}" if prefix else key
        _append_cli_overrides(cli, full_key, value)
    return cli


def unwrap_shortcuts(
    argv: list[str],
    config_path: str,
    config_name: str,
) -> list[str]:
    """
    Unwrap shortcuts by replacing them with commands from corresponding yaml files.
    All shortcuts should be in the form of `@shortcut_name`.
    Example:
    - @latent -> unwrap configurations/shortcut/latent/base.yaml and configurations/shortcut/latent/dataset_name.yaml
    - @mit_vision/h100 -> unwrap configurations/shortcut/mit_vision/h100.yaml
    """
    # find the default dataset
    defaults = OmegaConf.load(f"{config_path}/{config_name}.yaml").defaults
    dataset = next(default.dataset for default in defaults if "dataset" in default)
    # check if dataset is overridden
    for arg in argv:
        if arg.startswith("dataset="):
            dataset = arg.split("=")[1]

    if dataset is None:
        raise ValueError("Dataset name is not provided.")

    new_argv = []
    for arg in argv:
        if arg.startswith("@"):
            shortcut = arg[1:]
            base_path = f"{config_path}/shortcut/{shortcut}/base.yaml"

            if os.path.exists(base_path):
                new_argv += _yaml_to_cli(base_path)
                dataset_path = f"{config_path}/shortcut/{shortcut}/{dataset}.yaml"
                if os.path.exists(dataset_path):
                    new_argv += _yaml_to_cli(dataset_path)
            else:
                default_path = f"{config_path}/shortcut/{shortcut}.yaml"
                if os.path.exists(default_path):
                    new_argv += _yaml_to_cli(default_path)
                else:
                    raise ValueError(f"Shortcut @{shortcut} not found.")
        elif arg.startswith("algorithm/backbone="):
            # this is a workaround to enable overriding the backbone in the command line
            # otherwise, the backbone could be re-overridden by
            # the backbone cfgs in dataset-experiment dependent cfgs
            new_argv += override_backbone(arg[19:])
        else:
            new_argv.append(arg)

    return new_argv


def override_backbone(name: str) -> list[str]:
    """
    Override the backbone with the specified name.
    """
    return ["algorithm.backbone=null"] + _yaml_to_cli(
        f"configurations/algorithm/backbone/{name}.yaml", prefix="algorithm.backbone"
    )

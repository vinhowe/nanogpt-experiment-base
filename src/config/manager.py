# pyright: standard
import os
import sys
from dataclasses import fields, is_dataclass
from typing import Any

import tyro

try:
    import tomllib  # pyright: ignore
except ModuleNotFoundError:
    import tomli as tomllib  # pyright: ignore

from .job_config import JobConfig


class ConfigManager:
    """
    Parses, merges, and validates a JobConfig from TOML and CLI sources.

    Configuration precedence: CLI args > TOML file > JobConfig defaults
    CLI arguments use the format <section>.<field> to map to TOML entries.
    Example: model.name → [model] name
    """

    config_cls: type[JobConfig]

    def __init__(self, config_cls: type[JobConfig] = JobConfig):
        self.config_cls = config_cls
        self.config: JobConfig = config_cls()
        self.register_tyro_rules(custom_registry)

    def parse_args(self, args: list[str] | None = None) -> JobConfig:
        if args is None:
            args = sys.argv[1:]

        toml_values = self._maybe_load_toml(args)

        base_config = (
            self._dict_to_dataclass(self.config_cls, toml_values)
            if toml_values
            else self.config_cls()
        )

        self.config = tyro.cli(
            self.config_cls, args=args, default=base_config, registry=custom_registry
        )

        self._validate_config()
        return self.config

    def _maybe_load_toml(self, args: list[str]) -> dict[str, Any] | None:
        """Load TOML config file if specified in CLI args."""
        # Check CLI for config file
        valid_keys = {"--job.config-file", "--job.config_file"}
        file_path = None

        for i, arg in enumerate(args):
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in valid_keys:
                    file_path = value
                    break
            elif i < len(args) - 1 and arg in valid_keys:
                file_path = args[i + 1]
                break

        if not file_path:
            return None

        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
            print(f"Error while loading config file: {file_path}")
            raise e

    def _dict_to_dataclass(self, cls, data: dict[str, Any]) -> Any:
        """Convert dictionary to dataclass, handling nested structures."""
        if not is_dataclass(cls):
            return data

        result = {}
        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                if is_dataclass(f.type) and isinstance(value, dict):
                    result[f.name] = self._dict_to_dataclass(f.type, value)
                else:
                    result[f.name] = value

        return cls(**result)  # pyright: ignore

    def _validate_config(self) -> None:
        """Validate the parsed configuration."""
        # Add custom validation logic here if needed

        # Ensure log folder exists
        if self.config.logging.log_folder:
            os.makedirs(self.config.logging.log_folder, exist_ok=True)

        # Ensure checkpoint folder exists
        if self.config.logging.checkpoint_folder:
            os.makedirs(self.config.logging.checkpoint_folder, exist_ok=True)

    @staticmethod
    def register_tyro_rules(registry: tyro.constructors.ConstructorRegistry) -> None:
        """Register custom parsing rules for tyro."""

        @registry.primitive_rule
        def list_str_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Support for comma separated string parsing"""
            if type_info.type != list[str]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="A,B,C,...",
                instance_from_str=lambda args: args[0].split(","),
                is_instance=lambda instance: all(isinstance(i, str) for i in instance),
                str_from_instance=lambda instance: [",".join(instance)],
            )


# Initialize the custom registry for tyro
custom_registry = tyro.constructors.ConstructorRegistry()


if __name__ == "__main__":
    # Run this module directly to debug or inspect configuration parsing
    from rich import print as rprint
    from rich.pretty import Pretty

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    rprint(Pretty(config))

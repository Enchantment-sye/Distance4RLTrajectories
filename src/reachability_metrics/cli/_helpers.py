"""Shared argparse and config helpers for experiment CLIs."""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Callable, Iterable, Mapping


def experiment_parser(description: str | None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def add_dataset_output_args(
    parser: argparse.ArgumentParser,
    *,
    datasets_default: list[str] | None,
    output_dir_default: str,
    include_cache_dir: bool = True,
) -> None:
    parser.add_argument("--datasets", nargs="+", default=datasets_default)
    parser.add_argument("--output_dir", default=output_dir_default)
    if include_cache_dir:
        parser.add_argument("--cache_dir", default=None)


def add_seed_minari_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minari_datasets_path", default=None)


def add_state_ik_args(parser: argparse.ArgumentParser, *, include_batch_size: bool = True) -> None:
    parser.add_argument("--ik_ensemble_size", type=int, default=100)
    parser.add_argument("--ik_subsample_size", type=int, default=32)
    parser.add_argument("--ik_temperature", type=float, default=0.01)
    if include_batch_size:
        parser.add_argument("--ik_batch_size", type=int, default=4096)
    parser.add_argument("--ik_device", default="auto")


def add_successor_ik_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ik_ensemble_sizes", nargs="+", type=int, default=[100])
    parser.add_argument("--ik_subsample_sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--ik_temperatures", nargs="+", type=float, default=[0.01])
    parser.add_argument("--ik_batch_size", type=int, default=4096)
    parser.add_argument("--ik_device", default="auto")


def config_from_args(
    config_cls: Callable[..., Any],
    args: argparse.Namespace,
    fields: Iterable[str] | None = None,
    *,
    aliases: Mapping[str, str] | None = None,
    list_fields: Iterable[str] = (),
    tuple_fields: Iterable[str] = (),
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    aliases = aliases or {}
    field_names = list(fields) if fields is not None else _arg_config_fields(config_cls, args, aliases)
    list_field_names = set(list_fields)
    tuple_field_names = set(tuple_fields)
    kwargs: dict[str, Any] = {}
    for field in field_names:
        attr = aliases.get(field, field)
        value = getattr(args, attr)
        if field in list_field_names:
            value = list(value)
        elif field in tuple_field_names:
            value = tuple(value)
        kwargs[field] = value
    if overrides:
        kwargs.update(overrides)
    return config_cls(**kwargs)


def _arg_config_fields(
    config_cls: Callable[..., Any],
    args: argparse.Namespace,
    aliases: Mapping[str, str],
) -> list[str]:
    if is_dataclass(config_cls):
        return [
            field.name
            for field in dataclass_fields(config_cls)
            if hasattr(args, aliases.get(field.name, field.name))
        ]
    return [key for key in vars(args) if key not in set(aliases.values())]

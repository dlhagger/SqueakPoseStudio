"""Compatibility export for dataset YAML creation.

New code should import from :mod:`squeakpose.services.dataset_yaml`.
"""

from squeakpose.services.dataset_yaml import create_dataset_yaml

__all__ = ["create_dataset_yaml"]

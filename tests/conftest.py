"""Shared pytest configuration and fixtures.

Provides a minimal torch.utils.data.Dataset stub so that the dump script can be
imported in CI environments where PyTorch is not installed.  When torch is already
available the real module is used unchanged.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _install_torch_stub() -> None:
    """Insert a minimal torch stub into sys.modules if torch is absent."""
    if "torch" in sys.modules:
        return

    class _Dataset:
        """Minimal stub for torch.utils.data.Dataset."""

        def __class_getitem__(cls, item: object) -> type:
            return cls

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)

    mock_data = MagicMock()
    mock_data.Dataset = _Dataset
    mock_data.DataLoader = MagicMock()

    mock_utils = MagicMock()
    mock_utils.data = mock_data

    mock_torch = MagicMock()
    mock_torch.utils = mock_utils

    sys.modules.setdefault("torch", mock_torch)
    sys.modules.setdefault("torch.utils", mock_utils)
    sys.modules.setdefault("torch.utils.data", mock_data)


_install_torch_stub()

"""Structured logging for streamllm.

Everything logs under the ``streamllm`` logger (prompt §14). We never ``print``
except in CLI user-facing output. The level is taken from ``STREAMLLM_LOG_LEVEL``
(default ``WARNING``) so that importing the library is quiet by default but a
single env var turns on the full tier-decision / cache-hit trace.
"""

from __future__ import annotations

import logging
import os

_LOGGER_NAME = "streamllm"
_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``streamllm`` namespace.

    Args:
        name: Optional dotted suffix, e.g. ``"runner"`` → ``streamllm.runner``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    _configure_root_once()
    if name is None or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def _configure_root_once() -> None:
    """Attach a single stderr handler to the ``streamllm`` logger.

    Idempotent. We attach our own handler (rather than relying on the root
    logger) so library users who haven't called ``logging.basicConfig`` still
    see warnings, but we set ``propagate = False`` to avoid double-emitting for
    those who have.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    level_name = os.environ.get("STREAMLLM_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()  # stderr
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def set_level(level: int | str) -> None:
    """Programmatically set the ``streamllm`` logger level (overrides env)."""
    _configure_root_once()
    logging.getLogger(_LOGGER_NAME).setLevel(level)

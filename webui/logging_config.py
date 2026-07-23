"""Idempotent terminal logging for the Decision Lab process."""

from __future__ import annotations

import logging
import os

_HANDLER_MARKER = "_tradingagents_terminal_handler"
_DEFAULT_LEVEL = "INFO"


def configure_terminal_logging() -> None:
    """Attach one concise terminal handler to application-owned namespaces."""

    requested = os.getenv("TRADINGAGENTS_LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = getattr(logging, requested, None)
    invalid_level = not isinstance(level, int)
    if invalid_level:
        requested = _DEFAULT_LEVEL
        level = logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s · %(message)s",
        datefmt="%H:%M:%S",
    )
    installed = False
    for namespace in ("webui", "tradingagents"):
        namespace_logger = logging.getLogger(namespace)
        namespace_logger.setLevel(level)
        namespace_logger.propagate = False
        if any(getattr(handler, _HANDLER_MARKER, False) for handler in namespace_logger.handlers):
            continue
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARKER, True)
        namespace_logger.addHandler(handler)
        installed = True

    if installed:
        logger = logging.getLogger(__name__)
        logger.info("Terminal logging ready level=%s", requested)
        if invalid_level:
            logger.warning(
                "Invalid TRADINGAGENTS_LOG_LEVEL; falling back to %s",
                _DEFAULT_LEVEL,
            )


__all__ = ["configure_terminal_logging"]

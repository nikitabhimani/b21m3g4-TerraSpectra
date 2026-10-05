"""Tests for terraspectra_pipeline.logging — plain/JSON log configuration."""

import json
import logging

from terraspectra_pipeline.logging import JsonFormatter, configure_logging, get_logger


def test_json_formatter_produces_valid_json():
    """A log record formatted as JSON should be parseable and contain the message."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello world",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["msg"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert "ts" in parsed


def test_json_formatter_includes_extra_fields():
    """Extra fields passed via logger.info(..., extra={...}) should appear in the JSON."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="processing scene",
        args=(),
        exc_info=None,
    )
    record.scene_id = "SCENE_001"

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["scene_id"] == "SCENE_001"


def test_json_formatter_includes_exception_info():
    """If exc_info is set, the formatted JSON should include an 'exc' field."""
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="something failed",
            args=(),
            exc_info=sys.exc_info(),
        )

    output = formatter.format(record)
    parsed = json.loads(output)

    assert "exc" in parsed
    assert "ValueError" in parsed["exc"]


def test_configure_logging_plain_sets_level():
    """configure_logging with default settings should set the root logger's level."""
    configure_logging(level="DEBUG", json_output=False)

    root = logging.getLogger()

    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert not isinstance(root.handlers[0].formatter, JsonFormatter)


def test_configure_logging_json_uses_json_formatter():
    """configure_logging with json_output=True should attach a JsonFormatter."""
    configure_logging(level="INFO", json_output=True)

    root = logging.getLogger()

    assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_configure_logging_quiets_noisy_loggers():
    """rasterio/httpx/httpcore loggers should be raised to at least WARNING."""
    configure_logging(level="DEBUG", json_output=False)

    assert logging.getLogger("rasterio").level >= logging.WARNING
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING


def test_configure_logging_is_idempotent():
    """Calling configure_logging twice should not stack up duplicate handlers."""
    configure_logging(level="INFO", json_output=False)
    configure_logging(level="INFO", json_output=False)

    root = logging.getLogger()

    assert len(root.handlers) == 1


def test_get_logger_returns_named_logger():
    """get_logger should return a standard logging.Logger with the given name."""
    logger = get_logger("terraspectra_pipeline.mymodule")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "terraspectra_pipeline.mymodule"
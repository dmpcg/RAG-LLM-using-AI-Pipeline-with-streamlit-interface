"""Tests for logging_config.py structured logging and secrets redaction."""

import json
import logging
from unittest.mock import patch

# ---------------------------------------------------------------------------
# _redact
# ---------------------------------------------------------------------------


class TestRedact:
    def test_redacts_password(self):
        from logging_config import _redact

        assert "***REDACTED***" in _redact("password=secret123")

    def test_redacts_api_key(self):
        from logging_config import _redact

        assert "***REDACTED***" in _redact("api_key=sk-abc123")
        assert "sk-abc123" not in _redact("api_key=sk-abc123")

    def test_redacts_token(self):
        from logging_config import _redact

        assert "***REDACTED***" in _redact("token: bearer_xyz")

    def test_redacts_credential(self):
        from logging_config import _redact

        assert "***REDACTED***" in _redact("credential=foo")

    def test_leaves_normal_text_untouched(self):
        from logging_config import _redact

        text = "Revenue is $1M and growth is 15%"
        assert _redact(text) == text

    def test_case_insensitive(self):
        from logging_config import _redact

        assert "***REDACTED***" in _redact("PASSWORD=mysecret")
        assert "***REDACTED***" in _redact("Api-Key=mykey")

    def test_multiple_secrets_all_redacted(self):
        from logging_config import _redact

        text = "password=abc token=xyz"
        result = _redact(text)
        assert "abc" not in result
        assert "xyz" not in result


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    def test_formats_as_valid_json(self):
        from logging_config import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Hello world"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_redacts_secrets_in_message(self):
        from logging_config import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Connection password=supersecret failed",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "supersecret" not in parsed["message"]
        assert "***REDACTED***" in parsed["message"]

    def test_includes_exception_info(self):
        from logging_config import JSONFormatter

        formatter = JSONFormatter()
        try:
            raise ValueError("bad value")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["exception"]["type"] == "ValueError"

    def test_redacts_secrets_in_exception_message(self):
        from logging_config import JSONFormatter

        formatter = JSONFormatter()
        try:
            raise RuntimeError("token=leaked_value problem")
        except RuntimeError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "leaked_value" not in parsed["exception"]["message"]


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_default_text_format(self):
        from logging_config import setup_logging

        with patch.dict("os.environ", {}, clear=False):
            setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) >= 1

    def test_json_format(self):
        from logging_config import JSONFormatter, setup_logging

        with patch.dict("os.environ", {"LOG_FORMAT": "json"}):
            setup_logging()
        root = logging.getLogger()
        assert any(isinstance(h.formatter, JSONFormatter) for h in root.handlers)

    def test_custom_log_level(self):
        from logging_config import setup_logging

        with patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"}):
            setup_logging()
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_suppresses_noisy_loggers(self):
        from logging_config import setup_logging

        with patch.dict("os.environ", {}, clear=False):
            setup_logging()
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING

"""Logging handler that writes to tqdm progress bar."""

import logging

import tqdm


class TqdmLoggingHandler(logging.Handler):
    """TqdmLoggingHandler."""

    def __init__(self, level=logging.INFO):
        """Initialize TqdmLoggingHandler."""
        super().__init__(level)

    def emit(self, record):
        """Emit a record."""
        try:
            msg = self.format(record)
            tqdm.auto.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

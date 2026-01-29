"""Environment helpers for curi1_control."""
import os


def setup_qt_environment() -> None:
    """Prepare Qt environment variables for WSL/desktop usage."""
    if 'QT_QPA_PLATFORM' in os.environ:
        del os.environ['QT_QPA_PLATFORM']
    if 'DISPLAY' in os.environ:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        os.environ['QT_LOGGING_RULES'] = '*=false'
        os.environ['QT_DEBUG_PLUGINS'] = '0'


class SuppressQtWarnings:
    """Context manager used to suppress Qt stderr warnings."""

    def __init__(self) -> None:
        self.null_fd = None
        self.old_stderr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def suppress_qt_warnings() -> None:
    """Filter QObject::moveToThread warnings printed by Qt."""
    import io
    import sys

    class _FilteredStderr(io.TextIOBase):
        def __init__(self, original):
            self.original = original

        def write(self, text):
            if "QObject::moveToThread" not in text and "Cannot move to target thread" not in text:
                return self.original.write(text)
            return len(text)

        def flush(self):
            return self.original.flush()

    sys.stderr = _FilteredStderr(sys.stderr)

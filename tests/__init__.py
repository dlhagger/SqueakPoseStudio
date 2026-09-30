"""Shared test-process isolation."""

import os
import tempfile

# UI construction must never discover or refresh a developer's real ChatGPT token.
os.environ.setdefault(
    "SQUEAKPOSE_OPENAI_CONFIG_DIR",
    os.path.join(tempfile.gettempdir(), f"squeakpose-openai-tests-{os.getpid()}"),
)

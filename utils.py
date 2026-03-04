"""Shared utility helpers for shotrename."""

import os
import sys
from pathlib import Path


_ANSI_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_RESET = "\033[0m"
_STYLES = {
    "title": "\033[1;38;5;45m",
    "muted": "\033[38;5;250m",
    "dim": "\033[2;38;5;245m",
    "info": "\033[1;38;5;81m",
    "scan": "\033[1;38;5;39m",
    "rename": "\033[1;38;5;42m",
    "preview": "\033[1;38;5;220m",
    "skip": "\033[1;38;5;214m",
    "error": "\033[1;38;5;196m",
    "filename": "\033[1;38;5;159m",
    "caption": "\033[38;5;188m",
    "rule": "\033[38;5;240m",
}


def style(text: str, tone: str) -> str:
    if not _ANSI_ENABLED:
        return text
    return f"{_STYLES.get(tone, '')}{text}{_RESET}"


def status_label(name: str, tone: str) -> str:
    return style(f"[{name}]", tone)


def section_title(title: str) -> str:
    return style(title, "title")


def muted(text: str) -> str:
    return style(text, "muted")


def dim(text: str) -> str:
    return style(text, "dim")


def rule(char: str = "─", width: int = 64) -> str:
    return style(char * width, "rule")


def key_value(label: str, value: str, tone: str = "muted") -> str:
    return f"{style(label, tone)} {value}"


def print_banner() -> None:
    lines = [
        section_title("shotrename"),
        muted("terminal-first screenshot renamer"),
    ]
    print("\n".join(lines))


def human_readable_size(path: Path) -> str:
    """Return human-readable file size string."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

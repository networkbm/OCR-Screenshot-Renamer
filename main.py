#!/usr/bin/env python3
"""CLI entry point for the screenshot renamer."""

import argparse
import sys
from pathlib import Path

from renamer import FolderRenamer
from utils import key_value, muted, status_label


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("folder", type=str, help="Path to the folder containing screenshots.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan subfolders.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "blip", "llava", "ocr-only"],
        help="Local backend: 'auto' (BLIP then OCR fallback), 'blip', 'llava', or 'ocr-only'.",
    )
    parser.add_argument(
        "--llava-model",
        type=str,
        default=None,
        help="Optional Hugging Face model id to use with --backend llava.",
    )


def _build_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shotrename",
        description="Scan screenshot folders and rename images from local image analysis.",
    )
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser(
        "scan",
        help="Preview screenshot renames without modifying files.",
    )
    _add_common_arguments(scan_parser)

    rename_parser = subparsers.add_parser(
        "rename",
        help="Rename screenshot files on disk.",
    )
    _add_common_arguments(rename_parser)
    return parser


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shotrename",
        description="Legacy compatibility mode for folder-first invocation.",
    )
    parser.add_argument("folder", type=str, help="Path to the folder containing screenshots.")
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Rename files on disk instead of previewing changes.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan subfolders.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "blip", "llava", "ocr-only"],
        help="Local backend: 'auto' (BLIP then OCR fallback), 'blip', 'llava', or 'ocr-only'.",
    )
    parser.add_argument(
        "--llava-model",
        type=str,
        default=None,
        help="Optional Hugging Face model id to use with --backend llava.",
    )
    return parser


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command_names = {"scan", "rename"}

    if not argv or argv[0] in command_names or argv[0] in {"-h", "--help"}:
        return _build_command_parser().parse_args(argv)

    args = _build_legacy_parser().parse_args(argv)
    args.command = "rename" if args.apply else "scan"
    return args


def _resolve_folder(folder_arg: str) -> Path:
    folder = Path(folder_arg).expanduser()
    if not folder.exists() or not folder.is_dir():
        print(f"{status_label('ERROR', 'error')} Folder not found: {folder}")
        sys.exit(1)
    return folder


def main(argv=None):
    args = parse_args(argv)
    folder = _resolve_folder(args.folder)
    dry_run = args.command == "scan"

    if dry_run:
        print(f"{status_label('SCAN', 'scan')} Previewing renames")
        print(key_value("Folder:", str(folder)))
        print(key_value("Mode:", "dry run", "scan"))
        print(muted("Use the 'rename' command to commit changes.\n"))
    else:
        print(f"{status_label('RENAME', 'rename')} Applying renames")
        print(key_value("Folder:", str(folder)))
        print(key_value("Mode:", "write changes", "rename"))
        print()

    renamer = FolderRenamer(backend=args.backend, llava_model=args.llava_model)
    renamer.process_folder(folder, recursive=args.recursive, dry_run=dry_run)


if __name__ == "__main__":
    main()

"""
renamer.py — Folder scanning and renaming engine.

Orchestrates:
  1. Scan folder for supported images
  2. Caption each image via the selected backend
  3. Generate clean filename
  4. Resolve duplicates
  5. Rename (or preview in dry-run mode)
"""

from pathlib import Path
from typing import Optional

from captioner import get_captioner
from filename_cleaner import (
    caption_to_filename,
    is_low_signal_caption,
    is_low_signal_filename,
    make_unique_filename,
)
from utils import dim, key_value, muted, rule, status_label, style

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}


class FolderRenamer:
    def __init__(self, backend: str = "auto", llava_model: Optional[str] = None):
        self.backend = backend
        self.llava_model = llava_model
        self._captioner = None  # lazy init so we only load models when needed

    @property
    def captioner(self):
        if self._captioner is None:
            print(f"{status_label('INFO', 'info')} Initializing backend: {style(self.backend, 'filename')}")
            self._captioner = get_captioner(self.backend, llava_model=self.llava_model)
        return self._captioner

    def process_folder(self, folder: Path, recursive: bool = False, dry_run: bool = True):
        """
        Main entry point. Scans folder and renames (or previews) images.
        """
        images = self._collect_images(folder, recursive)

        if not images:
            print(f"{status_label('INFO', 'info')} No supported images found in {folder}")
            return

        print(f"{status_label('INFO', 'info')} Found {style(str(len(images)), 'filename')} image(s) to process.\n")

        # Track used filenames per-directory to handle duplicates correctly
        used_names: dict[Path, set] = {}

        results = []
        for image_path in images:
            parent = image_path.parent
            if parent not in used_names:
                # Pre-populate with existing filenames in this directory
                used_names[parent] = {
                    p.name for p in parent.iterdir() if p.is_file()
                }

            result = self._process_image(image_path, used_names[parent])
            if result:
                results.append((image_path, result["status"], result["name"], result["reason"]))
                used_names[parent].add(result["name"])

        # Summary
        print("\n" + rule())
        renamed_count = 0
        skipped_count = 0

        for image_path, status, new_name, reason in results:
            old_name = image_path.name
            new_path = image_path.parent / new_name

            if status == "skip":
                print(f"  {status_label('SKIP', 'skip')} {style(old_name, 'muted')}  {dim(f'({reason})')}")
                skipped_count += 1
                continue

            if dry_run:
                print(f"  {status_label('PREVIEW', 'preview')} {style(old_name, 'muted')}")
                print(f"           {style('→', 'preview')}  {style(new_name, 'filename')}\n")
            else:
                try:
                    image_path.rename(new_path)
                    print(f"  {status_label('RENAMED', 'rename')} {style(old_name, 'muted')}")
                    print(f"           {style('→', 'rename')}  {style(new_name, 'filename')}\n")
                    renamed_count += 1
                except OSError as e:
                    print(f"  {status_label('ERROR', 'error')} {old_name} {dim('-')} {e}")

        print(rule())
        if dry_run:
            print(f"\n{status_label('SCAN COMPLETE', 'scan')} {style(str(len(results) - skipped_count), 'filename')} file(s) would be renamed.")
            print(muted("Run with the 'rename' command to commit changes."))
        else:
            print(f"\n{status_label('DONE', 'rename')} {style(str(renamed_count), 'filename')} renamed, {style(str(skipped_count), 'skip')} skipped.")

    def _collect_images(self, folder: Path, recursive: bool) -> list[Path]:
        """Collect all supported image files in the folder."""
        pattern = "**/*" if recursive else "*"
        images = []
        for path in sorted(folder.glob(pattern)):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(path)
        return images

    def _process_image(self, image_path: Path, used_names: set) -> Optional[dict]:
        """Caption one image and return the new filename."""
        print(f"  {status_label('FILE', 'info')} {style(image_path.name, 'filename')}")
        try:
            caption = self.captioner.caption(image_path)
            print(key_value("    Caption:", style(caption, 'caption')))

            if is_low_signal_caption(caption):
                print(f"    {status_label('SKIP', 'skip')} {muted('Caption too generic to safely rename')}")
                return {
                    "status": "skip",
                    "name": image_path.name,
                    "reason": "caption too generic",
                }

            ext = image_path.suffix.lower()
            desired = caption_to_filename(caption, extension=ext)
            if is_low_signal_filename(desired):
                print(f"    {status_label('SKIP', 'skip')} {muted('Generated filename too generic to safely rename')}")
                return {
                    "status": "skip",
                    "name": image_path.name,
                    "reason": "generated filename too generic",
                }
            existing_other_names = set(used_names)
            existing_other_names.discard(image_path.name)
            unique = make_unique_filename(desired, existing_other_names)
            print(key_value("    Filename:", style(unique, 'filename')))
            return {
                "status": "skip" if unique == image_path.name else "rename",
                "name": unique,
                "reason": "already a good name" if unique == image_path.name else "",
            }

        except Exception as e:
            print(f"    {status_label('ERROR', 'error')} Could not process {image_path.name}: {e}")
            return None

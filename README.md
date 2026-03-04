# shotrename

Renames screenshots based on visible text and image content. It scans a folder of images and turns generic names into readable filenames like `aws-vpc-route-tables.png`.

---

## Requirements

```bash
pip install pytesseract Pillow
brew install tesseract
```

Optional local vision backends:

```bash
pip install transformers torch accelerate Pillow
```

## Launch

From the project folder:

```bash
cd /path/to/shotrename
python3 main.py scan /path/to/screenshots
python3 main.py rename /path/to/screenshots
```

Optional installed command:

```bash
python3 -m pip install --user -e .
shotrename scan /path/to/screenshots
shotrename rename /path/to/screenshots
```

---

### Examples

```bash
# Preview renames
python3 main.py scan /path/to/screenshots

# Apply renames
python3 main.py rename /path/to/screenshots

# Recursively rename all images in nested folders
python3 main.py rename /path/to/screenshots --recursive

# OCR only
python3 main.py scan /path/to/screenshots --backend ocr-only

# BLIP
python3 main.py scan /path/to/screenshots --backend blip

# LLaVA
python3 main.py scan /path/to/screenshots --backend llava
```

---

## Backends

| Backend | Accuracy | Privacy | Requires |
|---------|----------|---------|----------|
| `auto` | ⭐⭐⭐ | 100% local | OCR first, then BLIP if OCR is weak |
| `blip` | ⭐⭐⭐ | 100% local | `transformers`, `torch` |
| `llava` | ⭐⭐⭐⭐ | 100% local | `transformers`, `torch`, larger model weights |
| `ocr-only` | ⭐⭐ | 100% local | `pytesseract` + system Tesseract |

---

## Supported Image Formats

`.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tiff`

---

## Tips

- Use `python3 main.py scan ...` first to verify filenames before committing
- Start with `auto`
- Use `ocr-only` for speed
- Use `llava` only when you need stronger image understanding
- `--recursive` processes nested folder trees

---

## Screenshot

![shotrename output](Screenshots/Terminal-Output.png)

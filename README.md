# OCR Screenshot Renamer

Renames screenshots based on visible text and image content. It scans a folder of images and turns generic names into readable filenames like `aws-vpc-route-tables.png`.

---


## macOS Commands

```bash
pip install pytesseract Pillow
brew install tesseract
```

Optional local vision backends:

```bash
pip install transformers torch accelerate Pillow
```

## Windows Commands

```bash
pip install pytesseract Pillow
winget install UB-Mannheim.TesseractOCR
```

Optional local vision backends:

```powershell
pip install transformers torch accelerate Pillow
```

## Launch (macOS)

```bash
cd /path/to/OCR-Screenshot-Renamer
python3 main.py scan /path/to/screenshots
python3 main.py rename /path/to/screenshots
```

## Launch (Windows)

```bash
cd C:\path\to\OCR-Screenshot-Renamer
python main.py scan C:\path\to\screenshots
python main.py rename C:\path\to\screenshots
```

## Supported Image Formats

`.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tiff`

## Screenshot

![Overview](Screenshots/Terminal-Output.png)

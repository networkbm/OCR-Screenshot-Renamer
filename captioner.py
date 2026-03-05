"""
captioner.py — Local vision and OCR captioning module.

Supports three backends:
  - auto       : Try OCR first, then fall back to BLIP if OCR is weak
  - blip       : Local BLIP image captioning via HuggingFace Transformers
  - llava      : Local LLaVA vision-language model via HuggingFace Transformers
  - ocr-only   : Tesseract OCR text extraction only
"""

import re
import shutil
import platform
from contextlib import redirect_stderr, redirect_stdout
from os import environ
from pathlib import Path
from tempfile import TemporaryFile


OCR_NOISE_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "page",
    "screen",
    "window",
    "button",
    "menu",
    "search",
    "dashboard",
    "settings",
    "home",
    "open",
    "close",
    "cancel",
    "save",
    "edit",
    "view",
}
OCR_HINT_WORDS = {
    "aws",
    "azure",
    "gcp",
    "github",
    "gitlab",
    "docker",
    "kubernetes",
    "terraform",
    "grafana",
    "datadog",
    "jenkins",
    "vercel",
    "netlify",
    "route",
    "routes",
    "subnet",
    "subnets",
    "vpc",
    "cluster",
    "namespace",
    "pod",
    "pods",
    "logs",
    "policy",
    "policies",
    "workflow",
    "deploy",
    "deployment",
    "project",
    "repo",
    "repository",
    "billing",
    "invoice",
    "alert",
    "alerts",
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseCaptioner:
    def caption(self, image_path: Path) -> str:
        raise NotImplementedError


def _normalize_ocr_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9._:/-]+$", "", token)
    token = token.replace("|", "")
    return token


def _token_score(token: str) -> int:
    lowered = token.lower()
    score = 0
    if len(lowered) >= 3:
        score += 1
    if lowered in OCR_HINT_WORDS:
        score += 4
    if any(char.isdigit() for char in token):
        score += 2
    if any(char in token for char in "._:/-"):
        score += 2
    if token != lowered:
        score += 1
    if lowered in OCR_NOISE_WORDS:
        score -= 2
    return score


def _is_id_like_token(token: str) -> bool:
    lowered = token.lower()
    if len(lowered) >= 14 and any(ch.isdigit() for ch in lowered) and any(ch.isalpha() for ch in lowered):
        return True
    if len(lowered) >= 10 and lowered.count("-") >= 2 and any(ch.isdigit() for ch in lowered):
        return True
    if lowered.endswith(".png") or lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return True
    return False


def _is_word_like_token(token: str) -> bool:
    lowered = token.lower()
    if _is_id_like_token(lowered):
        return False
    alpha_count = sum(1 for ch in lowered if ch.isalpha())
    return alpha_count >= 3


def _dedupe_preserving_order(tokens: list[str]) -> list[str]:
    seen = set()
    result = []
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
    return result


def _quiet_model_load(loader):
    with TemporaryFile(mode="w+") as stdout_buffer, TemporaryFile(mode="w+") as stderr_buffer:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            return loader()


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


# ---------------------------------------------------------------------------
# BLIP (Local HuggingFace)
# ---------------------------------------------------------------------------

class BlipCaptioner(BaseCaptioner):
    """
    Uses BLIP locally via HuggingFace Transformers.
    Privacy-friendly — no images leave your machine.
    Requires: pip install transformers torch Pillow
    """

    def __init__(self):
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from transformers.utils import logging as transformers_logging
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "BLIP backend requires: pip install transformers torch Pillow\n"
                "Or use --backend ocr-only for a lighter local fallback."
            )

        print("[BLIP] Loading BLIP model (first run downloads ~1.9GB)...")
        self._Image = Image
        self._torch = torch
        transformers_logging.set_verbosity_error()
        self.processor = _quiet_model_load(
            lambda: BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        )
        self.model = _quiet_model_load(
            lambda: BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[BLIP] Model loaded on {self.device}.")

    def caption(self, image_path: Path) -> str:
        image = self._Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()


class LlavaCaptioner(BaseCaptioner):
    """
    Uses a local LLaVA checkpoint via HuggingFace Transformers.
    This backend is more capable on dense technical screenshots, but much heavier than BLIP.
    """

    DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
    PROMPT = (
        "Describe this screenshot in one short phrase for a filename. "
        "Focus on the main tool, screen, or action. No punctuation."
    )

    def __init__(self, model_name: str | None = None):
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            from transformers.utils import logging as transformers_logging
        except ImportError:
            raise ImportError(
                "LLaVA backend requires: pip install transformers torch accelerate Pillow\n"
                "Then run with --backend llava."
            )

        self._torch = torch
        self._Image = Image
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[LLaVA] Loading model: {self.model_name}")
        transformers_logging.set_verbosity_error()
        self.processor = _quiet_model_load(lambda: AutoProcessor.from_pretrained(self.model_name))
        self.model = _quiet_model_load(
            lambda: LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            )
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[LLaVA] Model loaded on {self.device}.")

    def caption(self, image_path: Path) -> str:
        image = self._Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{self.PROMPT} ASSISTANT:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with self._torch.inference_mode():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if "ASSISTANT:" in generated_text:
            return generated_text.split("ASSISTANT:", 1)[1].strip()
        return generated_text.strip()


class AutoCaptioner(BaseCaptioner):
    """Select the best available local backend."""

    def __init__(self):
        self._ocr_captioner = None
        self._vision_captioner = None
        self.backend_name = "unavailable"

        ocr_error = None
        try:
            self._ocr_captioner = OcrCaptioner()
            self.backend_name = "ocr-only"
        except Exception as exc:
            ocr_error = exc

        blip_error = None
        try:
            self._vision_captioner = BlipCaptioner()
            if self._ocr_captioner is None:
                self.backend_name = "blip"
        except Exception as exc:
            blip_error = exc

        if self._ocr_captioner is None and self._vision_captioner is None:
            raise RuntimeError(
                "No local captioning backend is available.\n"
                "Install OCR dependencies: pip install pytesseract Pillow\n"
                "Or install BLIP dependencies: pip install transformers torch accelerate Pillow"
            ) from (ocr_error or blip_error)

    def caption(self, image_path: Path) -> str:
        from filename_cleaner import is_low_signal_caption

        ocr_caption = None
        if self._ocr_captioner is not None:
            try:
                ocr_caption = self._ocr_captioner.caption(image_path)
                if not is_low_signal_caption(ocr_caption):
                    print("    [AUTO] Using OCR-derived caption")
                    return ocr_caption
                print("    [AUTO] OCR signal weak, trying BLIP fallback")
            except Exception as exc:
                print(f"    [AUTO] OCR failed, trying BLIP fallback: {exc}")

        if self._vision_captioner is not None:
            return self._vision_captioner.caption(image_path)

        return ocr_caption or "unknown-screenshot"


# ---------------------------------------------------------------------------
# OCR-Only (Tesseract)
# ---------------------------------------------------------------------------

class OcrCaptioner(BaseCaptioner):
    """
    Extracts visible text from the image using Tesseract OCR.
    Useful as a fast fallback when vision models aren't available.
    Requires: pip install pytesseract Pillow  +  system tesseract binary
    """

    def __init__(self):
        try:
            import pytesseract
            from PIL import Image
            from pytesseract import Output
        except ImportError:
            raise ImportError(
                "OCR backend requires: pip install pytesseract Pillow\n"
                "And Tesseract installed: https://github.com/tesseract-ocr/tesseract"
            )
        self._pytesseract = pytesseract
        self._configure_tesseract_binary()
        self._Image = Image
        self._Output = Output

    def _configure_tesseract_binary(self) -> None:
        """
        Ensure pytesseract can locate the system tesseract executable.
        On Windows, many installs do not add Tesseract to PATH by default.
        """
        cmd = getattr(self._pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        if shutil.which(cmd):
            return

        env_value = environ.get("TESSERACT_CMD")
        if env_value and Path(env_value).exists():
            self._pytesseract.pytesseract.tesseract_cmd = env_value
            return

        windows_candidates = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        )
        for candidate in windows_candidates:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                self._pytesseract.pytesseract.tesseract_cmd = str(candidate_path)
                return

    def caption(self, image_path: Path) -> str:
        if platform.system() != "Windows":
            return self._caption_standard(image_path)
        return self._caption_windows_tuned(image_path)

    def _caption_standard(self, image_path: Path) -> str:
        image = self._Image.open(image_path).convert("RGB")
        data = self._pytesseract.image_to_data(image, output_type=self._Output.DICT)

        tokens = []
        line_buckets: dict[tuple[int, int, int], list[dict]] = {}
        for index, (raw_text, raw_conf) in enumerate(zip(data["text"], data["conf"])):
            token = _normalize_ocr_token(raw_text)
            if not token:
                continue
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = -1
            if confidence < 35:
                continue
            try:
                height = float(data["height"][index])
            except (TypeError, ValueError, KeyError):
                height = 0.0

            token_info = {
                "token": token,
                "score": _token_score(token),
                "confidence": confidence,
                "height": height,
            }
            tokens.append(token_info)

            line_key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            line_buckets.setdefault(line_key, []).append(token_info)

        heights = [item["height"] for item in tokens if item["height"] > 0]
        median_height = _median(heights)
        large_text_threshold = max(median_height * 1.35, median_height + 4.0)

        prominent_lines = []
        for line_tokens in line_buckets.values():
            avg_height = sum(item["height"] for item in line_tokens) / len(line_tokens)
            avg_score = sum(item["score"] for item in line_tokens) / len(line_tokens)
            prominent_lines.append(
                {
                    "tokens": [item["token"] for item in line_tokens],
                    "avg_height": avg_height,
                    "avg_score": avg_score,
                }
            )

        prominent_lines.sort(
            key=lambda item: (item["avg_height"], item["avg_score"], len(item["tokens"])),
            reverse=True,
        )

        large_text_tokens: list[str] = []
        for line in prominent_lines:
            if line["avg_height"] < large_text_threshold:
                continue
            large_text_tokens.extend(line["tokens"])
            if len(large_text_tokens) >= 8:
                break

        large_text_tokens = _dedupe_preserving_order(large_text_tokens)
        if len(large_text_tokens) >= 2:
            return " ".join(large_text_tokens[:8])

        high_signal = [
            token
            for token, score, confidence in (
                (item["token"], item["score"], item["confidence"]) for item in tokens
            )
            if score >= 2 or confidence >= 70
        ]
        high_signal = _dedupe_preserving_order(high_signal)
        if high_signal:
            return " ".join(high_signal[:8])

        text = self._pytesseract.image_to_string(image)
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 4]
        lines = [line for line in lines if any(ch.isalnum() for ch in line)]
        if lines:
            ranked_lines = sorted(
                lines,
                key=lambda line: sum(_token_score(token) for token in line.split()),
                reverse=True,
            )
            best_line = ranked_lines[0]
            words = [_normalize_ocr_token(word) for word in best_line.split()]
            words = [word for word in words if word]
            words = _dedupe_preserving_order(words)
            if words:
                return " ".join(words[:8])
        return "unknown-screenshot"

    def _caption_windows_tuned(self, image_path: Path) -> str:
        image = self._Image.open(image_path).convert("RGB")
        ocr_configs = (
            "--oem 3 --psm 6",
            "--oem 3 --psm 11",
        )
        image_variants = (
            image,
            self._enhance_for_ocr(image),
        )

        token_rows = []
        for variant in image_variants:
            for config in ocr_configs:
                try:
                    data = self._pytesseract.image_to_data(
                        variant,
                        output_type=self._Output.DICT,
                        config=config,
                    )
                except Exception:
                    continue
                token_rows.extend(self._extract_token_rows(data))

        if not token_rows:
            return "unknown-screenshot"

        heights = [item["height"] for item in token_rows if item["height"] > 0]
        median_height = _median(heights)
        large_text_threshold = max(median_height * 1.35, median_height + 4.0)
        large_candidate_threshold = max(median_height * 1.15, median_height + 2.0)

        tokens = []
        line_buckets: dict[tuple[int, int, int], list[dict]] = {}
        for row in token_rows:
            confidence = row["confidence"]
            height = row["height"]
            if confidence < 30 and height < large_candidate_threshold:
                continue

            token_info = {
                "token": row["token"],
                "score": _token_score(row["token"]),
                "confidence": confidence,
                "height": height,
            }
            tokens.append(token_info)
            line_key = row["line_key"]
            line_buckets.setdefault(line_key, []).append(token_info)

        prominent_lines = []
        for line_tokens in line_buckets.values():
            avg_height = sum(item["height"] for item in line_tokens) / len(line_tokens)
            avg_score = sum(item["score"] for item in line_tokens) / len(line_tokens)
            tokens_only = [item["token"] for item in line_tokens]
            word_like_count = sum(1 for token in tokens_only if _is_word_like_token(token))
            id_like_count = sum(1 for token in tokens_only if _is_id_like_token(token))
            quality = (avg_height * 0.8) + (avg_score * 1.4) + (word_like_count * 3.0) - (id_like_count * 2.5)
            prominent_lines.append(
                {
                    "tokens": tokens_only,
                    "avg_height": avg_height,
                    "avg_score": avg_score,
                    "quality": quality,
                }
            )

        prominent_lines.sort(
            key=lambda item: (item["quality"], item["avg_height"], len(item["tokens"])),
            reverse=True,
        )

        large_text_tokens: list[str] = []
        for line in prominent_lines:
            if line["avg_height"] < large_text_threshold:
                continue
            large_text_tokens.extend(line["tokens"])
            if len(large_text_tokens) >= 8:
                break

        large_text_tokens = _dedupe_preserving_order(large_text_tokens)
        if len(large_text_tokens) >= 2:
            filtered_large_text_tokens = [token for token in large_text_tokens if not _is_id_like_token(token)]
            filtered_large_text_tokens = _dedupe_preserving_order(filtered_large_text_tokens)
            if self._is_good_caption_candidate(filtered_large_text_tokens):
                return " ".join(filtered_large_text_tokens[:8])

        high_signal = [
            token
            for token, score, confidence in (
                (item["token"], item["score"], item["confidence"]) for item in tokens
            )
            if (score >= 2 or confidence >= 65) and not _is_id_like_token(token)
        ]
        high_signal = _dedupe_preserving_order(high_signal)
        if self._is_good_caption_candidate(high_signal):
            return " ".join(high_signal[:8])

        text = self._pytesseract.image_to_string(
            self._enhance_for_ocr(image),
            config="--oem 3 --psm 6",
        )
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 4]
        lines = [line for line in lines if any(ch.isalnum() for ch in line)]
        if lines:
            ranked_lines = sorted(
                lines,
                key=lambda line: sum(_token_score(token) for token in line.split()),
                reverse=True,
            )
            best_line = ranked_lines[0]
            words = [_normalize_ocr_token(word) for word in best_line.split()]
            words = [word for word in words if word]
            words = [word for word in words if not _is_id_like_token(word)]
            words = _dedupe_preserving_order(words)
            if self._is_good_caption_candidate(words):
                return " ".join(words[:8])
        return "unknown-screenshot"

    def _is_good_caption_candidate(self, tokens: list[str]) -> bool:
        if not tokens:
            return False
        word_like = [token for token in tokens if _is_word_like_token(token)]
        if len(word_like) < 2:
            return False
        meaningful = [token for token in word_like if token.lower() not in OCR_NOISE_WORDS]
        return len(meaningful) >= 2

    def _enhance_for_ocr(self, image):
        """Lightweight preprocessing to improve OCR on dense UI screenshots."""
        from PIL import ImageOps

        grayscale = ImageOps.grayscale(image)
        enhanced = ImageOps.autocontrast(grayscale, cutoff=1)
        # Upscale to improve recognition of small UI text.
        width, height = enhanced.size
        return enhanced.resize((width * 2, height * 2), self._Image.Resampling.BICUBIC)

    def _extract_token_rows(self, data: dict) -> list[dict]:
        rows = []
        text_values = data.get("text", [])
        conf_values = data.get("conf", [])
        for index, (raw_text, raw_conf) in enumerate(zip(text_values, conf_values)):
            token = _normalize_ocr_token(raw_text)
            if not token:
                continue

            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = -1.0

            try:
                height = float(data["height"][index])
            except (TypeError, ValueError, KeyError, IndexError):
                height = 0.0

            try:
                line_key = (
                    int(data["block_num"][index]),
                    int(data["par_num"][index]),
                    int(data["line_num"][index]),
                )
            except (TypeError, ValueError, KeyError, IndexError):
                line_key = (0, 0, index)

            rows.append(
                {
                    "token": token,
                    "confidence": confidence,
                    "height": height,
                    "line_key": line_key,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_captioner(backend: str, llava_model: str | None = None) -> BaseCaptioner:
    if backend == "auto":
        return AutoCaptioner()
    elif backend == "blip":
        return BlipCaptioner()
    elif backend == "llava":
        return LlavaCaptioner(model_name=llava_model)
    elif backend == "ocr-only":
        return OcrCaptioner()
    else:
        raise ValueError(f"Unknown backend: {backend}")

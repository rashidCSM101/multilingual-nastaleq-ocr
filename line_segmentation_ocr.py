"""
=============================================================================
Multilingual Nastaleeq OCR — Segmentation-Free Line OCR Pipeline
=============================================================================
Pipeline:
  [Page Image] 
       ↓
  [Line Segmentation]  →  (Robust Horizontal Projection se lines crop karna)
       ↓
  [CRNN Model + CTC]   →  (Har line seedhi trained model mein jayegi)
       ↓
  [Urdu Text Output]   →  (Segmentation-Free direct Urdu text recognition)
=============================================================================
"""

import os
import sys
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --- Default Vocabulary (74 characters) ---
URDU_VOCAB_74 = [
    '<BLANK>', ' ', 'آ', 'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ',
    'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'ں', 'و', 'ؤ', 'ہ',
    'ۂ', 'ۃ', 'ھ', 'ء', 'ی', 'ئ', 'ے', '۰', '۱', '۲', '۳', '۴', '۵',
    '۶', '۷', '۸', '۹', '۔', '،', '؟', '!', '"', '(', ')', ':', ';',
    '-', '/', '٪', 'ؒ', 'ؓ', 'ؔ', 'ؐ', 'ؑ', '؛'
]


# =============================================================================
# 1. CRNN Model Architecture (ResNet-34 + BiLSTM + CTC)
# =============================================================================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )

    def forward(self, x):
        return self.feature_extractor(x)


class UrduCRNN(nn.Module):
    def __init__(self, vocab_size=74, hidden_size=256):
        super().__init__()
        self.cnn = ResNetFeatureExtractor()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(256, hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        feat = self.cnn(x)
        feat = self.adaptive_pool(feat).squeeze(2).permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        return self.fc(lstm_out).log_softmax(2)


def ctc_decode(predictions, vocab):
    """
    Standard CTC greedy decoding (collapses consecutive duplicates & drops blank=0)
    """
    arg_maxes = torch.argmax(predictions, dim=2)[0].tolist()
    decoded, prev = [], -1
    for idx in arg_maxes:
        if idx != 0 and idx != prev and idx < len(vocab):
            decoded.append(vocab[idx])
        prev = idx
    return ''.join(decoded)


# =============================================================================
# 2. Robust Line Segmenter (Horizontal Projection Profile)
# =============================================================================
class LineSegmenter:
    """
    Robust horizontal projection profile:
    - Adaptive + Otsu binarization specifically for finding line gaps
    - Detects genuine text line bands
    - Avoids merging whole page into 1 line
    """
    def __init__(self, gap_fraction=0.015, min_line_h=15, min_line_w=30, margin_frac=0.02):
        self.gap_fraction = gap_fraction
        self.min_line_h = min_line_h
        self.min_line_w = min_line_w
        self.margin_frac = margin_frac

    def segment(self, gray_image):
        H, W = gray_image.shape

        # Sharpen slightly for clear contrast
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # Robust Otsu Thresholding for projection
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Mask page borders (removes book edge shadows)
        mx = max(6, int(W * self.margin_frac))
        my = max(6, int(H * self.margin_frac))
        binary[:my, :] = 0
        binary[-my:, :] = 0
        binary[:, :mx] = 0
        binary[:, -mx:] = 0

        # Horizontal projection (dark pixels per row)
        row_sums = np.sum(binary > 0, axis=1).astype(np.float32)

        # Smooth curve to avoid small diacritic cuts
        kernel_size = max(5, int(H * 0.005))
        if kernel_size % 2 == 0:
            kernel_size += 1
        row_sums_smooth = np.convolve(row_sums, np.ones(kernel_size) / kernel_size, mode='same')

        # Threshold for detecting line vs gap
        gap_threshold = max(5.0, self.gap_fraction * W)
        is_gap = row_sums_smooth < gap_threshold

        # Find line bands (start, end rows)
        line_bands = []
        in_text = False
        start_row = 0
        for r in range(H):
            if not is_gap[r] and not in_text:
                in_text = True
                start_row = r
            elif is_gap[r] and in_text:
                in_text = False
                if (r - start_row) >= self.min_line_h:
                    line_bands.append((start_row, r - 1))
        if in_text and (H - start_row) >= self.min_line_h:
            line_bands.append((start_row, H - 1))

        # Fallback: Agar sirf 1 band bana jo 60% se zyada page le raha hai, Adaptive thresholding use karein
        if len(line_bands) <= 1 and H > 200:
            binary_adapt = cv2.adaptiveThreshold(
                gray_image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                25, 11
            )
            binary_adapt[:my, :] = 0
            binary_adapt[-my:, :] = 0
            binary_adapt[:, :mx] = 0
            binary_adapt[:, -mx:] = 0
            row_sums_adapt = np.sum(binary_adapt > 0, axis=1).astype(np.float32)
            row_sums_adapt_smooth = np.convolve(row_sums_adapt, np.ones(kernel_size) / kernel_size, mode='same')
            mean_val = np.mean(row_sums_adapt_smooth)
            is_gap_adapt = row_sums_adapt_smooth < (mean_val * 0.4)
            
            line_bands = []
            in_text = False
            for r in range(H):
                if not is_gap_adapt[r] and not in_text:
                    in_text = True
                    start_row = r
                elif is_gap_adapt[r] and in_text:
                    in_text = False
                    if (r - start_row) >= self.min_line_h:
                        line_bands.append((start_row, r - 1))
            if in_text and (H - start_row) >= self.min_line_h:
                line_bands.append((start_row, H - 1))

        # Extract horizontal boundaries [x, y, w, h]
        boxes = []
        for (r1, r2) in line_bands:
            h = r2 - r1 + 1
            strip = binary[r1:r2+1, :]
            col_sums = np.sum(strip > 0, axis=0)
            nonzero_cols = np.where(col_sums > 0)[0]
            if len(nonzero_cols) == 0:
                continue

            x1 = max(0, nonzero_cols[0] - 6)
            x2 = min(W - 1, nonzero_cols[-1] + 6)
            w = x2 - x1 + 1
            if w < self.min_line_w:
                continue

            boxes.append((int(x1), int(r1), int(w), int(h)))

        # Sort top to bottom
        return sorted(boxes, key=lambda b: b[1])


# =============================================================================
# 3. Segmentation-Free OCR Engine
# =============================================================================
class SegmentationFreeUrduOCR:
    def __init__(self, model_path=None, output_dir="output_segmentation_free"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.lines_dir = os.path.join(output_dir, "segmented_lines")
        os.makedirs(self.lines_dir, exist_ok=True)

        self.segmenter = LineSegmenter()
        self.model, self.vocab = self.load_model(model_path)

    def auto_find_model(self):
        """Auto detects best_model_urdu_line.pt or best_model_urdu_v2.pt"""
        candidates = [
            # 'best_model_urdu_line.pt',
            'best_model_urdu_v2.pt',
            # '../best_model_urdu_line.pt',
            # '../IEEE_Paper/best_model_urdu_line.pt'
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        for root in ['.', '..']:
            for pt in glob.glob(f"{root}/**/*.pt", recursive=True):
                return pt
        return None

    def load_model(self, model_path=None):
        if not model_path or not os.path.exists(model_path):
            model_path = self.auto_find_model()

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] Model checkpoint not found! Please provide valid path.")

        print(f"[*] Loading model from: {model_path}")
        print(f"[*] Execution device: {self.device}")

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            print(f"[*] Checkpoint keys: {list(ckpt.keys())}")
            state_dict = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
        else:
            print("[*] Checkpoint is direct state_dict (not a dict of metadata)")
            state_dict = ckpt

        # Auto-detect vocab size from fc.weight
        fc_key = next((k for k in state_dict if k.endswith('fc.weight')), None)
        vocab_size = state_dict[fc_key].shape[0] if fc_key else 74

        vocab = None
        if isinstance(ckpt, dict):
            for k in ['vocab', 'classes', 'char_list', 'chars', 'idx_to_char', 'vocabulary']:
                if k in ckpt:
                    vocab = ckpt[k]
                    print(f"[+] Found vocabulary in checkpoint under key '{k}'! Length: {len(vocab)}")
                    print(f"    First 15 chars: {vocab[:15]}")
                    break

        if vocab is None:
            print(f"[!] WARNING: No vocabulary found in checkpoint! Using fallback URDU_VOCAB_74.")
            vocab = URDU_VOCAB_74
            if len(vocab) < vocab_size:
                print(f"[!] Checkpoint fc layer expects {vocab_size} classes, but fallback has {len(vocab)}!")
                vocab += [f'<X{i}>' for i in range(vocab_size - len(vocab))]
            vocab = vocab[:vocab_size]

        model = UrduCRNN(vocab_size=vocab_size).to(self.device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[!] Missing keys when loading state_dict: {missing}")
        if unexpected:
            print(f"[!] Unexpected keys when loading state_dict: {unexpected}")
        model.eval()

        print(f"[+] Model loaded successfully! Final vocab size: {len(vocab)}")
        return model, vocab

    def preprocess_line(self, crop, target_h=64, target_w=2048):
        """
        Line crop ko H=64, W=2048 canvas par white padding ke sath scale karta hai.
        """
        ch, cw = crop.shape[:2]
        if ch < 2 or cw < 2:
            canvas = np.ones((target_h, target_w), np.uint8) * 255
        else:
            scale = target_h / float(ch)
            new_w = max(1, min(int(cw * scale), target_w))
            resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_AREA)
            canvas = np.ones((target_h, target_w), np.uint8) * 255
            canvas[:, :new_w] = resized

        # Normalize to [-1, 1]
        norm = (canvas.astype(np.float32) / 127.5) - 1.0
        return torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def process_image(self, image_path, first_line_only=False):
        """
        End-to-End Segmentation-Free Line OCR
        """
        if not os.path.exists(image_path):
            print(f"[!] File not found: {image_path}")
            return None

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            print(f"[!] Could not read image: {image_path}")
            return None

        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        print(f"\n{'='*70}")
        print(f"Processing Page: {os.path.basename(image_path)} ({W}x{H} px)")
        print(f"Mode: Segmentation-Free Line OCR (CRNN + CTC)")
        print(f"{'='*70}")

        # Step 1: Detect lines automatically
        print(f"[*] Step 1: Detecting text lines (Automatic Horizontal Profile)...")
        boxes = self.segmenter.segment(gray)
        print(f"[+] Found {len(boxes)} line(s) on page.")

        if not boxes:
            print("[!] No lines detected. Check image quality.")
            return None

        lines_to_process = [boxes[0]] if first_line_only else boxes
        annotated = original_bgr.copy()
        results = []

        print(f"\n[*] Step 2: Running Segmentation-Free Recognition (ResNet34 + BiLSTM + CTC)...")
        print("-" * 70)

        with torch.no_grad():
            for idx, (x, y, w, h) in enumerate(lines_to_process, 1):
                # Padding around line
                pad_y = max(3, int(h * 0.08))
                pad_x = max(4, int(w * 0.02))
                crop = gray[max(0, y - pad_y):min(H, y + h + pad_y),
                            max(0, x - pad_x):min(W, x + w + pad_x)]

                if crop.shape[0] < 2 or crop.shape[1] < 2:
                    continue

                # Save cropped line image
                crop_path = os.path.join(self.lines_dir, f"{base_name}_line_{idx:02d}.png")
                cv2.imwrite(crop_path, crop)

                # Feed directly to CRNN
                tensor = self.preprocess_line(crop).to(self.device)
                logits = self.model(tensor)
                line_text = ctc_decode(logits, self.vocab)

                results.append({
                    "line_number": idx,
                    "box": {"x": x, "y": y, "w": w, "h": h},
                    "text": line_text
                })

                # Visual annotation
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(annotated, f"L{idx}", (max(5, x - 30), y + h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                print(f"  Line {idx:02d} [y={y}:{y+h}]: {line_text}")

        print("-" * 70)

        # Step 3: Save recognized text file
        text_file = os.path.join(self.output_dir, f"{base_name}_recognized_text.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(f"Line {item['line_number']:02d}: {item['text']}\n")
        print(f"[+] Transcribed text saved → {text_file}")

        # Step 4: Save annotated image
        annotated_file = os.path.join(self.output_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_file, annotated)
        print(f"[+] Annotated page saved  → {annotated_file}")

        # Step 5: Save JSON results
        json_file = os.path.join(self.output_dir, f"{base_name}_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "image": os.path.basename(image_path),
                "total_lines_detected": len(boxes),
                "processed_lines": len(results),
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"[+] JSON results saved    → {json_file}")

        return results


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    print("=" * 70)
    print("URDU NASTALEEQ SEGMENTATION-FREE LINE OCR")
    print("CRNN (ResNet34 + BiLSTM) + CTC Loss")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\n📝 Usage:")
        print("   python line_segmentation_ocr.py <image_path> [first_line_only]")
        print("\n📌 Example:")
        print('   python line_segmentation_ocr.py demo.jpeg')
        print('   python line_segmentation_ocr.py demo.jpeg True')
        return

    image_path = sys.argv[1]
    first_line_only = False

    if len(sys.argv) > 2:
        first_line_only = sys.argv[2].lower() in ['true', '1', 'yes']

    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    ocr_system = SegmentationFreeUrduOCR(model_path=model_path)
    ocr_system.process_image(image_path, first_line_only=first_line_only)


if __name__ == "__main__":
    main()

import sys, os, requests, torch
from io import BytesIO
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

if len(sys.argv) < 2:
    print("Usage: python run_age_manual.py <image_path_or_url>")
    sys.exit(1)

src = sys.argv[1]

# Load image
if src.startswith("http://") or src.startswith("https://"):
    im = Image.open(BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
else:
    if not os.path.exists(src):
        raise FileNotFoundError(f"No such file: {src}")
    im = Image.open(src).convert("RGB")

model_id = "nateraw/vit-age-classifier"
processor = AutoImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(model_id)

inputs = processor(im, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs)

proba = output.logits.softmax(1)[0]
id2label = model.config.id2label
topk = torch.topk(proba, k=min(5, proba.numel()))
print("\nTop predictions:")
for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
    print(f"{id2label[idx]:<15}  prob={score:.4f}")

best_idx = int(proba.argmax())
print(f"\nTop-1: {id2label[best_idx]}  (p={float(proba[best_idx]):.4f})")


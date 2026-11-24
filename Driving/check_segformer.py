"""
check_segformer.py -- is SegFormer loading its full weights in THIS environment?

Why: in the CARLA env, from_pretrained left 4 decode-head layers randomly
initialised (decode_head.linear_c.*), because the checkpoint was saved with
transformers >= 5.9, which renamed them to linear_projections. Segmentation
from such a model is noise, and so is every point painted with it.

Run it in EVERY environment that loads this SegFormer: the CARLA env (live
fusion) and the env Fusion/paint_points.py ran in (training paint).

  python check_segformer.py --seg-model ..\\Detection_YOLO\\runs\\seg\\best --images "..\\path\\to\\some_rgb_frames\\*.png"

Prints
  1. transformers version, and what the stock from_pretrained left missing
  2. per image: class pixel shares for stock vs fixed loading, and how many
     pixels the two agree on
Saves
  seg_check_<image>.png : image | stock loading | fixed loading
Reading it
  "stock loader: complete" in an env  -> that env was never affected
  "stock loader: N tensors random"    -> anything run there with plain
                                          from_pretrained used broken segmentation
  The fixed column should look like a sensible road scene.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from perception_scheduler import IMAGENET_MEAN, IMAGENET_STD, load_segformer


def palette(n):
    rng = np.random.default_rng(3)
    p = rng.integers(40, 255, size=(max(n, 1), 3)).astype(np.uint8)
    p[0] = (128, 64, 128)
    return p


def predict(model, rgb, hw):
    h, w = hw
    x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    with torch.no_grad():
        logits = model(pixel_values=torch.from_numpy(x.transpose(2, 0, 1))[None]).logits
    lab = logits.argmax(1)[0].numpy().astype(np.uint8)
    return cv2.resize(lab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)


def shares(lab, names):
    n = np.bincount(lab.ravel(), minlength=len(names)) / lab.size
    top = np.argsort(-n)[:5]
    return ', '.join(f"{names[i]} {100 * n[i]:.0f}%" for i in top if n[i] > 0.005)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seg-model', required=True)
    ap.add_argument('--images', nargs='*', default=[], help='paths or globs of RGB frames (png/jpg)')
    ap.add_argument('--max-images', type=int, default=4)
    ap.add_argument('--seg-img-size', type=int, nargs=2, default=[384, 768], metavar=('H', 'W'))
    ap.add_argument('--out-dir', default='seg_check')
    a = ap.parse_args()

    import transformers
    from transformers import SegformerForSemanticSegmentation
    from transformers.utils import logging as hf_logging
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")

    torch.manual_seed(0)
    hf_logging.set_verbosity_error()
    stock, info = SegformerForSemanticSegmentation.from_pretrained(a.seg_model, output_loading_info=True)
    stock.eval()
    missing = [k for k in info.get('missing_keys', []) if not k.endswith('num_batches_tracked')]
    if missing:
        print(f"stock loader: {len(missing)} tensors random (missing from the checkpoint): {sorted(missing)}")
    else:
        print("stock loader: complete, nothing random")
    fixed, rep = load_segformer(a.seg_model)
    names = [fixed.config.id2label.get(i, str(i)) for i in range(fixed.config.num_labels)]
    print(f"classes ({len(names)}): {names}")

    files = []
    for pat in a.images:
        files += sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat]
    files = [f for f in files if os.path.isfile(f)][:a.max_images]
    if not files:
        print("\nno --images given or found: loading check only")
        return
    os.makedirs(a.out_dir, exist_ok=True)
    pal = palette(len(names))
    for f in files:
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  cannot read {f}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ls, lf = predict(stock, rgb, a.seg_img_size), predict(fixed, rgb, a.seg_img_size)
        agree = float((ls == lf).mean())
        print(f"\n{os.path.basename(f)}   stock vs fixed agree on {100 * agree:.0f}% of pixels")
        print(f"  stock: {shares(ls, names)}")
        print(f"  fixed: {shares(lf, names)}")
        panels = [bgr] + [cv2.addWeighted(bgr, 0.35, cv2.cvtColor(pal[l], cv2.COLOR_RGB2BGR), 0.65, 0)
                          for l in (ls, lf)]
        for img, title in zip(panels, ('image', 'stock from_pretrained', 'fixed loader')):
            cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out = os.path.join(a.out_dir, f"seg_check_{os.path.splitext(os.path.basename(f))[0]}.png")
        cv2.imwrite(out, np.hstack(panels))
        print(f"  saved {out}")


if __name__ == '__main__':
    main()

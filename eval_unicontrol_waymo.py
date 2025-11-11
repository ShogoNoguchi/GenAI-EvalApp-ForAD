#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#コード内に実行コマンドや/home/shogo/coding/eval/ucn_eval/docker/Dockerfileと/home/shogo/coding/eval/ucn_eval/docker/entrypoint.shの内容も記しておくことでわかりやすく（重要）。
"""/home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py
Uni-ControlNet を “自動運転用画像水増し装置” として網羅的に定量評価する単一スクリプト（完全版）
- 入力: 元RGB(X) と 生成画像 F(X)（Uni-ControlNet_offline 産物）を WaymoV2 の相対パスでペアリング
- 評価（--tasks で切替）:
  (1) リアリティ指標:
      - CLIP-CMMD（推奨; CLIP embedding + RBF-MMD）
      - CLIP-FID（CLIP embedding での Fréchet 距離）
      - Inception-FID（警告表示のみ; 実装外）
  (2) 構造・意味忠実度:
      - Edge(Canny): L1/RMSE/IoU/F1（X の既存 Canny と F(X) の Canny 再推論を比較）
      - Depth(Metric3Dv2, ONNX): RMSE / scale-invariant RMSE / 相対誤差（X: 既存 NPY, F(X): ONNX 再推論）
      - Semseg(OneFormer Cityscapes): 19クラス混同行列→mIoU
      - すべて /mnt/hdd/ucn_eval_cache にキャッシュ（npz/npy/png）して再利用
  (3) 物体保持・ハルシネーション:
      - Grounding DINO（オープン語彙検出; Transformers 実装）
      - （任意）YOLOP（drivable ROI によるフィルタは既定OFF・必要時のみ --roi-filter）
      - OCR(Tesseract) による標識読み取り
      - 指標: 保持再現率(PR) / 保持適合率(PP) / F1 / ハルシネーション率(HAL)
              幾何安定性（中心誤差/サイズ比誤差/IoU中央値）
              カウント整合性（平均差 / 擬EMD）
  (4) 走行可能領域の保持（新規: --tasks drivable または all）
      - YOLOP の drivable マスク（X vs F(X)）を直接比較
      - 指標: IoU / Dice(F1) / Precision / Recall / L1 / RMSE / Boundary-IoU（境界的一致）
      - 後処理: 2値化閾値 (--drivable-thr), モルフォロジー閉処理 (--drivable-morph-k), 境界許容 (--drivable-edge-tol)

- アノテーション可視化（--annotation-mode）:
  objects: X/FX の検出ボックス（+drivable ROI 任意）を描画
  structure: FX の Edge/Depth/Semseg を横並び保存
  drivable: X/FX の走行可能領域オーバレイを並列保存（新規）
  all: 両方
  off: なし
- 既定パスは翔伍さんの資産に合わせて固定。CLI で上書き可。
- 再現性: 乱数固定（--seed）、TensorBoard（--tb）対応。

注意:
- Docker 側で torch==2.7.0+cu128 / torchvision==0.22.0+cu128 を固定。
- 追加依存（timm, thop, prefetch_generator, scikit-image）は overlay で導入。
- onnxruntime は CPU/GPU どちらでも動作可（CUDAExecutionProvider があれば GPU 使用）。

実行例：
## All（Reality＋Structure＋Objects＋Drivable）一括評価＋可視化（推奨・論文用）
- **OneFormer(Swin) 用に `timm` を overlay 導入**（事前に「0) リセット」実行推奨）
- アノテ：`all` を各split 24枚保存
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e PIP_OVERLAY_DIR=/mnt/hdd/ucn_eval_cache/pip-overlay \
  -e REQS_OVERLAY_PATH=/mnt/hdd/ucn_eval_cache/requirements.overlay.txt \
  -e 'PIP_INSTALL=timm yacs prefetch_generator pytesseract huggingface_hub>=0.34,<1.0 einops matplotlib' \
  -v /home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh:/app/entrypoint.sh:ro \
  -v /home/shogo/coding/eval/ucn_eval/eval_unicontrol_waymo.py:/app/eval_unicontrol_waymo.py:ro \
  -v /home/shogo/coding/datasets/WaymoV2:/home/shogo/coding/datasets/WaymoV2:ro \
  -v /home/shogo/coding/Metric3D:/home/shogo/coding/Metric3D:ro \
  -v /mnt/hdd/ucn_eval_cache:/mnt/hdd/ucn_eval_cache \
  -v /home/shogo/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/hdd/ucn_eval_cache/torch_hub:/root/.cache/torch/hub \
  ucn-eval \
  --splits training validation testing \
  --camera front \
  --tasks all \
  --reality-metric clip-cmmd \
  --gdinomodel IDEA-Research/grounding-dino-base \
  --det-prompts car truck bus motorcycle bicycle person pedestrian "traffic light" "traffic sign" "stop sign" "speed limit sign" "crosswalk sign" "construction sign" "traffic cone" \
  --gdino-box-thr 0.25 \
  --gdino-text-thr 0.20 \
  --use-yolop \
  --yolop-roi-filter \
  --ocr-engine tesseract \
  --iou-thr 0.5 \
  --orig-root /home/shogo/coding/datasets/WaymoV2/extracted \
  --gen-root  /home/shogo/coding/datasets/WaymoV2/UniControlNet_offline \
  --annotation-mode all \
  --annotate-limit 24 \
  --tb --tb-dir /mnt/hdd/ucn_eval_cache/tensorboard \
  --verbose

"""

import os
import sys
import argparse
import json
import logging
from logging import handlers
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import hashlib
import time
import math
import traceback
import shutil
import multiprocessing as mp
import warnings

# YOLOP/Transformers の将来変更に伴う FutureWarning を既定で抑止
warnings.filterwarnings(
    "ignore",
    message=".*GroundingDinoProcessor.*Use `text_labels` instead.*",
    category=FutureWarning
)

# 数値・画像
import numpy as np
import cv2

# 科学計算
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

# 進捗
from tqdm import tqdm

# Torch/Transformers
import torch
from torch.utils.tensorboard import SummaryWriter

# HF models（必要時に遅延ロード）
from transformers import AutoProcessor, OneFormerForUniversalSegmentation, CLIPModel, CLIPImageProcessor
from transformers import AutoProcessor as GDinoProcessor
from transformers import AutoModelForZeroShotObjectDetection

# ========== 既定パス（環境に完全準拠） ==========
DEFAULT_ORIG_IMAGE_ROOT = "/home/shogo/coding/datasets/WaymoV2/extracted"
DEFAULT_GEN_ROOT        = "/home/shogo/coding/datasets/WaymoV2/UniControlNet_offline"

# 既存の X 側予測（再利用）
DEFAULT_CANNY_ROOT_X     = "/home/shogo/coding/datasets/WaymoV2/CannyEdge"
DEFAULT_DEPTH_NPY_ROOT_X = "/home/shogo/coding/datasets/WaymoV2/Metricv2DepthNPY"
DEFAULT_SEMSEG_ROOT_X    = "/home/shogo/coding/datasets/WaymoV2/OneFormer_cityscapes"

# F(X) 側の推論キャッシュ（HDD）
DEFAULT_HDD_CACHE_ROOT   = "/mnt/hdd/ucn_eval_cache"

# Metric3Dv2 ONNX
DEFAULT_METRIC3D_ONNX    = "/home/shogo/coding/Metric3D/onnx/onnx/model.onnx"  # 既存

# OneFormer / Grounding DINO / CLIP
DEFAULT_ONEFORMER_ID     = "shi-labs/oneformer_cityscapes_swin_large"
DEFAULT_GDINO_ID         = "IDEA-Research/grounding-dino-base"
DEFAULT_CLIP_ID          = "openai/clip-vit-large-patch14-336"

# 評価対象クラス（OVD: Grounding DINO）
DEFAULT_DET_PROMPTS = [
    "car", "truck", "bus", "motorcycle", "bicycle", "person", "pedestrian",
    "traffic light", "traffic sign", "stop sign", "speed limit sign",
    "crosswalk sign", "construction sign", "traffic cone",
]

# Cityscapes trainId（0..18）— OneFormer と整合
CITYSCAPES_TRAINID = list(range(19))

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# OneFormer/深度の既定入力サイズ等
IN_H, IN_W = 512, 1088  # Metric3D 推奨（既存の資産に整合）


# ========== ユーティリティ（ロガー/環境） ==========
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def setup_logger(log_dir: str, verbose: bool) -> logging.Logger:
    """
    - 外部で注入されたハンドラ/フォーマッタ（%(timestamp)s など）を完全排除
    - 自前の Stream & RotatingFile ハンドラのみ使用
    """
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "eval.log")

    logger = logging.getLogger("ucn_eval")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    # 外部で付いたハンドラを完全クリア（%(timestamp)s を要求する異種フォーマッタ対策）
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.handlers.clear()

    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))

    fh = handlers.RotatingFileHandler(log_path, maxBytes=20*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style='%'))

    logger.addHandler(ch)
    logger.addHandler(fh)

    # 例外時にロギングが止まらないように
    logging.raiseExceptions = False
    return logger



def log_env(logger: logging.Logger) -> None:
    try:
        logger.info("torch: %s | build_cuda: %s | cuda_available: %s",
                    torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
            # CUDA 12.8 確認（警告のみ）
            if getattr(torch.version, "cuda", "") and not str(torch.version.cuda).startswith("12.8"):
                logger.warning("警告: この PyTorch ビルドの CUDA 表示は %s です（12.8 以外）。既存環境に合わせて続行します。",
                               getattr(torch.version, "cuda", None))
    except Exception:
        pass


# ========== データ列挙 / ペアリング ==========
def _ext_lower(p: str) -> str:
    return os.path.splitext(p)[1].lower()

def list_images(root: str, split: str, camera: str) -> List[str]:
    base = os.path.join(root, split, camera)
    out: List[str] = []
    if not os.path.isdir(base):
        return out
    for r, _, fs in os.walk(base):
        for f in fs:
            if _ext_lower(f) in ALLOWED_IMG_EXT:
                out.append(os.path.join(r, f))
    out.sort()
    return out

def rel_dir_and_stem(image_path: str, split_root: str) -> Tuple[str, str]:
    # rel_dir: front/{segment} / stem: e.g., 1510593619939663_first
    rel_dir = os.path.relpath(os.path.dirname(image_path), split_root)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return rel_dir, stem

def path_ucn_png(gen_root: str, split: str, rel_dir: str, stem: str) -> str:
    # 生成結果は {stem}_ucn.png（num_samples>1 は _ucn-000.png などに対応）
    p = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn.png")
    if os.path.exists(p):
        return p
    cand = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn-000.png")
    return cand

def enumerate_pairs(orig_root: str, gen_root: str, split: str, camera: str, limit: int) -> List[Tuple[str, str, str, str]]:
    """
    戻り値: List[(image_path_X, image_path_FX, rel_dir, stem)]
    """
    items = list_images(orig_root, split, camera)
    if limit > 0:
        items = items[:limit]
    pairs: List[Tuple[str, str, str, str]] = []
    for p in items:
        rel_dir, stem = rel_dir_and_stem(p, os.path.join(orig_root, split))
        fx = path_ucn_png(gen_root, split, rel_dir, stem)
        if os.path.exists(fx):
            pairs.append((p, fx, rel_dir, stem))
    return pairs


# ========== HDD キャッシュ管理 ==========
class Cache:
    """
    /mnt/hdd/ucn_eval_cache 配下へのキャッシュ管理
    - clip/{split}/{rel_dir}/{stem}_{x|fx}.npz
    - canny_fx/{...}.png
    - depth_fx/{...}.npy
    - semseg_fx/{...}.npy
    - yolo_{x,fx}/{...}.json
    - gdino_{x,fx}/{...}.json
    - ocr_{x,fx}/{...}.json
    """
    def __init__(self, root: str):
        self.root = root
        self.d_clip   = os.path.join(root, "clip")
        self.d_canny  = os.path.join(root, "canny_fx")
        self.d_depth  = os.path.join(root, "depth_fx")
        self.d_semseg = os.path.join(root, "semseg_fx")
        self.d_yolo_x = os.path.join(root, "yolop_x")
        self.d_yolo_f = os.path.join(root, "yolop_fx")
        self.d_gd_x   = os.path.join(root, "gdino_x")
        self.d_gd_f   = os.path.join(root, "gdino_fx")
        self.d_ocr_x  = os.path.join(root, "ocr_x")
        self.d_ocr_f  = os.path.join(root, "ocr_fx")
        self.d_logs   = os.path.join(root, "logs")
        for d in [self.d_clip, self.d_canny, self.d_depth, self.d_semseg,
                  self.d_yolo_x, self.d_yolo_f, self.d_gd_x, self.d_gd_f,
                  self.d_ocr_x, self.d_ocr_f, self.d_logs]:
            ensure_dir(d)

    def clip_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        # which in {"x","fx"}
        return os.path.join(self.d_clip, split, rel_dir, f"{stem}_{which}.npz")

    def canny_fx_path(self, split: str, rel_dir: str, stem: str) -> str:
        return os.path.join(self.d_canny, split, rel_dir, f"{stem}_edge.png")

    def depth_fx_path(self, split: str, rel_dir: str, stem: str) -> str:
        return os.path.join(self.d_depth, split, rel_dir, f"{stem}_depth.npy")

    def semseg_fx_path(self, split: str, rel_dir: str, stem: str) -> str:
        return os.path.join(self.d_semseg, split, rel_dir, f"{stem}_predTrainId.npy")

    def yolo_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        d = self.d_yolo_x if which == "x" else self.d_yolo_f
        return os.path.join(d, split, rel_dir, f"{stem}_yolop.json")

    def gdino_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        d = self.d_gd_x if which == "x" else self.d_gd_f
        return os.path.join(d, split, rel_dir, f"{stem}_gdino.json")

    def ocr_path(self, split: str, rel_dir: str, stem: str, which: str) -> str:
        d = self.d_ocr_x if which == "x" else self.d_ocr_f
        return os.path.join(d, split, rel_dir, f"{stem}_ocr.json")


# ========== 画像 I/O と注釈 ==========
def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imread_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Failed to read image(gray): {path}")
    return g

def save_indexed_png(path: str, arr_u8: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, arr_u8)

# ---- 注釈描画ユーティリティ ----
_CITYSCAPES_TRAINID_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]  # BGR(OpenCV)

def colorize_trainId(trainId: np.ndarray) -> np.ndarray:
    h, w = trainId.shape[:2]
    out = np.zeros((h,w,3), dtype=np.uint8)
    for idx, (b,g,r) in enumerate(_CITYSCAPES_TRAINID_COLORS):
        if idx >= 19: break
        out[trainId==idx] = (b,g,r)
    return out

def draw_boxes(rgb: np.ndarray, dets: List[Dict[str,Any]], color=(0,255,0)) -> np.ndarray:
    img = rgb.copy()
    for d in dets:
        x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
        cls = d.get("cls","?")
        score = d.get("score",0.0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{cls}:{score:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float=0.4, color=(0,255,0)) -> np.ndarray:
    col = np.zeros_like(rgb); col[mask>0] = color
    out = cv2.addWeighted(rgb, 1.0, col, alpha, 0.0)
    return out

def save_annotations_objects(outdir: str, split: str, rel_dir: str, stem: str,
                             rgbx: np.ndarray, rgbf: np.ndarray,
                             G: List[Dict[str,Any]], D: List[Dict[str,Any]],
                             drv_x: Optional[np.ndarray]=None, drv_f: Optional[np.ndarray]=None) -> None:
    ensure_dir(os.path.join(outdir, split, rel_dir))
    x_vis = draw_boxes(rgbx, G, color=(0,255,0))
    f_vis = draw_boxes(rgbf, D, color=(0,0,255))
    if drv_x is not None:
        x_vis = overlay_mask(x_vis, drv_x>0, alpha=0.3, color=(0,255,255))
    if drv_f is not None:
        f_vis = overlay_mask(f_vis, drv_f>0, alpha=0.3, color=(255,255,0))
    cv2.imwrite(os.path.join(outdir, split, rel_dir, f"{stem}_x_objs.jpg"), cv2.cvtColor(x_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(outdir, split, rel_dir, f"{stem}_fx_objs.jpg"), cv2.cvtColor(f_vis, cv2.COLOR_RGB2BGR))

def save_annotations_structure(outdir: str, split: str, rel_dir: str, stem: str,
                               rgbf: np.ndarray, edge_fx: np.ndarray, depth_fx: np.ndarray, seg_fx: np.ndarray) -> None:
    ensure_dir(os.path.join(outdir, split, rel_dir))
    # Edge 可視化
    e3 = cv2.cvtColor((edge_fx>0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    # Depth は min-max で可視化
    d = depth_fx.copy()
    d -= np.min(d); d /= (np.max(d)+1e-9); d = (d*255).clip(0,255).astype(np.uint8)
    d3 = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    # Semseg 可視化
    s3 = colorize_trainId(seg_fx)
    # 横並びで保存（RGBも左端に付与）
    cat = np.concatenate([rgbf, e3, d3, s3], axis=1)
    cv2.imwrite(os.path.join(outdir, split, rel_dir, f"{stem}_fx_struct.jpg"), cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))

def save_annotations_drivable(outdir: str, split: str, rel_dir: str, stem: str,
                              rgbx: np.ndarray, rgbf: np.ndarray,
                              drv_x: np.ndarray, drv_f: np.ndarray) -> None:
    ensure_dir(os.path.join(outdir, split, rel_dir))
    x_vis = overlay_mask(rgbx, drv_x>0, alpha=0.35, color=(0,255,255))   # シアン系
    f_vis = overlay_mask(rgbf, drv_f>0, alpha=0.35, color=(255,255,0))   # 黄系
    cat = np.concatenate([x_vis, f_vis], axis=1)
    cv2.imwrite(os.path.join(outdir, split, rel_dir, f"{stem}_drivable.jpg"), cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))


# ========== (1) リアリティ指標：CLIP特徴, FID/CMMD ==========
class ClipEmbedder:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPImageProcessor] = None
        self.max_batch: Optional[int] = None

    def load(self, logger: logging.Logger):
        if self.model is None:
            logger.info("[CLIP] loading model: %s", self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self.processor = CLIPImageProcessor.from_pretrained(self.model_id)

    def autotune(self, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 1024) -> int:
        self.max_batch = autotune_clip_bs(self, logger, sample_rgb, cap=cap)
        return self.max_batch

    @torch.inference_mode()
    def embed_batch(self, images: List[np.ndarray]) -> np.ndarray:
        assert self.model is not None and self.processor is not None
        inputs = self.processor(images=images, return_tensors="pt")
        pix = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(pix)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.detach().cpu().numpy().astype(np.float32)


def feats_to_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    CLIP-FID 用: 平均ベクトルと共分散行列を返す
    feats: (N, D)
    """
    feats = np.asarray(feats, dtype=np.float64)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """FID 距離（Fréchet distance）"""
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

def gaussian_mmd(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    CLIP 埋め込みに対する MMD^2（RBF）
    - sigma が None のときは median heuristic（自己距離の 0 を除外）
    - 計算は特徴の内積からの二乗距離で行い、gamma=1/(2*sigma^2)
    """
    def pdist_sq(a: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)          # (N,1)
        return aa + aa.T - 2.0 * (a @ a.T)                 # (N,N)

    def cdist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)          # (N,1)
        bb = np.sum(b * b, axis=1, keepdims=True)          # (M,1)
        return aa + bb.T - 2.0 * (a @ b.T)                 # (N,M)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xx = pdist_sq(x)
    yy = pdist_sq(y)
    xy = cdist_sq(x, y)

    if sigma is None:
        n = xx.shape[0]; m = yy.shape[0]
        # 自己距離を除外
        xx_off = xx[~np.eye(n, dtype=bool)]
        yy_off = yy[~np.eye(m, dtype=bool)]
        pool = np.concatenate([xx_off, yy_off, xy.ravel()])
        # 数値安定性のための下限
        med = float(np.median(pool)) if pool.size > 0 else 1.0
        med = max(med, 1e-12)
        sigma = math.sqrt(0.5 * med)
    sigma = max(float(sigma), 1e-9)
    gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = np.exp(-gamma * xx)
    k_yy = np.exp(-gamma * yy)
    k_xy = np.exp(-gamma * xy)
    m = x.shape[0]; n = y.shape[0]
    mmd2 = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1) + 1e-9) \
         + (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1) + 1e-9) \
         - 2.0 * np.sum(k_xy) / (m * n + 1e-9)
    return float(mmd2)



# ========== (2) 構造・意味忠実度 ==========
# --- Edge(Canny) ---
def canny_cpu(rgb: np.ndarray, t1: float = 100.0, t2: float = 200.0, blur_ksize: int = 3) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur_ksize > 0:
        g = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 0)
    e = cv2.Canny(g, t1, t2)
    return e

def edge_metrics(edge_x: np.ndarray, edge_fx: np.ndarray) -> Dict[str, float]:
    """
    2値エッジの一致度（L1/RMSE/IoU/F1）。
    画像サイズが異なる場合は edge_x を edge_fx サイズに最近傍でリサイズしてから比較する。
    """
    if edge_x.shape != edge_fx.shape:
        edge_x = cv2.resize(edge_x, (edge_fx.shape[1], edge_fx.shape[0]), interpolation=cv2.INTER_NEAREST)

    a = (edge_x > 0).astype(np.uint8)
    b = (edge_fx > 0).astype(np.uint8)

    l1 = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    rmse = float(np.sqrt(np.mean((a - b)**2)))

    inter = float(np.sum((a & b) > 0))
    union = float(np.sum((a | b) > 0) + 1e-9)
    iou = inter / union

    tp = inter
    fp = float(np.sum((b > 0) & (a == 0)))
    fn = float(np.sum((a > 0) & (b == 0)))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1 = 2*prec*rec / (prec + rec + 1e-9)
    return {"edge_l1": l1, "edge_rmse": rmse, "edge_iou": float(iou), "edge_f1": float(f1)}

# --- Depth(Metric3Dv2; si-RMSE 等) ---
import onnxruntime as ort

def build_metric3d_session(onnx_path: str) -> Tuple[ort.InferenceSession, str, str, List[str]]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name, providers

def infer_metric3d_np(session: ort.InferenceSession, in_name: str, out_name: str, rgb: np.ndarray) -> np.ndarray:
    h0, w0 = rgb.shape[:2]
    rgb_resized = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    t = np.transpose(rgb_resized, (2,0,1))[None, ...]  # (1,3,H,W)
    y = session.run([out_name], {in_name: t})[0]
    depth = np.squeeze(y).astype(np.float32)
    depth_up = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return depth_up

def depth_metrics(depth_x: np.ndarray, depth_fx: np.ndarray) -> Dict[str, float]:
    """
    Depth の RMSE / scale-invariant RMSE / 相対誤差。サイズが異なる場合は depth_x を depth_fx サイズにリサイズして比較。
    """
    if depth_x.shape != depth_fx.shape:
        depth_x = cv2.resize(depth_x, (depth_fx.shape[1], depth_fx.shape[0]), interpolation=cv2.INTER_LINEAR)

    eps = 1e-6
    dx = np.maximum(depth_x, eps); df = np.maximum(depth_fx, eps)
    rmse = float(np.sqrt(np.mean((dx - df)**2)))
    d = np.log(dx) - np.log(df)
    si_rmse = float(np.sqrt(np.mean(d**2) - (np.mean(d)**2)))
    rel = np.abs(dx - df) / (dx + eps)
    rel_mean = float(np.mean(rel))
    return {"depth_rmse": rmse, "depth_si_rmse": si_rmse, "depth_rel": rel_mean}

# --- Semseg(OneFormer Cityscapes) ---
def build_oneformer(model_id: str, device: str = "cuda", fp16: bool = True):
    """
    OneFormer (Cityscapes) のロード
    - まず torch_dtype= を試し、TypeError の場合は dtype= にフォールバック
    - GPU & 半精度は環境に応じて自動
    """
    processor = AutoProcessor.from_pretrained(model_id)
    use_cuda = (device.startswith("cuda") and torch.cuda.is_available())
    dtype = torch.float16 if (fp16 and use_cuda) else torch.float32

    try:
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id, torch_dtype=dtype).eval()
    except TypeError:
        # 古い Transformers などへの後方互換
        model = OneFormerForUniversalSegmentation.from_pretrained(model_id, dtype=dtype).eval()

    if use_cuda:
        model = model.to("cuda")
    return processor, model



def oneformer_semseg(processor, model, rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    enc = processor(images=rgb, task_inputs=["semantic"], return_tensors="pt")
    pv = enc.get("pixel_values")
    if isinstance(pv, torch.Tensor):
        if pv.ndim == 3: pv = pv.unsqueeze(0)
    else:
        pv = torch.from_numpy(np.array(pv))
    pv = pv.to(model.device, dtype=model.dtype)
    ti = enc.get("task_inputs")
    if isinstance(ti, torch.Tensor): ti = ti.to(model.device)
    with torch.inference_mode():
        if (model.dtype == torch.float16) and (model.device.type == "cuda"):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(pixel_values=pv, task_inputs=ti)
        else:
            out = model(pixel_values=pv, task_inputs=ti)
    seg = processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
    return seg.to("cpu").numpy().astype(np.uint8)

def confusion_19(gt: np.ndarray, pr: np.ndarray, ncls: int = 19) -> np.ndarray:
    mask = (gt >= 0) & (gt < ncls)
    hist = np.bincount(ncls * gt[mask].astype(int) + pr[mask].astype(int), minlength=ncls**2).reshape(ncls, ncls)
    return hist

def miou_from_conf(hist: np.ndarray) -> Tuple[float, np.ndarray]:
    # IoU_c = TP / (TP + FP + FN)
    tp = np.diag(hist).astype(np.float64)
    fp = np.sum(hist, axis=0) - tp
    fn = np.sum(hist, axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    miou = float(np.nanmean(iou))
    return miou, iou

# ======== Auto-Batch Utilities: 各モデルの最大バッチを OOM を避けて自動探索 ========

def _chunks(lst, n):
    for i in range(0, len(lst), max(1, n)):
        yield lst[i:i+n]

def _is_cuda_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return ("cuda out of memory" in msg) or ("cudnn" in msg and "error" in msg) or ("allocator" in msg and "memory" in msg)

def _torch_sync_empty_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()

# ---------- CLIP: 自動バッチ計測 ----------
def autotune_clip_bs(clipper, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 1024) -> int:
    """
    CLIP の get_image_features で使える最大バッチを探索。
    - 二分探索（指数増加→失敗→範囲内二分）
    """
    if not torch.cuda.is_available() or clipper.device == "cpu":
        logger.info("[AutoBatch][CLIP] CPU 推論のため batch=1 固定")
        return 1
    clipper.load(logger)
    best, b, failed = 1, 1, False
    while b <= cap:
        try:
            imgs = [sample_rgb] * b
            enc = clipper.processor(images=imgs, return_tensors="pt")
            pix = enc["pixel_values"].to(clipper.device)
            with torch.inference_mode():
                _ = clipper.model.get_image_features(pix)
            del enc, pix, _
            _torch_sync_empty_cache()
            best = b
            b *= 2
        except RuntimeError as e:
            if _is_cuda_oom(e):
                failed = True
                _torch_sync_empty_cache()
                break
            raise
    lo = best
    hi = min(cap, b-1) if failed else best
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            imgs = [sample_rgb] * mid
            enc = clipper.processor(images=imgs, return_tensors="pt")
            pix = enc["pixel_values"].to(clipper.device)
            with torch.inference_mode():
                _ = clipper.model.get_image_features(pix)
            del enc, pix, _
            _torch_sync_empty_cache()
            lo = mid
        except RuntimeError as e:
            if _is_cuda_oom(e):
                _torch_sync_empty_cache()
                hi = mid - 1
            else:
                raise
    logger.info("[AutoBatch][CLIP] batch=%d", lo)
    return max(1, int(lo))

# ---------- OneFormer: 一括推論 + 自動バッチ ----------
def oneformer_semseg_batch(processor, model, rgbs: List[np.ndarray]) -> List[np.ndarray]:
    sizes = [(im.shape[0], im.shape[1]) for im in rgbs]
    enc = processor(images=rgbs, task_inputs=["semantic"]*len(rgbs), return_tensors="pt")
    pv = enc.get("pixel_values")
    if isinstance(pv, torch.Tensor):
        if pv.ndim == 3:
            pv = pv.unsqueeze(0)
    else:
        pv = torch.from_numpy(np.array(pv))
    pv = pv.to(model.device, dtype=model.dtype)
    ti = enc.get("task_inputs")
    if isinstance(ti, torch.Tensor):
        ti = ti.to(model.device)
    with torch.inference_mode():
        if (model.dtype == torch.float16) and (model.device.type == "cuda"):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(pixel_values=pv, task_inputs=ti)
        else:
            out = model(pixel_values=pv, task_inputs=ti)
    segs = processor.post_process_semantic_segmentation(out, target_sizes=sizes)
    out_list = [s.to("cpu").numpy().astype(np.uint8) for s in segs]
    return out_list

def autotune_oneformer_bs(processor, model, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 64) -> int:
    if not (torch.cuda.is_available() and model.device.type == "cuda"):
        logger.info("[AutoBatch][OneFormer] CPU/非CUDA のため batch=1 固定")
        return 1
    best, b, failed = 1, 1, False
    while b <= cap:
        try:
            _ = oneformer_semseg_batch(processor, model, [sample_rgb]*b)
            del _
            _torch_sync_empty_cache()
            best = b
            b *= 2
        except RuntimeError as e:
            if _is_cuda_oom(e):
                failed = True
                _torch_sync_empty_cache()
                break
            raise
    lo = best
    hi = min(cap, b-1) if failed else best
    while lo < hi:
        mid = (lo + hi + 1)//2
        try:
            _ = oneformer_semseg_batch(processor, model, [sample_rgb]*mid)
            del _
            _torch_sync_empty_cache()
            lo = mid
        except RuntimeError as e:
            if _is_cuda_oom(e):
                _torch_sync_empty_cache()
                hi = mid - 1
            else:
                raise
    logger.info("[AutoBatch][OneFormer] batch=%d", lo)
    return max(1, int(lo))

# ---------- YOLOP: 一括推論 + 自動バッチ ----------
@torch.inference_mode()
def yolop_infer_batch(model, rgbs: List[np.ndarray]) -> List[Dict[str, Any]]:
    dev = next(model.parameters()).device
    imgs = [cv2.resize(rgb, (640,640), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    ten = torch.from_numpy(np.stack(imgs, axis=0)).permute(0,3,1,2).float().to(dev)
    det_out, da_seg_out, ll_seg_out = model(ten)
    da = da_seg_out[:,0].detach().cpu().numpy()
    ll = ll_seg_out[:,0].detach().cpu().numpy()
    out = []
    for i in range(len(rgbs)):
        out.append({"det": None, "drivable": da[i], "lane": ll[i]})
    return out

def autotune_yolop_bs(model, logger: logging.Logger, sample_rgb: np.ndarray, cap: int = 64) -> int:
    if not (torch.cuda.is_available() and next(model.parameters()).is_cuda):
        logger.info("[AutoBatch][YOLOP] CPU/非CUDA のため batch=1 固定")
        return 1
    best, b, failed = 1, 1, False
    while b <= cap:
        try:
            _ = yolop_infer_batch(model, [sample_rgb]*b)
            del _
            _torch_sync_empty_cache()
            best = b
            b *= 2
        except RuntimeError as e:
            if _is_cuda_oom(e):
                failed = True
                _torch_sync_empty_cache()
                break
            raise
    lo = best
    hi = min(cap, b-1) if failed else best
    while lo < hi:
        mid = (lo + hi + 1)//2
        try:
            _ = yolop_infer_batch(model, [sample_rgb]*mid)
            del _
            _torch_sync_empty_cache()
            lo = mid
        except RuntimeError as e:
            if _is_cuda_oom(e):
                _torch_sync_empty_cache()
                hi = mid - 1
            else:
                raise
    logger.info("[AutoBatch][YOLOP] batch=%d", lo)
    return max(1, int(lo))

# ---------- Metric3D(ONNXRuntime): 一括推論(対応時) + 自動バッチ ----------
def infer_metric3d_batch(session: ort.InferenceSession, in_name: str, out_name: str, rgbs: List[np.ndarray]) -> List[np.ndarray]:
    # 入力を [B,3,H,W] に積む（ダイナミックバッチ非対応モデルなら例外）
    h0, w0 = rgbs[0].shape[:2]
    resized = [cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR).astype(np.float32) for rgb in rgbs]
    arr = np.stack([np.transpose(r, (2,0,1)) for r in resized], axis=0)
    y = session.run([out_name], {in_name: arr})[0]  # [B,H,W] or [B,1,H,W]
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:,0]
    out_list = []
    for i in range(y.shape[0]):
        depth = y[i].astype(np.float32)
        depth_up = cv2.resize(depth, (rgbs[i].shape[1], rgbs[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        out_list.append(depth_up)
    return out_list

def autotune_metric3d_bs(session: ort.InferenceSession, in_name: str, out_name: str, logger: logging.Logger,
                         sample_rgb: np.ndarray, cap: int = 16) -> int:
    # ONNX のダイナミックバッチに対応していない場合は 1 にフォールバック
    try:
        _ = infer_metric3d_batch(session, in_name, out_name, [sample_rgb, sample_rgb])
        del _
        # 2 が通るなら探索開始
        best, b, failed = 1, 2, False
        while b <= cap:
            try:
                _ = infer_metric3d_batch(session, in_name, out_name, [sample_rgb]*b)
                del _
                best = b
                b *= 2
            except Exception:
                failed = True
                break
        lo = best
        hi = min(cap, b-1) if failed else best
        while lo < hi:
            mid = (lo + hi + 1)//2
            try:
                _ = infer_metric3d_batch(session, in_name, out_name, [sample_rgb]*mid)
                del _
                lo = mid
            except Exception:
                hi = mid - 1
        logger.info("[AutoBatch][Metric3D] batch=%d", lo)
        return max(1, int(lo))
    except Exception:
        logger.info("[AutoBatch][Metric3D] 動的バッチ非対応 → batch=1")
        return 1

def _recanonize_dets(dets: List[Dict[str,Any]], prompts: List[str]) -> List[Dict[str,Any]]:
    out = []
    for d in dets:
        nd = dict(d)
        nd["cls"] = _canonicalize_label(d.get("cls",""), prompts)
        out.append(nd)
    return out

# ========== (3) 物体保持・ハルシネーション ==========



# YOLOP: PyTorch Hub（hustvl/YOLOP）で BDD100K の det+drivable+lane を同時出力
# 依存: prefetch_generator, thop, scikit-image（overlay で導入）。torch は Docker ベース層固定。
def load_yolop(logger: logging.Logger):
    """
    YOLOP を PyTorch Hub からロード。
    - 事前に overlay へ 'prefetch-generator', 'thop', 'scikit-image' を導入しておく（entrypoint が対応）
    - ここでは存在チェックのみ行い、欠如時は明確なエラーを出して停止（再現性重視）
    """
    logger.info("[YOLOP] load via torch.hub hustvl/yolop (pretrained=True, trust_repo=True)")

    # 依存の事前チェック
    try:
        import prefetch_generator  # noqa: F401
    except Exception as e:
        logger.error(
            "YOLOP 依存 'prefetch_generator' が見つかりません。overlay を確認してください。発生元: %s", e
        ); raise
    try:
        import thop  # noqa: F401
    except Exception as e:
        logger.warning("thop が見つかりません（続行可能ですが一部計測が無効になる可能性）: %s", e)

    # モデルロード
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

@torch.inference_mode()
def yolop_infer(model, rgb: np.ndarray) -> Dict[str, Any]:
    # 入力は (1,3,640,640) に正規化（RGB）
    img = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0)
    ten = ten.to(next(model.parameters()).device)
    det_out, da_seg_out, ll_seg_out = model(ten)
    return {
        "det": det_out,                                   # raw tensor（今回未使用）
        "drivable": da_seg_out[0][0].detach().cpu().numpy(),  # (H,W)=640
        "lane":     ll_seg_out[0][0].detach().cpu().numpy(),  # (H,W)=640
    }

# ---- ラベル正規化ユーティリティ ----
_CANON_PRIOR = [
    # より「具体的」→「汎用」の順で優先（同長の場合はこの順を優先）
    "speed limit sign", "stop sign", "crosswalk sign", "construction sign",
    "traffic light", "traffic cone", "traffic sign",
    "motorcycle", "bicycle", "truck", "bus", "car",
    "pedestrian", "person",
]

# ---- 検出ボックス座標の座標系変換（src_size=(H,W) → dst_size=(H,W)） ----
def _rescale_dets_to(dets: List[Dict[str,Any]],
                     src_size: Tuple[int,int],
                     dst_size: Tuple[int,int]) -> List[Dict[str,Any]]:
    """
    dets: [{'cls': str, 'score': float, 'bbox': [x1,y1,x2,y2]}, ...]
    src_size: (H_src, W_src)  ← dets の現在の座標系
    dst_size: (H_dst, W_dst)  ← 変換先（ここでは X 側画像の解像度）

    画像サイズが異なる場合、D 側（F(X)）のボックスを X 側の解像度に合わせてスケーリングする。
    """
    Hs, Ws = src_size
    Hd, Wd = dst_size
    if (Hs == Hd) and (Ws == Wd):
        return dets

    sx = float(Wd) / max(1.0, float(Ws))
    sy = float(Hd) / max(1.0, float(Hs))

    out: List[Dict[str,Any]] = []
    for d in dets:
        x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
        x1 = x1 * sx; x2 = x2 * sx
        y1 = y1 * sy; y2 = y2 * sy
        # 範囲クリップ（境界外は切り詰め）
        x1 = float(np.clip(x1, 0.0, max(0.0, Wd - 1)))
        x2 = float(np.clip(x2, 0.0, max(0.0, Wd - 1)))
        y1 = float(np.clip(y1, 0.0, max(0.0, Hd - 1)))
        y2 = float(np.clip(y2, 0.0, max(0.0, Hd - 1)))
        nd = dict(d); nd["bbox"] = [x1, y1, x2, y2]
        out.append(nd)
    return out

def _canonicalize_label(raw: str, prompts: List[str]) -> str:
    """
    Grounding DINO が返す複合ラベル（例: 'car bus', 'person pedestrian'）を
    事前に与えた prompts 内の「最も具体的」な1語に正規化する。
    - 長い語を優先、同長は _CANON_PRIOR の順でタイブレーク
    - pedestrian は 'person' に吸収
    """
    if not isinstance(raw, str):
        return str(raw)

    s = raw.lower().strip()
    # 候補抽出（部分一致）
    cands = [p for p in prompts if p in s]
    if not cands:
        return s

    # 長さ降順で並べ、同長は _CANON_PRIOR の順で優先
    def _key(p):
        return (len(p), -_CANON_PRIOR.index(p) if p in _CANON_PRIOR else 0)
    cands.sort(key=_key, reverse=True)
    canon = cands[0]

    # 同義吸収
    if canon == "pedestrian":
        return "person"
    return canon

class GroundingDINO:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_batch: Optional[int] = None

    def load(self, logger: logging.Logger):
        logger.info("[GroundingDINO] loading: %s", self.model_id)
        self.processor = GDinoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

    def autotune(self, logger: logging.Logger, sample_rgb: np.ndarray, prompts: List[str], cap: int = 64) -> int:
        if not torch.cuda.is_available():
            self.max_batch = 1
            logger.info("[AutoBatch][GDINO] CPU のため batch=1 固定")
            return 1
        # バッチ探索
        best, b, failed = 1, 1, False
        H, W = sample_rgb.shape[:2]
        while b <= cap:
            try:
                text_labels = [prompts] * b
                inputs = self.processor(images=[sample_rgb]*b, text=text_labels, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs)
                _ = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H,W)]*b
                )
                del inputs, outputs, _
                _torch_sync_empty_cache()
                best = b
                b *= 2
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    failed = True
                    _torch_sync_empty_cache()
                    break
                raise
        lo = best
        hi = min(cap, b-1) if failed else best
        while lo < hi:
            mid = (lo + hi + 1)//2
            try:
                text_labels = [prompts] * mid
                inputs = self.processor(images=[sample_rgb]*mid, text=text_labels, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs)
                _ = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H,W)]*mid
                )
                del inputs, outputs, _
                _torch_sync_empty_cache()
                lo = mid
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    _torch_sync_empty_cache()
                    hi = mid - 1
                else:
                    raise
        logger.info("[AutoBatch][GDINO] batch=%d", lo)
        self.max_batch = max(1, int(lo))
        return self.max_batch

    @torch.inference_mode()
    def detect(self, rgb: np.ndarray, prompts: List[str],
               box_thr: float = 0.35, txt_thr: float = 0.25) -> List[Dict[str, Any]]:
        text_labels = [prompts]
        inputs = self.processor(images=rgb, text=text_labels, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        H, W = rgb.shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, text_threshold=txt_thr, target_sizes=[(H, W)]
        )[0]
        raw_labels = results.get("text_labels", results.get("labels", []))
        boxes  = results.get("boxes", [])
        scores = results.get("scores", [])
        out = []
        for box, score, label in zip(boxes, scores, raw_labels):
            x1,y1,x2,y2 = [float(v) for v in box.tolist()]
            raw = str(label)
            cls = _canonicalize_label(raw, prompts)
            out.append({"cls": cls, "score": float(score), "bbox": [x1,y1,x2,y2]})
        return out

    @torch.inference_mode()
    def detect_batch(self, rgbs: List[np.ndarray], prompts: List[str],
                     box_thr: float = 0.35, txt_thr: float = 0.25) -> List[List[Dict[str, Any]]]:
        sizes = [im.shape[:2] for im in rgbs]
        text_labels = [prompts] * len(rgbs)
        inputs = self.processor(images=rgbs, text=text_labels, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, text_threshold=txt_thr, target_sizes=sizes
        )
        outs: List[List[Dict[str,Any]]] = []
        for res, (H, W) in zip(results, sizes):
            raw_labels = res.get("text_labels", res.get("labels", []))
            boxes  = res.get("boxes", [])
            scores = res.get("scores", [])
            dets=[]
            for box, score, label in zip(boxes, scores, raw_labels):
                x1,y1,x2,y2 = [float(v) for v in box.tolist()]
                raw = str(label)
                cls = _canonicalize_label(raw, prompts)
                dets.append({"cls": cls, "score": float(score), "bbox": [x1,y1,x2,y2]})
            outs.append(dets)
        return outs


# ========== 走行可能領域（Drivable）メトリクス ==========
def binarize_mask(arr: np.ndarray, thr: float=0.5, morph_k: int=0) -> np.ndarray:
    """連続値マスクを 0/1 に。必要ならモルフォロジー閉処理で穴埋め。"""
    m = (arr > thr).astype(np.uint8)
    if morph_k and morph_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def mask_metrics(mx: np.ndarray, mf: np.ndarray) -> Dict[str, float]:
    """
    2値マスクの一致度（IoU / Dice(F1) / Precision / Recall / L1 / RMSE）。
    形状が異なる場合は mx を mf のサイズに最近傍でリサイズ。
    """
    if mx.shape != mf.shape:
        mx = cv2.resize(mx, (mf.shape[1], mf.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = (mx > 0).astype(np.uint8)
    b = (mf > 0).astype(np.uint8)
    tp = float(np.sum((a==1) & (b==1)))
    fp = float(np.sum((a==0) & (b==1)))
    fn = float(np.sum((a==1) & (b==0)))
    tn = float(np.sum((a==0) & (b==0)))
    union = tp + fp + fn + 1e-9
    inter = tp
    iou = inter / union
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)
    l1   = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    rmse = float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32))**2)))
    return {
        "drv_iou": float(iou),
        "drv_f1": float(f1),
        "drv_precision": float(prec),
        "drv_recall": float(rec),
        "drv_l1": float(l1),
        "drv_rmse": float(rmse),
    }

def boundary_iou(mx: np.ndarray, mf: np.ndarray, tol: int=1) -> float:
    """
    境界一致度（Boundary IoU）。境界をモルフォロジー勾配で抽出し、tol ピクセルだけ膨張して相互 IoU。
    """
    if mx.shape != mf.shape:
        mx = cv2.resize(mx, (mf.shape[1], mf.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = (mx > 0).astype(np.uint8)
    b = (mf > 0).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    ea = cv2.morphologyEx(a, cv2.MORPH_GRADIENT, k3)
    eb = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, k3)
    if tol and tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
        ea = cv2.dilate(ea, k)
        eb = cv2.dilate(eb, k)
    inter = float(np.sum((ea>0) & (eb>0)))
    union = float(np.sum((ea>0) | (eb>0))) + 1e-9
    return float(inter / union)

def upsize_to(rgb: np.ndarray, m640: np.ndarray) -> np.ndarray:
    """YOLOP(640x640) の 2値マスクを RGB の元解像度に最近傍でアップサンプル"""
    h0, w0 = rgb.shape[:2]
    return cv2.resize((m640>0).astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)

# OCR（Tesseract 優先。未導入ならスキップ可 → 空文字）
def run_ocr_tesseract(rgb_crop: np.ndarray) -> str:
    """
    Tesseract バイナリ未導入・実行時例外でも落ちないように完全ガード。
    未導入時は空文字 "" を返す。
    """
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except Exception:
        return ""
    try:
        g = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        txt = pytesseract.image_to_string(g, lang="eng", config="--psm 7")
        return (txt or "").strip()
    except TesseractNotFoundError:
        # バイナリが無い場合
        return ""
    except Exception:
        # OCR 失敗時は評価を継続するため空文字で返す
        return ""


# ==== bbox_iou: 完全置換（安全な正規化＋正しい面積計算）====================
def bbox_iou(a: List[float], b: List[float]) -> float:
    """
    IoU between two boxes [x1,y1,x2,y2]. 座標が入れ替わっていても安全に補正。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 座標の正規化（x1<=x2, y1<=y2 を保証）
    if ax2 < ax1: ax1, ax2 = ax2, ax1
    if ay2 < ay1: ay1, ay2 = ay2, ay1
    if bx2 < bx1: bx1, bx2 = bx2, bx1
    if by2 < by1: by1, by2 = by2, by1

    # 交差
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    # 面積（必ず非負）
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / (union + 1e-9)
# ===========================================================================

def match_by_iou(G: List[Dict[str,Any]], D: List[Dict[str,Any]], thr: float) -> List[Tuple[int,int,float]]:
    if len(G)==0 or len(D)==0:
        return []
    C = np.zeros((len(G), len(D)), dtype=np.float32)
    for i,g in enumerate(G):
        for j,d in enumerate(D):
            if g["cls"] != d["cls"]:
                C[i,j] = 1.0  # 不一致クラスにはペナルティ
            else:
                iou = bbox_iou(g["bbox"], d["bbox"])
                C[i,j] = 1.0 - (iou if iou>=thr else 0.0)
    row_ind, col_ind = linear_sum_assignment(C)
    matches = []
    for i,j in zip(row_ind, col_ind):
        iou = bbox_iou(G[i]["bbox"], D[j]["bbox"]) if G[i]["cls"]==D[j]["cls"] else 0.0
        if iou >= thr:
            matches.append((i,j,iou))
    return matches

def det_metrics(
    G_all: List[Dict[str,Any]],
    D_all: List[Dict[str,Any]],
    iou_thr: float,
    img_wh_x: Tuple[int,int],
    img_wh_f: Tuple[int,int],
) -> Dict[str, Any]:
    """
    X と F(X) の bbox は元々ピクセル座標だが、解像度が異なる可能性が高い。
    比較の直前に [0,1] に正規化してから IoU / 中心誤差 / 面積比を計算する。
    """
    Wx, Hx = float(max(1, img_wh_x[0])), float(max(1, img_wh_x[1]))
    Wf, Hf = float(max(1, img_wh_f[0])), float(max(1, img_wh_f[1]))

    def _to_norm(dets: List[Dict[str,Any]], W: float, H: float) -> List[Dict[str,Any]]:
        out = []
        for d in dets:
            x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
            # [0,1] 正規化
            x1 /= W; x2 /= W; y1 /= H; y2 /= H
            # 並び保証 + [0,1] クリップ
            x1 = max(0.0, min(1.0, x1)); x2 = max(0.0, min(1.0, x2))
            y1 = max(0.0, min(1.0, y1)); y2 = max(0.0, min(1.0, y2))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            nd = dict(d); nd["bbox"] = [x1, y1, x2, y2]
            out.append(nd)
        return out

    # 正規化
    Gn = _to_norm(G_all, Wx, Hx)
    Dn = _to_norm(D_all, Wf, Hf)

    classes = sorted(list({g["cls"] for g in Gn} | {d["cls"] for d in Dn}))
    if len(classes) == 0:
        return {
            "per_class": {},
            "center_error_median": 0.0,
            "size_log_ratio_median": 0.0,
            "iou_median": 0.0,
            "count_absdiff_mean": 0.0,
            "count_wasserstein": 0.0,
            "classes": [],
        }

    # 正規化座標での IoU マッチング（座標系に依存しない）
    def _match_by_iou(G: List[Dict[str,Any]], D: List[Dict[str,Any]], thr: float) -> List[Tuple[int,int,float]]:
        if len(G)==0 or len(D)==0:
            return []
        C = np.zeros((len(G), len(D)), dtype=np.float32)
        for i,g in enumerate(G):
            for j,d in enumerate(D):
                if g["cls"] != d["cls"]:
                    C[i,j] = 1.0
                else:
                    iou = bbox_iou(g["bbox"], d["bbox"])
                    C[i,j] = 1.0 - (iou if iou>=thr else 0.0)
        row_ind, col_ind = linear_sum_assignment(C)
        matches = []
        for i,j in zip(row_ind, col_ind):
            iou = bbox_iou(G[i]["bbox"], D[j]["bbox"]) if G[i]["cls"]==D[j]["cls"] else 0.0
            if iou >= thr:
                matches.append((i,j,iou))
        return matches

    per_cls: Dict[str, Dict[str, float]] = {}
    center_errs=[]; size_ratio=[]; iou_list=[]

    for c in classes:
        Gc = [g for g in Gn if g["cls"]==c]
        Dc = [d for d in Dn if d["cls"]==c]
        M  = _match_by_iou(Gc, Dc, iou_thr)

        nG, nD, nM = len(Gc), len(Dc), len(M)
        PR = nM/(nG+1e-9)                 # Preservation-Recall
        PP = nM/(nD+1e-9)                 # Preservation-Precision
        F1 = 2*PR*PP/(PR+PP+1e-9)
        HAL = (nD - nM)/(nD+1e-9)         # ハルシネーション率

        # 幾何安定性（正規化座標）
        c_err=[]; s_err=[]; ious=[]
        for (ii,jj,ij_iou) in M:
            gx1,gy1,gx2,gy2 = Gc[ii]["bbox"]; dx1,dy1,dx2,dy2 = Dc[jj]["bbox"]
            gcx,gcy = (gx1+gx2)/2.0, (gy1+gy2)/2.0
            dcx,dcy = (dx1+dx2)/2.0, (dy1+dy2)/2.0
            cdist   = math.sqrt((gcx-dcx)**2 + (gcy-dcy)**2)       # すでに無次元
            garea   = max(1e-8, (gx2-gx1)*(gy2-gy1))
            darea   = max(1e-8, (dx2-dx1)*(dy2-dy1))
            sratio  = abs(math.log(darea/garea))
            c_err.append(cdist); s_err.append(sratio); ious.append(ij_iou)

        center_errs += c_err
        size_ratio  += s_err
        iou_list    += ious
        per_cls[c]   = {"PR":float(PR), "PP":float(PP), "F1":float(F1), "HAL":float(HAL),
                        "count_G":nG, "count_D":nD}

    # カウント整合性
    cntG = np.array([sum(1 for g in Gn if g["cls"]==c) for c in classes], dtype=np.float64)
    cntD = np.array([sum(1 for d in Dn if d["cls"]==c) for c in classes], dtype=np.float64)
    abs_diff = float(np.mean(np.abs(cntG - cntD)))

    x = np.arange(len(classes), dtype=np.float64)
    sumG = float(cntG.sum()); sumD = float(cntD.sum())
    if sumG <= 0.0 and sumD <= 0.0:
        emd = 0.0
    elif sumG > 0.0 and sumD > 0.0:
        wG = cntG / sumG; wD = cntD / sumD
        emd = float(wasserstein_distance(x, x, u_weights=wG, v_weights=wD))
    else:
        wG = cntG / (sumG + 1e-12)
        wD = cntD / (sumD + 1e-12)
        cdfG = np.cumsum(wG); cdfD = np.cumsum(wD)
        emd = float(np.sum(np.abs(cdfG - cdfD)))

    return {
        "per_class": per_cls,
        "center_error_median": float(np.median(center_errs)) if center_errs else 0.0,
        "size_log_ratio_median": float(np.median(size_ratio)) if size_ratio else 0.0,
        "iou_median": float(np.median(iou_list)) if iou_list else 0.0,
        "count_absdiff_mean": abs_diff,
        "count_wasserstein": emd,
        "classes": classes,
    }



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Uni-ControlNet: WaymoV2 定量評価スクリプト（単一スクリプト完結）")
    ap.add_argument("--suppress-gdino-futurewarning", action="store_true",
                help="GroundingDINO の FutureWarning を抑止（デフォルト: ON）")
    ap.add_argument("--orig-root", type=str, default=DEFAULT_ORIG_IMAGE_ROOT)
    ap.add_argument("--gen-root", type=str, default=DEFAULT_GEN_ROOT)
    ap.add_argument("--canny-x-root", type=str, default=DEFAULT_CANNY_ROOT_X)
    ap.add_argument("--depth-x-npy-root", type=str, default=DEFAULT_DEPTH_NPY_ROOT_X)
    ap.add_argument("--semseg-x-root", type=str, default=DEFAULT_SEMSEG_ROOT_X)
    ap.add_argument("--cache-root", type=str, default=DEFAULT_HDD_CACHE_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=["training","validation","testing"])
    ap.add_argument("--camera", type=str, default="front")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--tasks", type=str, choices=["all","reality","structure","objects","drivable"], default="all")
    ap.add_argument("--reality-metric", type=str, choices=["clip-cmmd","clip-fid","inception-fid"], default="clip-cmmd")
    ap.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_ID)
    ap.add_argument("--clip-batch", type=int, default=16)
    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument("--metric3d-onnx", type=str, default=DEFAULT_METRIC3D_ONNX)
    ap.add_argument("--use-yolop", action="store_true")
    ap.add_argument("--yolop-roi-filter", action="store_true",
                    help="YOLOP の走行可能マスクで検出中心をフィルタ（デフォルト無効）")
    ap.add_argument("--gdinomodel", type=str, default=DEFAULT_GDINO_ID)
    ap.add_argument("--det-prompts", type=str, nargs="*", default=DEFAULT_DET_PROMPTS)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--ocr-engine", type=str, choices=["none","tesseract"], default="tesseract")
    ap.add_argument("--annotation-mode", type=str, choices=["off","objects","structure","all","drivable"], default="off",
                    help="注釈を保存: objects=検出, structure=Edge/Depth/Semseg, drivable=YOLOP マスク, all=両方, off=なし")
    ap.add_argument("--annotate-limit", type=int, default=32, help="注釈を保存する最大枚数（split毎）")
    ap.add_argument("--annotate-out", type=str, default=os.path.join(DEFAULT_HDD_CACHE_ROOT, "viz"),
                    help="注釈画像の保存先（HDD）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--tb-dir", type=str, default=os.path.join(DEFAULT_HDD_CACHE_ROOT, "tensorboard"))
    ap.add_argument("--verbose", action="store_true")
    # ---- Auto-Batch 制御 ----
    ap.add_argument("--max-batch-cap", type=int, default=1024, help="自動探索の上限（CLIP/GDINO）。OneFormer/YOLOP は 64 まで。")
    ap.add_argument("--auto-batch", dest="auto_batch", action="store_true", help="自動バッチ探索を有効化（既定）")
    ap.add_argument("--no-auto-batch", dest="auto_batch", action="store_false", help="自動バッチ探索を無効化（手動 clip-batch のみ使用）")
    ap.set_defaults(auto_batch=True)
    # ---- ★ 追加: GDINO のしきい値を CLI から調整 ----
    ap.add_argument("--gdino-box-thr", type=float, default=0.35,
                    help="GroundingDINO の box しきい値（既定 0.35。小物体の Recall を上げるには 0.25 前後に下げる）")
    ap.add_argument("--gdino-text-thr", type=float, default=0.25,
                    help="GroundingDINO の text しきい値（既定 0.25。標識などは 0.20 前後に下げると Recall↑/HAL↑）")
    return ap.parse_args()



def main() -> None:
    args = parse_args()
    cache = Cache(args.cache_root)
    logger = setup_logger(cache.d_logs, args.verbose)
    log_env(logger)

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    sw: Optional[SummaryWriter] = SummaryWriter(args.tb_dir) if args.tb else None
    if sw:
        logger.info("TensorBoard: %s", args.tb_dir)

    # モデルハンドル
    clipper = None
    if args.tasks in ("all","reality"):
        clipper = ClipEmbedder(args.clip_model)

    sess_m3d = None; in_m3d = None; out_m3d = None; m3d_bs = 1
    # Metric3D(ONNX) は Structure 系タスクでのみ必要
    need_metric3d = (args.tasks in ("all", "structure"))
    if need_metric3d:
        try:
            sess_m3d, in_m3d, out_m3d, providers = build_metric3d_session(args.metric3d_onnx)
            logger.info("Metric3Dv2 ORT providers: %s", providers)
        except Exception as e:
            logger.error("Metric3Dv2 ONNX Runtime 構築失敗: %s", repr(e))
            sys.exit(1)


    onef_proc = None; onef_model = None; onef_bs = 1
    if args.tasks in ("all","structure"):
        onef_proc, onef_model = build_oneformer(DEFAULT_ONEFORMER_ID, device="cuda", fp16=True)
        logger.info("OneFormer loaded: %s", DEFAULT_ONEFORMER_ID)

    yolop_model = None; yolop_bs = 1
    if args.tasks in ("all","objects","drivable") and args.use_yolop:
        yolop_model = load_yolop(logger)

    gdino = None; gdino_bs = 1
    if args.tasks in ("all","objects"):
        gdino = GroundingDINO(args.gdinomodel); gdino.load(logger)

    ann_counter = {"objects": defaultdict(int), "structure": defaultdict(int), "drivable": defaultdict(int)}
    total_pairs = 0

    for split in args.splits:
        pairs = enumerate_pairs(args.orig_root, args.gen_root, split, args.camera, args.limit)
        if not pairs:
            logger.warning("[%s] ペアが見つかりません（split=%s, camera=%s）", split, split, args.camera)
            continue
        logger.info("[%s] 評価ペア数: %d", split, len(pairs))
        total_pairs += len(pairs)

        # サンプル画像（自動バッチ探索用）
        sample_rgb_x = imread_rgb(pairs[0][0])
        sample_rgb_f = imread_rgb(pairs[0][1])
        logger.info("[%s] sample resolution: X=%dx%d | F=%dx%d | scale=(%.3f, %.3f)",
                    split,
                    sample_rgb_x.shape[1], sample_rgb_x.shape[0],
                    sample_rgb_f.shape[1], sample_rgb_f.shape[0],
                    sample_rgb_f.shape[1] / max(1, sample_rgb_x.shape[1]),
                    sample_rgb_f.shape[0] / max(1, sample_rgb_x.shape[0]))


        # ---------- Reality: CLIP ----------
        if args.tasks in ("all","reality"):
            clipper.load(logger)
            if args.auto_batch:
                cap = int(args.max_batch_cap)
                try:
                    _ = clipper.autotune(logger, sample_rgb_f, cap=cap)
                except Exception as e:
                    logger.warning("CLIP AutoBatch 失敗: %s → 既定 clip-batch=%d を継続", repr(e), args.clip_batch)
            clip_bs = int(clipper.max_batch or args.clip_batch or 16)

            feats_x = []; feats_fx = []
            pbar = tqdm(pairs, desc=f"{split}-clip")
            batch_imgs_x=[]; batch_imgs_fx=[]; batch_indices=[]
            for (px, pfx, rel_dir, stem) in pbar:
                cp_x = cache.clip_path(split, rel_dir, stem, "x")
                cp_f = cache.clip_path(split, rel_dir, stem, "fx")
                if os.path.exists(cp_x) and os.path.exists(cp_f):
                    ex = np.load(cp_x)["feat"]; ef = np.load(cp_f)["feat"]
                    feats_x.append(ex[None,:] if ex.ndim==1 else ex)
                    feats_fx.append(ef[None,:] if ef.ndim==1 else ef)
                    pbar.set_postfix_str("cache")
                    continue
                imgx = imread_rgb(px); imgf = imread_rgb(pfx)
                batch_imgs_x.append(imgx); batch_imgs_fx.append(imgf); batch_indices.append((rel_dir, stem))
                if len(batch_imgs_x) >= clip_bs:
                    fx = clipper.embed_batch(batch_imgs_x); ff = clipper.embed_batch(batch_imgs_fx)
                    for (rel_dir2, stem2), ex, ef in zip(batch_indices, fx, ff):
                        ensure_dir(os.path.dirname(cache.clip_path(split, rel_dir2, stem2, "x")))
                        np.savez(cache.clip_path(split, rel_dir2, stem2, "x"), feat=ex)
                        np.savez(cache.clip_path(split, rel_dir2, stem2, "fx"), feat=ef)
                    feats_x.append(fx); feats_fx.append(ff)
                    batch_imgs_x=[]; batch_imgs_fx=[]; batch_indices=[]
                    pbar.set_postfix_str(f"embed{clip_bs}")
            if batch_imgs_x:
                fx = clipper.embed_batch(batch_imgs_x); ff = clipper.embed_batch(batch_imgs_fx)
                for (rel_dir2, stem2), ex, ef in zip(batch_indices, fx, ff):
                    ensure_dir(os.path.dirname(cache.clip_path(split, rel_dir2, stem2, "x")))
                    np.savez(cache.clip_path(split, rel_dir2, stem2, "x"), feat=ex)
                    np.savez(cache.clip_path(split, rel_dir2, stem2, "fx"), feat=ef)
                feats_x.append(fx); feats_fx.append(ff)
            feats_x = np.concatenate(feats_x, axis=0) if feats_x else np.zeros((0,512),dtype=np.float32)
            feats_fx = np.concatenate(feats_fx, axis=0) if feats_fx else np.zeros((0,512),dtype=np.float32)

            if feats_x.shape[0] > 0 and feats_fx.shape[0] > 0:
                if args.reality_metric == "clip-fid":
                    mu1, s1 = feats_to_stats(feats_x); mu2, s2 = feats_to_stats(feats_fx)
                    fid = compute_frechet_distance(mu1, s1, mu2, s2)
                    logger.info("[%s][CLIP-FID] = %.4f", split, fid)
                    if sw: sw.add_scalar(f"reality/clip_fid/{split}", fid, 0)
                elif args.reality_metric == "clip-cmmd":
                    mmd2 = gaussian_mmd(feats_x, feats_fx, sigma=None)
                    logger.info("[%s][CLIP-CMMD] (MMD^2) = %.4f", split, mmd2)
                    if sw: sw.add_scalar(f"reality/clip_cmmd/{split}", mmd2, 0)
                else:
                    logger.warning("inception-fid は本スクリプトでは非推奨。必要なら別計算に切替。")

        # ---------- Structure: Edge/Depth/Semseg（OneFormer をバッチ化、Metric3D は可能ならバッチ） ----------
        if args.tasks in ("all","structure"):
            if args.auto_batch and (onef_proc is not None):
                try:
                    onef_bs = autotune_oneformer_bs(onef_proc, onef_model, logger, sample_rgb_f, cap=64)
                except Exception as e:
                    logger.warning("OneFormer AutoBatch 失敗: %s → batch=1", repr(e)); onef_bs = 1
            if args.auto_batch and (sess_m3d is not None):
                try:
                    m3d_bs = autotune_metric3d_bs(sess_m3d, in_m3d, out_m3d, logger, sample_rgb_f, cap=16)
                except Exception as e:
                    logger.warning("Metric3D AutoBatch 失敗: %s → batch=1", repr(e)); m3d_bs = 1

            conf_mat = np.zeros((19,19), dtype=np.int64)
            edge_scores = []; depth_scores = []

            # セマンティクス・深度は「必要なものだけ」バッチ推論してキャッシュ
            # ここでは 1 パスで処理しつつ、semseg/depth は pending を溜めて吐く
            pend_for_seg = []; pend_meta_seg = []
            pend_for_dep = []; pend_meta_dep = []

            def _flush_seg():
                nonlocal pend_for_seg, pend_meta_seg
                if not pend_for_seg: return
                segs = oneformer_semseg_batch(onef_proc, onef_model, pend_for_seg) if onef_proc is not None else []
                for (split_, rd, st, pfx_), seg in zip(pend_meta_seg, segs):
                    seg_fx_path = cache.semseg_fx_path(split_, rd, st)
                    ensure_dir(os.path.dirname(seg_fx_path))
                    np.save(seg_fx_path, seg)
                pend_for_seg=[]; pend_meta_seg=[]

            def _flush_dep():
                nonlocal pend_for_dep, pend_meta_dep
                if not pend_for_dep: return
                if m3d_bs > 1:
                    depths = infer_metric3d_batch(sess_m3d, in_m3d, out_m3d, pend_for_dep)
                    for (split_, rd, st, pfx_), depth_fx in zip(pend_meta_dep, depths):
                        depth_fx_path = cache.depth_fx_path(split_, rd, st)
                        ensure_dir(os.path.dirname(depth_fx_path)); np.save(depth_fx_path, depth_fx)
                else:
                    for (split_, rd, st, pfx_) in pend_meta_dep:
                        rgbf = imread_rgb(pfx_)
                        depth_fx = infer_metric3d_np(sess_m3d, in_m3d, out_m3d, rgbf)
                        depth_fx_path = cache.depth_fx_path(split_, rd, st)
                        ensure_dir(os.path.dirname(depth_fx_path)); np.save(depth_fx_path, depth_fx)
                pend_for_dep=[]; pend_meta_dep=[]

            pbar = tqdm(pairs, desc=f"{split}-struct")
            for (px, pfx, rel_dir, stem) in pbar:
                # Edge
                edge_x_path = os.path.join(args.canny_x_root, split, rel_dir, f"{stem}_edge.png")
                if not os.path.exists(edge_x_path):
                    pbar.set_postfix_str("miss-edgeX"); continue
                edge_x = imread_gray(edge_x_path)
                edge_fx_path = cache.canny_fx_path(split, rel_dir, stem)
                if os.path.exists(edge_fx_path):
                    edge_fx = imread_gray(edge_fx_path)
                else:
                    rgbf = imread_rgb(pfx)
                    edge_fx = canny_cpu(rgbf, 100, 200, 3)
                    ensure_dir(os.path.dirname(edge_fx_path)); cv2.imwrite(edge_fx_path, edge_fx)
                e_met = edge_metrics(edge_x, edge_fx); edge_scores.append(e_met)

                # Depth
                depth_x_path = os.path.join(args.depth_x_npy_root, split, rel_dir, f"{stem}_depth.npy")
                if not os.path.exists(depth_x_path):
                    pbar.set_postfix_str("miss-depthX"); continue
                if not os.path.exists(cache.depth_fx_path(split, rel_dir, stem)):
                    if m3d_bs > 1:
                        pend_for_dep.append(imread_rgb(pfx))
                        pend_meta_dep.append((split, rel_dir, stem, pfx))
                        if len(pend_for_dep) >= m3d_bs: _flush_dep()
                    else:
                        rgbf = imread_rgb(pfx)
                        depth_fx = infer_metric3d_np(sess_m3d, in_m3d, out_m3d, rgbf)
                        ensure_dir(os.path.dirname(cache.depth_fx_path(split, rel_dir, stem))); np.save(cache.depth_fx_path(split, rel_dir, stem), depth_fx)
                depth_x = np.load(depth_x_path).astype(np.float32)
                depth_fx = np.load(cache.depth_fx_path(split, rel_dir, stem)).astype(np.float32)
                d_met = depth_metrics(depth_x, depth_fx); depth_scores.append(d_met)

                # Semseg
                seg_x_path = os.path.join(args.semseg_x_root, split, rel_dir, f"{stem}_predTrainId.npy")
                if not os.path.exists(seg_x_path):
                    pbar.set_postfix_str("miss-segX"); continue
                if not os.path.exists(cache.semseg_fx_path(split, rel_dir, stem)):
                    if onef_bs > 1:
                        pend_for_seg.append(imread_rgb(pfx))
                        pend_meta_seg.append((split, rel_dir, stem, pfx))
                        if len(pend_for_seg) >= onef_bs: _flush_seg()
                    else:
                        rgbf = imread_rgb(pfx)
                        seg_fx = oneformer_semseg(onef_proc, onef_model, rgbf)
                        ensure_dir(os.path.dirname(cache.semseg_fx_path(split, rel_dir, stem))); np.save(cache.semseg_fx_path(split, rel_dir, stem), seg_fx)
                seg_x = np.load(seg_x_path).astype(np.uint8)
                seg_fx = np.load(cache.semseg_fx_path(split, rel_dir, stem)).astype(np.uint8)
                if seg_x.shape != seg_fx.shape:
                    seg_x = cv2.resize(seg_x, (seg_fx.shape[1], seg_fx.shape[0]), interpolation=cv2.INTER_NEAREST)
                conf_mat += confusion_19(seg_x, seg_fx, ncls=19)

                if args.annotation_mode in ("structure","all"):
                    if ann_counter["structure"][split] < args.annotate_limit:
                        rgbf_vis = imread_rgb(pfx)
                        save_annotations_structure(args.annotate_out, split, rel_dir, stem, rgbf_vis, edge_fx, depth_fx, seg_fx)
                        ann_counter["structure"][split] += 1
                pbar.set_postfix_str("ok")

            _flush_seg(); _flush_dep()

            if edge_scores:
                es = {k: float(np.mean([d[k] for d in edge_scores])) for k in edge_scores[0].keys()}
                logger.info("[%s][Edge] %s", split, json.dumps(es, ensure_ascii=False))
                if sw:
                    for k,v in es.items(): sw.add_scalar(f"struct/edge_{k}/{split}", v, 0)
            if depth_scores:
                ds = {k: float(np.mean([d[k] for d in depth_scores])) for k in depth_scores[0].keys()}
                logger.info("[%s][Depth] %s", split, json.dumps(ds, ensure_ascii=False))
                if sw:
                    for k,v in ds.items(): sw.add_scalar(f"struct/depth_{k}/{split}", v, 0)
            if np.sum(conf_mat)>0:
                miou, ious = miou_from_conf(conf_mat)
                logger.info("[%s][Semseg] mIoU=%.4f", split, miou)
                if sw:
                    sw.add_scalar(f"struct/semseg_mIoU/{split}", miou, 0)
                    for cid, val in enumerate(ious):
                        sw.add_scalar(f"struct/semseg_IoU_c{cid}/{split}", float(val), 0)
                out_cm = os.path.join(cache.d_logs, f"{split}_confusion.npy")
                np.save(out_cm, conf_mat)

        # ---------- Objects: GDINO/YOLOP をバッチ化（2パス：キャッシュ→集計） ----------
        if args.tasks in ("all","objects"):
            # AutoBatch
            if args.auto_batch and (gdino is not None):
                try:
                    gdino_bs = gdino.autotune(logger, sample_rgb_f, args.det_prompts, cap=64)
                except Exception as e:
                    logger.warning("GDINO AutoBatch 失敗: %s → batch=1", repr(e)); gdino_bs = 1
            if args.auto_batch and (yolop_model is not None):
                try:
                    yolop_bs = autotune_yolop_bs(yolop_model, logger, sample_rgb_f, cap=64)
                except Exception as e:
                    logger.warning("YOLOP AutoBatch 失敗: %s → batch=1", repr(e)); yolop_bs = 1

            # ---- (1) 検出キャッシュの構築（X と FX を別々に、足りないものだけバッチ推論） ----
            # X 側
            miss_x = []
            for (px, pfx, rel_dir, stem) in pairs:
                gd_x_path = cache.gdino_path(split, rel_dir, stem, "x")
                if not os.path.exists(gd_x_path):
                    miss_x.append((px, rel_dir, stem, gd_x_path))
            pbar = tqdm(list(_chunks(miss_x, gdino_bs if gdino_bs>0 else 1)), desc=f"{split}-obj-gdX")
            for chunk in pbar:
                imgs = [imread_rgb(px) for (px,_,_,_) in chunk]
                # ★ しきい値を CLI から反映
                outs = gdino.detect_batch(
                    imgs, args.det_prompts,
                    box_thr=args.gdino_box_thr, txt_thr=args.gdino_text_thr
                )
                for (px, rel_dir, stem, outp), dets in zip(chunk, outs):
                    ensure_dir(os.path.dirname(outp)); json.dump(dets, open(outp,"w"), indent=2)


            # FX 側
            miss_f = []
            for (px, pfx, rel_dir, stem) in pairs:
                gd_f_path = cache.gdino_path(split, rel_dir, stem, "fx")
                if not os.path.exists(gd_f_path):
                    miss_f.append((pfx, rel_dir, stem, gd_f_path))
            pbar = tqdm(list(_chunks(miss_f, gdino_bs if gdino_bs>0 else 1)), desc=f"{split}-obj-gdF")
            for chunk in pbar:
                imgs = [imread_rgb(pfx) for (pfx,_,_,_) in chunk]
                # ★ しきい値を CLI から反映
                outs = gdino.detect_batch(
                    imgs, args.det_prompts,
                    box_thr=args.gdino_box_thr, txt_thr=args.gdino_text_thr
                )
                for (pfx, rel_dir, stem, outp), dets in zip(chunk, outs):
                    ensure_dir(os.path.dirname(outp)); json.dump(dets, open(outp,"w"), indent=2)


            # YOLOP（drivable マスク）キャッシュ（必要時）
            if args.use_yolop and yolop_model is not None:
                # X
                yx_miss = []
                for (px, pfx, rel_dir, stem) in pairs:
                    yx_path = cache.yolo_path(split, rel_dir, stem, "x")
                    if not os.path.exists(yx_path):
                        yx_miss.append((px, rel_dir, stem, yx_path))
                pbar = tqdm(list(_chunks(yx_miss, yolop_bs if yolop_bs>0 else 1)), desc=f"{split}-obj-yolopX")
                for chunk in pbar:
                    imgs = [imread_rgb(px) for (px,_,_,_) in chunk]
                    youts = yolop_infer_batch(yolop_model, imgs)
                    for (px, rel_dir, stem, outp), y in zip(chunk, youts):
                        drv_x = (y["drivable"]>0.5).astype(np.uint8)
                        ensure_dir(os.path.dirname(outp)); json.dump({"drivable": drv_x.tolist()}, open(outp,"w"))

                # F
                yf_miss = []
                for (px, pfx, rel_dir, stem) in pairs:
                    yf_path = cache.yolo_path(split, rel_dir, stem, "fx")
                    if not os.path.exists(yf_path):
                        yf_miss.append((pfx, rel_dir, stem, yf_path))
                pbar = tqdm(list(_chunks(yf_miss, yolop_bs if yolop_bs>0 else 1)), desc=f"{split}-obj-yolopF")
                for chunk in pbar:
                    imgs = [imread_rgb(pfx) for (pfx,_,_,_) in chunk]
                    youts = yolop_infer_batch(yolop_model, imgs)
                    for (pfx, rel_dir, stem, outp), y in zip(chunk, youts):
                        drv_f = (y["drivable"]>0.5).astype(np.uint8)
                        ensure_dir(os.path.dirname(outp)); json.dump({"drivable": drv_f.tolist()}, open(outp,"w"))

            # ---- (2) 指標計算（キャッシュ読取のみ、必要なら ROI フィルタ適用） ----
            # ---- (2) 指標計算（キャッシュ読取のみ、必要なら ROI フィルタ適用） ----
            # ---- (2) 指標計算（キャッシュ読取のみ、必要なら ROI フィルタ適用） ----
            per_image_results = []
            pbar = tqdm(pairs, desc=f"{split}-obj")
            for (px, pfx, rel_dir, stem) in pbar:
                rgbx = imread_rgb(px); rgbf = imread_rgb(pfx)
                H, W   = rgbx.shape[:2]
                Hf, Wf = rgbf.shape[:2]

                gd_x_path = cache.gdino_path(split, rel_dir, stem, "x")
                gd_f_path = cache.gdino_path(split, rel_dir, stem, "fx")
                G = json.load(open(gd_x_path,"r")); G = _recanonize_dets(G, args.det_prompts)
                D = json.load(open(gd_f_path,"r")); D = _recanonize_dets(D, args.det_prompts)

                # --- ROIフィルタ（drivable）: ★640→元解像度へアップサイズしてから使用する★
                drv_x = None
                drv_f = None
                if args.use_yolop and yolop_model is not None:
                    # JSONから640x640の2値マスクを読み出し → 元画像サイズにアップサイズ
                    yx_path = cache.yolo_path(split, rel_dir, stem, "x")
                    yf_path = cache.yolo_path(split, rel_dir, stem, "fx")

                    yx = json.load(open(yx_path, "r"))  # {"drivable": [[...]]}
                    yf = json.load(open(yf_path, "r"))

                    m640_x = (np.array(yx["drivable"], dtype=np.uint8) > 0).astype(np.uint8)
                    m640_f = (np.array(yf["drivable"], dtype=np.uint8) > 0).astype(np.uint8)

                    # ★ここが肝：元解像度に合わせる
                    drv_x = upsize_to(rgbx, m640_x)  # shape=(Hx,Wx)
                    drv_f = upsize_to(rgbf, m640_f)  # shape=(Hf,Wf)

                    if args.yolop_roi_filter:
                        def keep_roi(dets: List[Dict[str,Any]], drv_u8: np.ndarray) -> List[Dict[str,Any]]:
                            kept = []
                            h_, w_ = drv_u8.shape[:2]
                            for d in dets:
                                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                                cx = int(np.clip((x1 + x2) / 2, 0, w_ - 1))
                                cy = int(np.clip((y1 + y2) / 2, 0, h_ - 1))
                                if drv_u8[cy, cx] > 0:
                                    kept.append(d)
                            return kept
                        G = keep_roi(G, drv_x)
                        D = keep_roi(D, drv_f)
                # --- ここまで ROI フィルタ修正（アノテの重ね描きは後段で渡す）


                if args.ocr_engine == "tesseract":
                    ocrX = []; ocrF = []
                    for d in G:
                        if "sign" in d["cls"]:
                            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
                            crop = rgbx[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                            ocrX.append({"bbox":d["bbox"], "txt":run_ocr_tesseract(crop)})
                    for d in D:
                        if "sign" in d["cls"]:
                            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
                            crop = rgbf[max(0,y1):min(Hf,y2), max(0,x1):min(Wf,x2)]
                            ocrF.append({"bbox":d["bbox"], "txt":run_ocr_tesseract(crop)})
                    ocr_x_out = cache.ocr_path(split, rel_dir, stem, "x")
                    ocr_f_out = cache.ocr_path(split, rel_dir, stem, "f")
                    ensure_dir(os.path.dirname(ocr_x_out)); ensure_dir(os.path.dirname(ocr_f_out))
                    json.dump(ocrX, open(ocr_x_out,"w"), indent=2)
                    json.dump(ocrF, open(ocr_f_out,"w"), indent=2)

                # ★ 正規化比較：画像サイズ差を吸収
                dm = det_metrics(G, D, iou_thr=args.iou_thr, img_wh_x=(W,H), img_wh_f=(Wf,Hf))
                per_image_results.append(dm)

                if args.annotation_mode in ("objects","all"):
                    if ann_counter["objects"][split] < args.annotate_limit:
                        ann_drv_x = drv_x if args.yolop_roi_filter else None
                        ann_drv_f = drv_f if args.yolop_roi_filter else None
                        save_annotations_objects(
                            args.annotate_out, split, rel_dir, stem,
                            rgbx, rgbf, G, D, ann_drv_x, ann_drv_f
                        )

                        ann_counter["objects"][split] += 1
                pbar.set_postfix_str("ok")



            if per_image_results:
                classes = sorted(list({c for r in per_image_results for c in r["per_class"].keys()}))
                agg: Dict[str, Dict[str,float]] = {c: {"PR":0.0,"PP":0.0,"F1":0.0,"HAL":0.0,"N":0.0} for c in classes}
                iou_med=[]; cen_med=[]; sz_med=[]; cad=[]; emd=[]
                for r in per_image_results:
                    iou_med.append(r["iou_median"]); cen_med.append(r["center_error_median"]); sz_med.append(r["size_log_ratio_median"])
                    cad.append(r["count_absdiff_mean"]); emd.append(r["count_wasserstein"])
                    for c,dd in r["per_class"].items():
                        for k in ["PR","PP","F1","HAL"]:
                            agg[c][k] += dd[k]
                        agg[c]["N"] += 1.0
                for c in classes:
                    if agg[c]["N"]>0:
                        for k in ["PR","PP","F1","HAL"]:
                            agg[c][k] = float(agg[c][k]/agg[c]["N"])
                summary = {
                    "iou_median": (float(np.median(iou_med)) if iou_med else float("nan")),
                    "center_error_median": (float(np.median(cen_med)) if cen_med else float("nan")),
                    "size_log_ratio_median": (float(np.median(sz_med)) if sz_med else float("nan")),
                    "count_absdiff_mean": (float(np.mean(cad)) if cad else float("nan")),
                    "count_wasserstein_mean": (float(np.mean(emd)) if emd else float("nan")),
                    "per_class": agg,
                }
                logger.info("[%s][Objects] %s", split, json.dumps(summary, ensure_ascii=False))
                if sw:
                    sw.add_scalar(f"objects/iou_median/{split}", summary["iou_median"], 0)
                    sw.add_scalar(f"objects/center_err_med/{split}", summary["center_error_median"], 0)
                    sw.add_scalar(f"objects/size_log_ratio_med/{split}", summary["size_log_ratio_median"], 0)
                    sw.add_scalar(f"objects/count_absdiff_mean/{split}", summary["count_absdiff_mean"], 0)
                    sw.add_scalar(f"objects/count_wasserstein/{split}", summary["count_wasserstein_mean"], 0)
                    for c in classes:
                        for k in ["PR","PP","F1","HAL"]:
                            sw.add_scalar(f"objects/{k}/{split}/{c}", agg[c][k], 0)

        # ---------- Drivable（保持評価のみ独立に回す時用） ----------
        if args.tasks == "drivable" and args.use_yolop and yolop_model is not None:
            # 既にキャッシュされていなければ YOLOP をバッチ推論
            if args.auto_batch:
                try:
                    yolop_bs = autotune_yolop_bs(yolop_model, logger, sample_rgb_f, cap=64)
                except Exception as e:
                    logger.warning("YOLOP AutoBatch 失敗: %s → batch=1", repr(e)); yolop_bs = 1

            yx_miss=[]; yf_miss=[]
            for (px, pfx, rel_dir, stem) in pairs:
                yx_path = cache.yolo_path(split, rel_dir, stem, "x")
                yf_path = cache.yolo_path(split, rel_dir, stem, "fx")
                if not os.path.exists(yx_path): yx_miss.append((px, rel_dir, stem, yx_path))
                if not os.path.exists(yf_path): yf_miss.append((pfx, rel_dir, stem, yf_path))
            for chunk in tqdm(list(_chunks(yx_miss, yolop_bs)), desc=f"{split}-drvX"):
                imgs = [imread_rgb(px) for (px,_,_,_) in chunk]
                youts = yolop_infer_batch(yolop_model, imgs)
                for (px, rel_dir, stem, outp), y in zip(chunk, youts):
                    drv = (y["drivable"]>0.5).astype(np.uint8)
                    ensure_dir(os.path.dirname(outp)); json.dump({"drivable": drv.tolist()}, open(outp,"w"))
            for chunk in tqdm(list(_chunks(yf_miss, yolop_bs)), desc=f"{split}-drvF"):
                imgs = [imread_rgb(pfx) for (pfx,_,_,_) in chunk]
                youts = yolop_infer_batch(yolop_model, imgs)
                for (pfx, rel_dir, stem, outp), y in zip(chunk, youts):
                    drv = (y["drivable"]>0.5).astype(np.uint8)
                    ensure_dir(os.path.dirname(outp)); json.dump({"drivable": drv.tolist()}, open(outp,"w"))

            # 指標
            ious=[]; f1s=[]; precs=[]; recs=[]; l1s=[]; rmses=[]; bious=[]
            for (px, pfx, rel_dir, stem) in tqdm(pairs, desc=f"{split}-drivable"):
                yx = json.load(open(cache.yolo_path(split, rel_dir, stem, "x"),"r"))["drivable"]
                yf = json.load(open(cache.yolo_path(split, rel_dir, stem, "fx"),"r"))["drivable"]
                a = (np.array(yx, dtype=np.uint8)>0).astype(np.uint8)
                b = (np.array(yf, dtype=np.uint8)>0).astype(np.uint8)
                inter = float(np.sum((a&b)>0)); union = float(np.sum((a|b)>0)+1e-9)
                iou = inter/union
                tp = inter
                fp = float(np.sum((b>0)&(a==0))); fn = float(np.sum((a>0)&(b==0)))
                prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
                f1 = 2*prec*rec/(prec+rec+1e-9)
                l1 = float(np.mean(np.abs(a.astype(np.float32)-b.astype(np.float32))))
                rmse = float(np.sqrt(np.mean((a.astype(np.float32)-b.astype(np.float32))**2)))
                # 簡易境界 IoU（1px 膨張）
                k = np.ones((3,3), np.uint8)
                ba = cv2.dilate(a, k) ^ a; bb = cv2.dilate(b, k) ^ b
                b_inter = float(np.sum((ba&bb)>0)); b_union = float(np.sum((ba|bb)>0)+1e-9)
                b_iou = b_inter/b_union
                ious.append(iou); f1s.append(f1); precs.append(prec); recs.append(rec)
                l1s.append(l1); rmses.append(rmse); bious.append(b_iou)
                # 可視化
                if args.annotation_mode in ("drivable","all"):
                    if ann_counter["drivable"][split] < args.annotate_limit:
                        rgbx = imread_rgb(px); rgbf = imread_rgb(pfx)
                        x_vis = overlay_mask(rgbx, a>0, alpha=0.3, color=(0,255,255))
                        f_vis = overlay_mask(rgbf, b>0, alpha=0.3, color=(255,255,0))
                        ensure_dir(os.path.join(args.annotate_out, split, rel_dir))
                        cv2.imwrite(os.path.join(args.annotate_out, split, rel_dir, f"{stem}_x_drv.jpg"), cv2.cvtColor(x_vis, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(args.annotate_out, split, rel_dir, f"{stem}_fx_drv.jpg"), cv2.cvtColor(f_vis, cv2.COLOR_RGB2BGR))
                        ann_counter["drivable"][split] += 1
            avg = {
                "drv_iou": float(np.mean(ious)), "drv_f1": float(np.mean(f1s)),
                "drv_precision": float(np.mean(precs)), "drv_recall": float(np.mean(recs)),
                "drv_l1": float(np.mean(l1s)), "drv_rmse": float(np.mean(rmses)),
                "drv_boundary_iou": float(np.mean(bious)),
            }
            logger.info("[%s][Drivable] %s", split, json.dumps(avg, ensure_ascii=False))
            if sw:
                for k,v in avg.items(): sw.add_scalar(f"drivable/{k}/{split}", v, 0)

    logger.info("✅ 全 split 完了。総ペア数: %d | キャッシュ: %s", total_pairs, args.cache_root)
    if sw: sw.close()



if __name__ == "__main__":
    main()


"""(以下)/home/shogo/coding/eval/ucn_eval/docker/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail
#!/usr/bin/env bash
set -euo pipefail

# ============= GPU / Torch 情報 =============
nvidia-smi || true
python3 - <<'PY'
import sys
try:
    import torch
    print(f"[entrypoint] torch={torch.__version__}, torch.version.cuda(build)={getattr(torch.version,'cuda',None)}")
    print(f"[entrypoint] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[entrypoint] device0={torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[entrypoint] FATAL: import torch failed: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ============= 永続 pip オーバーレイ（再ビルド不要/ハッシュ差分適用） =============
# 仕組み:
# - --target でホストへインストール（永続）。PYTHONPATH 先頭に載せてベースを壊さない。
# - 入力（PIP_INSTALL/requirements.overlay.txt）のハッシュが変わったときだけ実行。
PIP_OVERLAY_DIR="${PIP_OVERLAY_DIR:-/mnt/hdd/ucn_eval_cache/pip-overlay}"
REQS_OVERLAY_PATH="${REQS_OVERLAY_PATH:-/mnt/hdd/ucn_eval_cache/requirements.overlay.txt}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "$PIP_OVERLAY_DIR"
export PYTHONPATH="${PIP_OVERLAY_DIR}:${PYTHONPATH:-}"

_overlay_inputs=""
if [ -n "${PIP_INSTALL:-}" ]; then _overlay_inputs="${PIP_INSTALL}"; fi
if [ -f "$REQS_OVERLAY_PATH" ]; then
  _overlay_inputs="${_overlay_inputs}"$'\n'"$(cat "$REQS_OVERLAY_PATH")"
fi

if [ -n "${_overlay_inputs}" ]; then
  need_hash="$(printf "%s" "${_overlay_inputs}" | sha1sum | awk '{print $1}')"
  prev_hash="$(cat "${PIP_OVERLAY_DIR}/.overlay_hash" 2>/dev/null || echo "")"
  if [ "${need_hash}" != "${prev_hash}" ]; then
    echo "[entrypoint] Installing/updating overlay packages into: ${PIP_OVERLAY_DIR}"

    # ===== Overlay 安全インストール（torch 系を絶対に入れない） =====
    SAFE_REQ="${PIP_OVERLAY_DIR}/.reqs.safe.txt"
    NODEPS_REQ="${PIP_OVERLAY_DIR}/.reqs.nodeps.txt"
    rm -f "$SAFE_REQ" "$NODEPS_REQ"

    if [ -f "$REQS_OVERLAY_PATH" ]; then
      # torch/torchvision/torchaudio は完全に除外。
      # thop と timm は --no-deps でインストール（依存として torch 等を引っ張らせない）。
      while IFS= read -r line; do
        trimmed="$(printf '%s' "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [ -z "$trimmed" ] && continue
        printf '%s\n' "$trimmed" | grep -Eqi '^(torch|torchvision|torchaudio)($|[<=>])' && continue
        printf '%s\n' "$trimmed" | grep -Eqi '^(thop|timm)($|[<=>])' && { echo "$trimmed" >> "$NODEPS_REQ"; continue; }
        echo "$trimmed" >> "$SAFE_REQ"
      done < "$REQS_OVERLAY_PATH"
    fi

    # 1) 依存ありで安全群をインストール（ここで torch は入らない）
    if [ -s "$SAFE_REQ" ]; then
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" -r "$SAFE_REQ"
    fi
    # 2) 依存なしで NoDeps 群（thop, timm）を個別導入
    if [ -s "$NODEPS_REQ" ]; then
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" --no-deps -r "$NODEPS_REQ"
    fi
    # 追加の単発指定（環境変数 PIP_INSTALL）も処理（torch* は除外）
    if [ -n "${PIP_INSTALL:-}" ]; then
      _pi_sanitized="$(printf '%s\n' "$PIP_INSTALL" | tr ' ' '\n' | grep -Evi '^(torch|torchvision|torchaudio)($|[<=>])' | tr '\n' ' ')"
      if [ -n "$_pi_sanitized" ]; then
        # shellcheck disable=SC2086
        python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" $_pi_sanitized
      fi
    fi

    # 3) 念のため overlay から torch 系を物理削除（誤混入の保険）
    rm -rf "${PIP_OVERLAY_DIR}/torch" "${PIP_OVERLAY_DIR}/torchvision" "${PIP_OVERLAY_DIR}/torchaudio" || true
    rm -rf "${PIP_OVERLAY_DIR}"/nvidia_* "${PIP_OVERLAY_DIR}"/nvidia* || true

    # 4) 事後検査: huggingface_hub が 1.x なら強制的に 0.44.1 に固定
    python3 - <<'PY'
import sys
try:
    import transformers, huggingface_hub
    from packaging.version import Version
    tv = Version(transformers.__version__)
    hv = Version(huggingface_hub.__version__)
    print(f"[entrypoint] versions: transformers={tv}, huggingface_hub={hv}")
    if hv.major >= 1:
        # signal to shell
        sys.exit(42)
except Exception as e:
    print(f"[entrypoint] version check note: {e}")
    sys.exit(0)
PY
    ret=$?
    if [ "$ret" -eq 42 ]; then
      echo "[entrypoint] Downgrading huggingface_hub to 0.44.1 for transformers compatibility (<1.0)"
      python3 -m pip install --no-cache-dir --upgrade --target "$PIP_OVERLAY_DIR" "huggingface_hub==0.44.1"
    fi

    echo "${need_hash}" > "${PIP_OVERLAY_DIR}/.overlay_hash"
  else
    echo "[entrypoint] Overlay unchanged; skip pip."
  fi
else
  echo "[entrypoint] No overlay inputs (PIP_INSTALL/requirements.overlay.txt)."
fi

# 可視化: 重要依存が読めるか即確認（バージョンも表示）
python3 - <<'PY'
import importlib.util, os
def status(m): return "OK" if importlib.util.find_spec(m) else "MISSING"
mods = [
  "prefetch_generator","easydict","thop","skimage",       # YOLOP系
  "timm","einops","transformers","huggingface_hub","safetensors",  # HF/GDINO/OneFormer系
  "onnxruntime","cv2","numpy","scipy","PIL","yaml","torchvision"   # 共通
]
print("[entrypoint] PYTHONPATH head:", os.environ.get("PYTHONPATH","").split(":")[0])
for m in mods:
    print(f"[entrypoint] import {m}:", status(m))
try:
    import transformers, huggingface_hub
    print(f"[entrypoint] versions summary: transformers={transformers.__version__}, huggingface_hub={huggingface_hub.__version__}")
except Exception:
    pass
PY

# ============= 評価スクリプト起動 =============
exec python3 -u /app/eval_unicontrol_waymo.py "$@"



"""
"""以下、/home/shogo/coding/eval/ucn_eval/docker/Dockerfile


#/home/shogo/coding/eval/ucn_eval/docker/Dockerfile
# CUDA 12.8 ランタイム（NVIDIA公式）
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 基本ツール + Tesseract（OCR）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git wget curl ca-certificates \
      libgl1 libglib2.0-0 tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ベース依存（torch以外）— 最低限のみ。上位依存は起動時オーバーレイに委譲。
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -U pip && \
    python3 -m pip install -r /app/requirements.txt

# PyTorch (cu128固定) — RTX 5090 / CUDA 12.8 に厳密整合
RUN python3 -m pip install \
      torch==2.7.0 \
      torchvision==0.22.0 \
      torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128

# スクリプト本体 & 既定エントリポイント
COPY eval_unicontrol_waymo.py /app/eval_unicontrol_waymo.py
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
      - （任意）YOLOP（drivable ROI を用いた検出フィルタ）
      - OCR(Tesseract) による標識読み取り
      - 指標: 保持再現率(PR) / 保持適合率(PP) / F1 / ハルシネーション率(HAL)
              幾何安定性（中心誤差/サイズ比誤差/IoU中央値）
              カウント整合性（平均差 / 擬EMD）
- アノテーション可視化（--annotation-mode）:
  objects: X/FX の検出ボックス（+drivable ROI）を描画
  structure: FX の Edge/Depth/Semseg を横並び保存
  all: 両方
  off: なし
- 既定パスは翔伍さんの資産に合わせて固定。CLI で上書き可。
- 再現性: 乱数固定（--seed）、TensorBoard（--tb）対応。

注意:
- Docker 側で torch==2.7.0+cu128 / torchvision==0.22.0+cu128 を固定。
- 追加依存（timm, thop, prefetch_generator, scikit-image）は overlay で導入。
- onnxruntime は CPU/GPU どちらでも動作可（CUDAExecutionProvider があれば GPU 使用）。
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
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "eval.log")
    logger = logging.getLogger("ucn_eval")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = handlers.RotatingFileHandler(log_path, maxBytes=20*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"))
    if not logger.handlers:
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

def log_env(logger: logging.Logger) -> None:
    try:
        logger.info("torch: %s | build_cuda: %s | cuda_available: %s",
                    torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("cuda device 0: %s", torch.cuda.get_device_name(0))
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
    rel_dir = os.path.relpath(os.path.dirname(image_path), split_root)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return rel_dir, stem

def path_ucn_png(gen_root: str, split: str, rel_dir: str, stem: str) -> str:
    p = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn.png")
    if os.path.exists(p):
        return p
    cand = os.path.join(gen_root, split, rel_dir, f"{stem}_ucn-000.png")
    return cand

def enumerate_pairs(orig_root: str, gen_root: str, split: str, camera: str, limit: int) -> List[Tuple[str, str, str, str]]:
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


# ==== [YOLOP mask helper: 新規関数を追加] ==================================
def resize_mask_to_image(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    YOLOP 等の (Hm,Wm) マスクを、画像サイズ (target_h, target_w) に最近傍で拡大。
    返り値は uint8 の {0,1} マスク。
    """
    m = (mask > 0).astype(np.uint8)
    if m.shape[0] == target_h and m.shape[1] == target_w:
        return m
    m_rs = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (m_rs > 0).astype(np.uint8)

def keep_by_drivable_center(det_list: List[Dict[str,Any]], drv_mask01: np.ndarray) -> List[Dict[str,Any]]:
    """
    ボックス中心が drivable(==1) 内にある検出のみ残す。
    drv_mask01 は画像サイズと同じ (H,W) の {0,1} マスクであること。
    """
    H, W = drv_mask01.shape[:2]
    kept = []
    for d in det_list:
        x1,y1,x2,y2 = d["bbox"]
        cx, cy = int(round((x1+x2)/2)), int(round((y1+y2)/2))
        if 0 <= cx < W and 0 <= cy < H and drv_mask01[cy, cx] > 0:
            kept.append(d)
    return kept
# ==== [YOLOP mask helper END] ==============================================

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
    e3 = cv2.cvtColor((edge_fx>0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    d = depth_fx.copy()
    d -= np.min(d); d /= (np.max(d)+1e-9); d = (d*255).clip(0,255).astype(np.uint8)
    d3 = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    s3 = colorize_trainId(seg_fx)
    cat = np.concatenate([rgbf, e3, d3, s3], axis=1)
    cv2.imwrite(os.path.join(outdir, split, rel_dir, f"{stem}_fx_struct.jpg"), cv2.cvtColor(cat, cv2.COLOR_RGB2BGR))


# ========== (1) リアリティ指標：CLIP特徴, FID/CMMD ==========
class ClipEmbedder:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPImageProcessor] = None

    def load(self, logger: logging.Logger):
        if self.model is None:
            logger.info("[CLIP] loading model: %s", self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            self.processor = CLIPImageProcessor.from_pretrained(self.model_id)

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
    feats = np.asarray(feats, dtype=np.float64)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
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
    def pdist_sq(a: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)
        return aa + aa.T - 2.0 * (a @ a.T)
    def cdist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True)
        return aa + bb.T - 2.0 * (a @ b.T)
    xx = pdist_sq(x); yy = pdist_sq(y); xy = cdist_sq(x, y)
    if sigma is None:
        med = np.median(np.concatenate([xx.ravel(), yy.ravel(), xy.ravel()]))
        sigma = max(math.sqrt(0.5 * med), 1e-6)
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = np.exp(-gamma * xx); k_yy = np.exp(-gamma * yy); k_xy = np.exp(-gamma * xy)
    m = x.shape[0]; n = y.shape[0]
    mmd2 = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1) + 1e-9) \
         + (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1) + 1e-9) \
         - 2.0 * np.sum(k_xy) / (m * n)
    return float(mmd2)


# ========== (2) 構造・意味忠実度 ==========
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
    t = np.transpose(rgb_resized, (2,0,1))[None, ...]
    y = session.run([out_name], {in_name: t})[0]
    depth = np.squeeze(y).astype(np.float32)
    depth_up = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return depth_up

def depth_metrics(depth_x: np.ndarray, depth_fx: np.ndarray) -> Dict[str, float]:
    """
    Depth の RMSE / scale-invariant RMSE / 相対誤差。
    サイズが異なる場合は depth_x を depth_fx サイズにリサイズして比較。
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

def build_oneformer(model_id: str, device: str = "cuda", fp16: bool = True):
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
    model = OneFormerForUniversalSegmentation.from_pretrained(model_id, torch_dtype=dtype).eval()
    if device.startswith("cuda") and torch.cuda.is_available():
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
    tp = np.diag(hist).astype(np.float64)
    fp = np.sum(hist, axis=0) - tp
    fn = np.sum(hist, axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    miou = float(np.nanmean(iou))
    return miou, iou


# ========== (3) 物体保持・ハルシネーション ==========
def load_yolop(logger: logging.Logger):
    logger.info("[YOLOP] load via torch.hub hustvl/yolop (pretrained=True, trust_repo=True)")
    try:
        import prefetch_generator  # noqa: F401
    except Exception as e:
        logger.error("YOLOP 依存 'prefetch_generator' が見つかりません: %s", e); raise
    try:
        import thop  # noqa: F401
    except Exception as e:
        logger.warning("thop が見つかりません（続行可能）: %s", e)
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

@torch.inference_mode()
def yolop_infer(model, rgb: np.ndarray) -> Dict[str, Any]:
    img = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0)
    ten = ten.to(next(model.parameters()).device)
    det_out, da_seg_out, ll_seg_out = model(ten)
    return {"det": det_out, "drivable": da_seg_out[0][0].detach().cpu().numpy(), "lane": ll_seg_out[0][0].detach().cpu().numpy()}

# ==== [PATCH 2/3: クラス名の正規化関数を新規追加] ==========================
# ---- ラベル正規化ユーティリティ ----
_CANON_PRIOR = [
    # より「具体的」→「汎用」の順で優先（同長の場合はこの順を優先）
    "speed limit sign", "stop sign", "crosswalk sign", "construction sign",
    "traffic light", "traffic cone", "traffic sign",
    "motorcycle", "bicycle", "truck", "bus", "car",
    "pedestrian", "person",
]

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

    def load(self, logger: logging.Logger):
        logger.info("[GroundingDINO] loading: %s", self.model_id)
        self.processor = GDinoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

    @torch.inference_mode()
    def detect(self, rgb: np.ndarray, prompts: List[str],
               box_thr: float = 0.35, txt_thr: float = 0.25) -> List[Dict[str, Any]]:
        """
        - Transformers v4.51+ の仕様変更に対応（text_labels を優先）
        - ラベルを _canonicalize_label() で正規化し、クラス一致性を高める
        """
        text_labels = [prompts]  # [["car","bus",...]]
        inputs = self.processor(images=rgb, text=text_labels, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        H, W = rgb.shape[:2]
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_thr, text_threshold=txt_thr, target_sizes=[(H, W)]
        )[0]

        # v4.51+ では labels が int id になり得るため text_labels を優先
        raw_labels = results.get("text_labels", results.get("labels", []))

        # 安全化（len 一致を保証）
        boxes  = results.get("boxes", [])
        scores = results.get("scores", [])
        if len(raw_labels) != len(boxes):
            # 長さ不一致時はラベルをすべて raw(str) 化して合わせる
            if isinstance(raw_labels, list):
                pass
            else:
                raw_labels = [raw_labels] * len(boxes)

        out = []
        for box, score, label in zip(boxes, scores, raw_labels):
            x1,y1,x2,y2 = [float(v) for v in box.tolist()]
            raw = str(label)
            cls = _canonicalize_label(raw, prompts)
            out.append({"cls": cls, "score": float(score), "bbox": [x1,y1,x2,y2]})
        return out

# ==== [PATCH 2/3 END of detect] ============================================

def run_ocr_tesseract(rgb_crop: np.ndarray) -> str:
    try:
        import pytesseract
    except Exception:
        return ""
    g = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    txt = pytesseract.image_to_string(g, lang="eng", config="--psm 7")
    return txt.strip()

# ==== [PATCH 1/3: bbox_iou を完全置換] ======================================
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

    # 面積（必ず正）
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / (union + 1e-9)
# ==== [PATCH 1/3 END] =======================================================


def match_by_iou(G: List[Dict[str,Any]], D: List[Dict[str,Any]], thr: float) -> List[Tuple[int,int,float]]:
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
# ＝＝＝＝＝＝＝＝（文脈：この少し上に bbox_iou, match_by_iou などが並んでいます）＝＝＝＝＝＝＝＝

def det_metrics(G_all: List[Dict[str,Any]], D_all: List[Dict[str,Any]], iou_thr: float, img_wh: Tuple[int,int]) -> Dict[str, Any]:
    """
    G_all: X 側（基準）の検出（クラス名, bbox）
    D_all: F(X) 側の検出
    - 保持再現率/適合率/F1/HAL
    - 幾何安定性（中心誤差/サイズ比/IoU 中央）
    - カウント整合性（平均絶対差 / 擬EMD）
    """
    W, H = img_wh
    classes = sorted(list({g["cls"] for g in G_all} | {d["cls"] for d in D_all}))
    per_cls: Dict[str, Dict[str, float]] = {}
    center_errs=[]; size_ratio=[]; iou_list=[]

    # クラス別指標
    for c in classes:
        G = [g for g in G_all if g["cls"]==c]
        D = [d for d in D_all if d["cls"]==c]
        M = match_by_iou(G, D, iou_thr)

        Gc, Dc = len(G), len(D)
        Mc = len(M)
        PR = Mc/(Gc+1e-9)     # Preservation-Recall
        PP = Mc/(Dc+1e-9)     # Preservation-Precision
        F1 = 2*PR*PP/(PR+PP+1e-9)
        HAL = (Dc - Mc)/(Dc+1e-9)  # ハルシネーション率

        # 幾何安定性
        c_err=[]; s_err=[]; ious=[]
        for (i,j,ij_iou) in M:
            gx1,gy1,gx2,gy2 = G[i]["bbox"]; dx1,dy1,dx2,dy2 = D[j]["bbox"]
            gcx,gcy = (gx1+gx2)/2,(gy1+gy2)/2
            dcx,dcy = (dx1+dx2)/2,(dy1+dy2)/2
            cdist = math.sqrt(((gcx-dcx)/W)**2 + ((gcy-dcy)/H)**2)  # 画像サイズ正規化
            garea = max(1e-6,(gx2-gx1)*(gy2-gy1)); darea=max(1e-6,(dx2-dx1)*(dy2-dy1))
            sratio = abs(math.log(darea/garea))
            c_err.append(cdist); s_err.append(sratio); ious.append(ij_iou)

        center_errs+=c_err; size_ratio+=s_err; iou_list+=ious
        per_cls[c] = {"PR":float(PR), "PP":float(PP), "F1":float(F1), "HAL":float(HAL),
                      "count_G":Gc, "count_D":Dc}

    # カウント整合性
    cntG = np.array([sum(1 for g in G_all if g["cls"]==c) for c in classes], dtype=np.float64)
    cntD = np.array([sum(1 for d in D_all if d["cls"]==c) for c in classes], dtype=np.float64)
    abs_diff = float(np.mean(np.abs(cntG - cntD)))

    # 擬EMD（両ゼロ/片ゼロケースに安全対応）
    x = np.arange(len(classes), dtype=np.float64)
    sumG = float(cntG.sum())
    sumD = float(cntD.sum())

    if sumG <= 0.0 and sumD <= 0.0:
        # 両側ともカウントが 0 → 分布差は 0 と見なす
        emd = 0.0
    elif sumG > 0.0 and sumD > 0.0:
        # どちらも有効 → SciPy の EMD（Wasserstein）を使用
        wG = cntG / sumG
        wD = cntD / sumD
        emd = float(wasserstein_distance(x, x, u_weights=wG, v_weights=wD))
    else:
        # 片側のみゼロ → 累積分布差分（CDF の L1）を擬似 EMD として使用
        wG = cntG / (sumG + 1e-12)  # 片側ゼロだと片方は全 0 になる
        wD = cntD / (sumD + 1e-12)
        cdfG = np.cumsum(wG)
        cdfD = np.cumsum(wD)
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

# ＝＝＝＝＝＝＝＝（文脈：この少し下に parse_args / main などが続きます）＝＝＝＝＝＝＝＝



# ========== メイン評価ルーチン ==========
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Uni-ControlNet: WaymoV2 定量評価スクリプト（単一スクリプト完結）")
    ap.add_argument("--orig-root", type=str, default=DEFAULT_ORIG_IMAGE_ROOT)
    ap.add_argument("--gen-root", type=str, default=DEFAULT_GEN_ROOT)
    ap.add_argument("--canny-x-root", type=str, default=DEFAULT_CANNY_ROOT_X)
    ap.add_argument("--depth-x-npy-root", type=str, default=DEFAULT_DEPTH_NPY_ROOT_X)
    ap.add_argument("--semseg-x-root", type=str, default=DEFAULT_SEMSEG_ROOT_X)
    ap.add_argument("--cache-root", type=str, default=DEFAULT_HDD_CACHE_ROOT)
    ap.add_argument("--splits", type=str, nargs="+", default=["training","validation","testing"])
    ap.add_argument("--camera", type=str, default="front")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--tasks", type=str, choices=["all","reality","structure","objects"], default="all")
    ap.add_argument("--reality-metric", type=str, choices=["clip-cmmd","clip-fid","inception-fid"], default="clip-cmmd")
    ap.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_ID)
    ap.add_argument("--clip-batch", type=int, default=16)
    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument("--metric3d-onnx", type=str, default=DEFAULT_METRIC3D_ONNX)
    ap.add_argument("--use-yolop", action="store_true")
    ap.add_argument("--gdinomodel", type=str, default=DEFAULT_GDINO_ID)
    ap.add_argument("--det-prompts", type=str, nargs="*", default=DEFAULT_DET_PROMPTS)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--ocr-engine", type=str, choices=["none","tesseract"], default="tesseract")
    ap.add_argument("--annotation-mode", type=str, choices=["off","objects","structure","all"], default="off",
                    help="可視化注釈を保存: objects=検出可視化, structure=Edge/Depth/Semseg 可視化, all=両方, off=なし")
    ap.add_argument("--annotate-limit", type=int, default=32, help="注釈を保存する最大枚数（split毎の上限）")
    ap.add_argument("--annotate-out", type=str, default=os.path.join(DEFAULT_HDD_CACHE_ROOT, "viz"),
                    help="注釈画像の保存先（HDD）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb", action="store_true")
    ap.add_argument("--tb-dir", type=str, default=os.path.join(DEFAULT_HDD_CACHE_ROOT, "tensorboard"))
    ap.add_argument("--verbose", action="store_true")
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

    clipper = None
    if args.tasks in ("all","reality"):
        clipper = ClipEmbedder(args.clip_model)

    sess_m3d = None; in_m3d = None; out_m3d = None
    if args.tasks in ("all","structure","objects"):
        try:
            sess_m3d, in_m3d, out_m3d, providers = build_metric3d_session(args.metric3d_onnx)
            logger.info("Metric3Dv2 ORT providers: %s", providers)
        except Exception as e:
            logger.error("Metric3Dv2 ONNX Runtime 構築失敗: %s", repr(e))
            sys.exit(1)

    onef_proc = None; onef_model = None
    if args.tasks in ("all","structure"):
        onef_proc, onef_model = build_oneformer(DEFAULT_ONEFORMER_ID, device="cuda", fp16=True)
        logger.info("OneFormer loaded: %s", DEFAULT_ONEFORMER_ID)

    yolop_model = None
    if args.tasks in ("all","objects") and args.use_yolop:
        yolop_model = load_yolop(logger)

    gdino = None
    if args.tasks in ("all","objects"):
        gdino = GroundingDINO(args.gdinomodel); gdino.load(logger)

    ann_counter = {"objects": defaultdict(int), "structure": defaultdict(int)}
    total_pairs = 0

    for split in args.splits:
        pairs = enumerate_pairs(args.orig_root, args.gen_root, split, args.camera, args.limit)
        if not pairs:
            logger.warning("[%s] ペアが見つかりません（split=%s, camera=%s）", split, split, args.camera)
            continue
        logger.info("[%s] 評価ペア数: %d", split, len(pairs))
        total_pairs += len(pairs)

        # ---- リアリティ ----
        feats_x = []; feats_fx = []
        if args.tasks in ("all","reality"):
            clipper.load(logger)
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
                if len(batch_imgs_x) >= args.clip_batch:
                    fx = clipper.embed_batch(batch_imgs_x); ff = clipper.embed_batch(batch_imgs_fx)
                    for (rel_dir2, stem2), ex, ef in zip(batch_indices, fx, ff):
                        ensure_dir(os.path.dirname(cache.clip_path(split, rel_dir2, stem2, "x")))
                        np.savez(cache.clip_path(split, rel_dir2, stem2, "x"), feat=ex)
                        np.savez(cache.clip_path(split, rel_dir2, stem2, "fx"), feat=ef)
                    feats_x.append(fx); feats_fx.append(ff)
                    batch_imgs_x=[]; batch_imgs_fx=[]; batch_indices=[]
                    pbar.set_postfix_str("embed")
            if batch_imgs_x:
                fx = clipper.embed_batch(batch_imgs_x); ff = clipper.embed_batch(batch_imgs_fx)
                for (rel_dir2, stem2), ex, ef in zip(batch_indices, fx, ff):
                    ensure_dir(os.path.dirname(cache.clip_path(split, rel_dir2, stem2, "x")))
                    np.savez(cache.clip_path(split, rel_dir2, stem2, "x"), feat=ex)
                    np.savez(cache.clip_path(split, rel_dir2, stem2, "fx"), feat=ef)
                feats_x.append(fx); feats_fx.append(ff)
            feats_x = np.concatenate(feats_x, axis=0)
            feats_fx = np.concatenate(feats_fx, axis=0)

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

        # ---- 構造・意味忠実度 ----
        if args.tasks in ("all","structure"):
            conf_mat = np.zeros((19,19), dtype=np.int64)
            edge_scores = []; depth_scores = []
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
                depth_x = np.load(depth_x_path).astype(np.float32)
                depth_fx_path = cache.depth_fx_path(split, rel_dir, stem)
                if os.path.exists(depth_fx_path):
                    depth_fx = np.load(depth_fx_path).astype(np.float32)
                else:
                    rgbf = imread_rgb(pfx)
                    depth_fx = infer_metric3d_np(sess_m3d, in_m3d, out_m3d, rgbf)
                    ensure_dir(os.path.dirname(depth_fx_path)); np.save(depth_fx_path, depth_fx)
                d_met = depth_metrics(depth_x, depth_fx); depth_scores.append(d_met)

                # Semseg
                semseg_x_path = os.path.join(args.semseg_x_root, split, rel_dir, f"{stem}_predTrainId.npy")
                if not os.path.exists(semseg_x_path):
                    pbar.set_postfix_str("miss-segX"); continue
                seg_x = np.load(semseg_x_path).astype(np.uint8)
                seg_fx_path = cache.semseg_fx_path(split, rel_dir, stem)
                if os.path.exists(seg_fx_path):
                    seg_fx = np.load(seg_fx_path).astype(np.uint8)
                else:
                    rgbf = imread_rgb(pfx)
                    seg_fx = oneformer_semseg(onef_proc, onef_model, rgbf)
                    ensure_dir(os.path.dirname(seg_fx_path)); np.save(seg_fx_path, seg_fx)

                if seg_x.shape != seg_fx.shape:
                    seg_x = cv2.resize(seg_x, (seg_fx.shape[1], seg_fx.shape[0]), interpolation=cv2.INTER_NEAREST)

                conf_mat += confusion_19(seg_x, seg_fx, ncls=19)

                if args.annotation_mode in ("structure","all"):
                    if ann_counter["structure"][split] < args.annotate_limit:
                        rgbf_vis = imread_rgb(pfx)
                        save_annotations_structure(args.annotate_out, split, rel_dir, stem, rgbf_vis, edge_fx, depth_fx, seg_fx)
                        ann_counter["structure"][split] += 1
                pbar.set_postfix_str("ok")

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

        # ---- 物体保持・ハルシネーション ----
        if args.tasks in ("all","objects"):
            if gdino is None:
                logger.error("Grounding DINO 未初期化"); sys.exit(1)
            per_image_results = []
            pbar = tqdm(pairs, desc=f"{split}-obj")
            for (px, pfx, rel_dir, stem) in pbar:
                rgbx = imread_rgb(px); rgbf = imread_rgb(pfx)
                H,W = rgbx.shape[:2]

                gd_x_path = cache.gdino_path(split, rel_dir, stem, "x")
                gd_f_path = cache.gdino_path(split, rel_dir, stem, "fx")
                if os.path.exists(gd_x_path):
                    G = json.load(open(gd_x_path,"r"))
                else:
                    G = gdino.detect(rgbx, args.det_prompts, box_thr=0.35, txt_thr=0.25)
                    ensure_dir(os.path.dirname(gd_x_path)); json.dump(G, open(gd_x_path,"w"), indent=2)
                if os.path.exists(gd_f_path):
                    D = json.load(open(gd_f_path,"r"))
                else:
                    D = gdino.detect(rgbf, args.det_prompts, box_thr=0.35, txt_thr=0.25)
                    ensure_dir(os.path.dirname(gd_f_path)); json.dump(D, open(gd_f_path,"w"), indent=2)

                drv_x = None; drv_f = None
                if args.use_yolop and yolop_model is not None:
                    yx_path = cache.yolo_path(split, rel_dir, stem, "x")
                    yf_path = cache.yolo_path(split, rel_dir, stem, "fx")
                    if os.path.exists(yx_path):
                        yx = json.load(open(yx_path,"r")); drv_x = np.array(yx["drivable"], dtype=np.uint8)
                    else:
                        yx_raw = yolop_infer(yolop_model, rgbx); drv_x = (yx_raw["drivable"]>0.5).astype(np.uint8)
                        ensure_dir(os.path.dirname(yx_path)); json.dump({"drivable": drv_x.tolist()}, open(yx_path,"w"))
                    if os.path.exists(yf_path):
                        yf = json.load(open(yf_path,"r")); drv_f = np.array(yf["drivable"], dtype=np.uint8)
                    else:
                        yf_raw = yolop_infer(yolop_model, rgbf); drv_f = (yf_raw["drivable"]>0.5).astype(np.uint8)
                        ensure_dir(os.path.dirname(yf_path)); json.dump({"drivable": drv_f.tolist()}, open(yf_path,"w"))
                    def keep_roi(det, drv):
                        kept=[]; h_, w_ = drv.shape[:2]
                        for d in det:
                            x1,y1,x2,y2 = d["bbox"]
                            cx,cy = int((x1+x2)/2), int((y1+y2)/2)
                            cx = int(np.clip(cx, 0, w_-1)); cy = int(np.clip(cy, 0, h_-1))
                            if drv[cy, cx]>0: kept.append(d)
                        return kept
                    G = keep_roi(G, drv_x); D = keep_roi(D, drv_f)

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
                            crop = rgbf[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                            ocrF.append({"bbox":d["bbox"], "txt":run_ocr_tesseract(crop)})

                    # 修正: OCR 書き出し前にディレクトリ作成
                    ocr_x_out = cache.ocr_path(split, rel_dir, stem, "x")
                    ocr_f_out = cache.ocr_path(split, rel_dir, stem, "f")
                    ensure_dir(os.path.dirname(ocr_x_out))
                    ensure_dir(os.path.dirname(ocr_f_out))
                    with open(ocr_x_out, "w") as fx:
                        json.dump(ocrX, fx, indent=2)
                    with open(ocr_f_out, "w") as ff:
                        json.dump(ocrF, ff, indent=2)

                dm = det_metrics(G, D, iou_thr=args.iou_thr, img_wh=(W,H))
                per_image_results.append(dm)

                if args.annotation_mode in ("objects","all"):
                    if ann_counter["objects"][split] < args.annotate_limit:
                        save_annotations_objects(args.annotate_out, split, rel_dir, stem, rgbx, rgbf, G, D, drv_x, drv_f)
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

    logger.info("✅ 全 split 完了。総ペア数: %d | キャッシュ: %s", total_pairs, args.cache_root)
    if sw: sw.close()

if __name__ == "__main__":
    main()

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


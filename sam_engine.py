"""
SAM (Segment Anything Model) inference engine.
Ref: https://github.com/facebookresearch/segment-anything
"""

import os
import logging
import urllib.request
from pathlib import Path
from threading import Lock

import numpy as np
import cv2
import torch

log = logging.getLogger(__name__)

# ── Checkpoint registry (not hard-coded to one model) ─────────────
CHECKPOINTS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
    },
}

# Read from env, default to vit_b
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")
SAM_CHECKPOINT_DIR = os.environ.get("SAM_CHECKPOINT_DIR", "checkpoints")


class SAMEngine:
    """Lazy-loaded, thread-safe SAM wrapper with embedding cache."""

    def __init__(self, model_type: str = None, checkpoint_dir: str = None,
                 cache_size: int = 20):
        self.model_type = model_type or SAM_MODEL_TYPE
        self.checkpoint_dir = Path(checkpoint_dir or SAM_CHECKPOINT_DIR)
        self.cache_size = cache_size

        self._model = None
        self._predictor = None
        self._lock = Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # LRU embedding cache: { cache_key: (embedding, original_size, input_size) }
        self._embed_cache = {}
        self._embed_order = []  # oldest first

    # ── Lazy model loading ────────────────────────────────
    def _ensure_model(self):
        if self._model is not None:
            return
        ckpt = CHECKPOINTS.get(self.model_type)
        if not ckpt:
            raise ValueError(
                f"Unknown SAM model type '{self.model_type}'. "
                f"Available: {list(CHECKPOINTS.keys())}"
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / ckpt["filename"]

        if not ckpt_path.exists():
            log.info("Downloading SAM checkpoint %s (~375 MB for vit_b) ...",
                     ckpt["filename"])
            urllib.request.urlretrieve(ckpt["url"], str(ckpt_path))
            log.info("Download complete: %s", ckpt_path)

        # Import here so startup doesn't fail if segment_anything
        # is not installed but SAM is never used.
        from segment_anything import sam_model_registry, SamPredictor

        log.info("Loading SAM model '%s' on %s ...", self.model_type, self._device)
        sam = sam_model_registry[self.model_type](checkpoint=ckpt_path).cuda()
        sam.to(self._device)
        sam.eval()
        self._model = sam
        self._predictor = SamPredictor(sam)
        log.info("SAM model ready.")

    # ── Embedding ─────────────────────────────────────────
    def compute_embedding(self, image_bgr: np.ndarray, cache_key: str):
        """Compute and cache the image embedding.  Returns the cache_key."""
        with self._lock:
            if cache_key in self._embed_cache:
                return cache_key
            self._ensure_model()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with self._lock:
            self._predictor.set_image(image_rgb)
            embedding = self._predictor.get_image_embedding()
            original_size = self._predictor.original_size
            input_size = self._predictor.input_size

            # Store in cache
            self._embed_cache[cache_key] = {
                "embedding": embedding,
                "original_size": original_size,
                "input_size": input_size,
            }
            self._embed_order.append(cache_key)

            # Evict oldest if over limit
            while len(self._embed_order) > self.cache_size:
                old = self._embed_order.pop(0)
                self._embed_cache.pop(old, None)

        return cache_key

    # ── Prediction ────────────────────────────────────────
    def predict(self, cache_key: str,
                points: list, point_labels: list,
                simplify_tolerance: float = 1.5,
                min_area: int = 100):
        """
        Given cached embedding + prompt points, return polygon(s).

        Args:
            cache_key:  key returned by compute_embedding
            points:     list of [x, y] in pixel coords
            point_labels: list of 1 (foreground) / 0 (background)
            simplify_tolerance: cv2.approxPolyDP epsilon
            min_area: ignore contour regions smaller than this

        Returns:
            list of polygons, each polygon is list of [x, y] pixel coords.
            Best mask (highest predicted IoU) is returned.
        """
        with self._lock:
            cached = self._embed_cache.get(cache_key)
            if cached is None:
                raise KeyError(f"No cached embedding for '{cache_key}'")

            self._ensure_model()
            # Restore predictor state from cache
            self._predictor.features = cached["embedding"]
            self._predictor.original_size = cached["original_size"]
            self._predictor.input_size = cached["input_size"]
            self._predictor.is_image_set = True

            pts = np.array(points, dtype=np.float32)
            lbls = np.array(point_labels, dtype=np.int32)

            masks, scores, _ = self._predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                multimask_output=True,
            )

        # Pick the mask with highest score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)

        # Mask → polygon via contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            epsilon = simplify_tolerance
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                poly = approx.squeeze().tolist()
                # Ensure list-of-lists
                if isinstance(poly[0], int):
                    poly = [poly]
                polygons.append(poly)

        return polygons, float(scores[best_idx])

    @property
    def available(self) -> bool:
        """Check if SAM can be used (dependencies installed)."""
        try:
            import segment_anything  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def device(self) -> str:
        return self._device


# Module-level singleton
sam_engine = SAMEngine()

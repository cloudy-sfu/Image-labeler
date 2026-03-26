"""
Flask-based image annotation tool with YOLO-format output.
Supports: classification, detection, instance segmentation,
semantic segmentation, skeleton/pose annotation.
Ref: https://docs.ultralytics.com/datasets/
"""
import hashlib
import json
import os
import socket
import sys
import uuid
import webbrowser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cart_product
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from flask import (
    Flask, render_template, request, jsonify,
    send_file, redirect, url_for, abort
)

from sam_engine import sam_engine

executor = ThreadPoolExecutor(max_workers=2)
sam_embed_status = {}
settings_prune_status = {}
app = Flask(__name__)
app.secret_key = "labeler-secret"

PROJECTS_FILE = Path("projects.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
IGNORE_FILES = {".DS_Store", "desktop.ini", "Thumbs.db", ".gitkeep"}

# Front-end display names for tasks
TASK_DISPLAY = {
    "classification": "Classification",
    "detection": "Object detection",
    "instance_segmentation": "Instance segmentation (polygon)",
    "semantic_segmentation": "Semantic segmentation (mask)",
    "skeleton": "Skeleton (pose) estimation",
}
TASK_KEYS = list(TASK_DISPLAY.keys())


# ── helpers ──────────────────────────────────────────────────────────────────

def _next_unused_id(used: set[int]) -> int:
    i = 0
    while i in used:
        i += 1
    return i


def _class_key_from_pairs(pairs: list[tuple[int, int]]) -> str:
    return "|".join(f"{gid}:{lid}" for gid, lid in pairs)


def _normalize_label_groups(new_groups: list[dict], old_groups: list[dict] | None = None) -> list[dict]:
    """Normalize groups/labels and preserve stable IDs using explicit IDs, then index fallback."""
    old_groups = old_groups or []
    old_groups_clean = []
    for i, g in enumerate(old_groups):
        gid = g.get("group_id", i)
        labels = []
        for j, l in enumerate(g.get("labels", [])):
            labels.append({
                "label_id": l.get("label_id", j),
                "name": l.get("name", ""),
                "description": l.get("description", ""),
            })
        old_groups_clean.append({
            "group_id": gid,
            "name": g.get("name", ""),
            "labels": labels,
        })

    used_group_ids = {int(g["group_id"]) for g in old_groups_clean if isinstance(g.get("group_id"), int)}
    used_old_group_indexes = set()
    out = []

    for gi, g in enumerate(new_groups or []):
        name = str(g.get("name", "")).strip()
        raw_labels = g.get("labels", []) or []
        old_group = None

        incoming_gid = g.get("group_id")
        if isinstance(incoming_gid, int):
            for oi, og in enumerate(old_groups_clean):
                if oi in used_old_group_indexes:
                    continue
                if og["group_id"] == incoming_gid:
                    old_group = og
                    used_old_group_indexes.add(oi)
                    break

        if old_group is None and gi < len(old_groups_clean) and gi not in used_old_group_indexes:
            old_group = old_groups_clean[gi]
            used_old_group_indexes.add(gi)

        if old_group is None:
            matches = [
                (oi, og) for oi, og in enumerate(old_groups_clean)
                if oi not in used_old_group_indexes and og.get("name") == name
            ]
            if len(matches) == 1:
                oi, old_group = matches[0]
                used_old_group_indexes.add(oi)

        if old_group is not None:
            gid = int(old_group["group_id"])
        else:
            gid = _next_unused_id(used_group_ids)
            used_group_ids.add(gid)

        old_labels = old_group.get("labels", []) if old_group else []
        used_old_label_indexes = set()
        used_label_ids = {int(l["label_id"]) for l in old_labels if isinstance(l.get("label_id"), int)}
        labels_out = []

        for li, l in enumerate(raw_labels):
            lname = str(l.get("name", "")).strip()
            ldesc = str(l.get("description", "") or "").strip()
            old_label = None

            incoming_lid = l.get("label_id")
            if isinstance(incoming_lid, int):
                for oi, ol in enumerate(old_labels):
                    if oi in used_old_label_indexes:
                        continue
                    if ol["label_id"] == incoming_lid:
                        old_label = ol
                        used_old_label_indexes.add(oi)
                        break

            if old_label is None and li < len(old_labels) and li not in used_old_label_indexes:
                old_label = old_labels[li]
                used_old_label_indexes.add(li)

            if old_label is None:
                matches = [
                    (oi, ol) for oi, ol in enumerate(old_labels)
                    if oi not in used_old_label_indexes and ol.get("name") == lname
                ]
                if len(matches) == 1:
                    oi, old_label = matches[0]
                    used_old_label_indexes.add(oi)

            if old_label is not None:
                lid = int(old_label["label_id"])
            else:
                lid = _next_unused_id(used_label_ids)
                used_label_ids.add(lid)

            labels_out.append({"label_id": lid, "name": lname, "description": ldesc})

        out.append({"group_id": gid, "name": name, "labels": labels_out})

    return out


def _build_class_catalog(label_groups: list[dict], old_catalog: list[dict] | None = None) -> list[dict]:
    old_catalog = old_catalog or []
    old_key_to_id = {}
    used_ids = set()
    for e in old_catalog:
        try:
            cid = int(e.get("class_id"))
        except (TypeError, ValueError):
            continue
        key = e.get("key")
        if not key:
            continue
        old_key_to_id[key] = cid
        used_ids.add(cid)

    dimensions = []
    for g in label_groups:
        rows = []
        for l in g.get("labels", []):
            rows.append((int(g.get("group_id", 0)), int(l.get("label_id", 0)), g.get("name", ""), l.get("name", "")))
        if not rows:
            return []
        dimensions.append(rows)

    out = []
    kept_ids = set()
    for combo in cart_product(*dimensions):
        pairs = [(gid, lid) for gid, lid, _gn, _ln in combo]
        key = _class_key_from_pairs(pairs)
        labels = {gn: ln for _gid, _lid, gn, ln in combo}
        if key in old_key_to_id:
            cid = old_key_to_id[key]
        else:
            cid = _next_unused_id(used_ids | kept_ids)
        kept_ids.add(cid)
        out.append({"class_id": cid, "key": key, "labels": labels})

    out.sort(key=lambda e: int(e["class_id"]))
    return out


def _upgrade_project_schema(project: dict) -> bool:
    """Backfill stable IDs for groups/labels/classes on older projects."""
    changed = False
    normalized_groups = _normalize_label_groups(project.get("label_groups", []), project.get("label_groups", []))
    if normalized_groups != project.get("label_groups", []):
        project["label_groups"] = normalized_groups
        changed = True

    rebuilt_catalog = _build_class_catalog(project.get("label_groups", []), project.get("class_catalog", []))
    if rebuilt_catalog != project.get("class_catalog", []):
        project["class_catalog"] = rebuilt_catalog
        changed = True
    return changed


def _label_lookup_maps(project: dict) -> tuple[list[dict], dict[str, int], dict[int, dict]]:
    label_groups = project.get("label_groups", [])
    dims = []
    for g in label_groups:
        gm = {}
        for l in g.get("labels", []):
            gm[l.get("name", "")] = int(l.get("label_id", 0))
        dims.append({
            "group_id": int(g.get("group_id", 0)),
            "group_name": g.get("name", ""),
            "name_to_label_id": gm,
        })

    key_to_cid = {}
    cid_to_labels = {}
    for e in project.get("class_catalog", []):
        if not e.get("key"):
            continue
        key_to_cid[e["key"]] = int(e["class_id"])
        cid_to_labels[int(e["class_id"])] = e.get("labels", {})
    return dims, key_to_cid, cid_to_labels


def _default_class_id(project: dict) -> int:
    cats = project.get("class_catalog", [])
    if not cats:
        return 0
    return min(int(c.get("class_id", 0)) for c in cats)


def load_projects() -> dict:
    if not PROJECTS_FILE.exists():
        return {}

    projects = json.loads(PROJECTS_FILE.read_text())
    changed = False
    for p in projects.values():
        changed = _upgrade_project_schema(p) or changed

    if changed:
        save_projects(projects)
    return projects


def save_projects(projects: dict):
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2))


def image_files(image_dir: str) -> list[str]:
    out = []
    for f in sorted(os.listdir(image_dir)):
        if f in IGNORE_FILES:
            continue
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
            out.append(f)
    return out


def composite_classes(project: dict) -> dict[int, dict]:
    out = {}
    for e in project.get("class_catalog", []):
        out[int(e.get("class_id", 0))] = e.get("labels", {})
    return out


def labels_to_cid(labels: dict, project: dict) -> int:
    dimensions, key_to_cid, _cid_to_labels_map = _label_lookup_maps(project)
    pairs = []
    for d in dimensions:
        gname = d["group_name"]
        lname = labels.get(gname)
        if lname not in d["name_to_label_id"]:
            return _default_class_id(project)
        pairs.append((d["group_id"], d["name_to_label_id"][lname]))
    key = _class_key_from_pairs(pairs)
    if key in key_to_cid:
        return key_to_cid[key]
    return _default_class_id(project)


def cid_to_labels(cid: int, project: dict) -> dict:
    catalog = composite_classes(project)
    return catalog.get(cid, {})


def annotation_path(project: dict, image_name: str, mkdir=False) -> Path:
    ann_dir = Path(project["annotation_dir"])
    stem = Path(image_name).stem
    d = ann_dir / "labels"
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    suffix = ".json" if project.get("task") == "semantic_segmentation" else ".txt"
    return d / (stem + suffix)


def semantic_mask_dir(project: dict, mkdir=False) -> Path:
    d = Path(project["annotation_dir"]) / "masks"
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _poly_norm_to_pixels(poly: list[list[float]], width: int, height: int) -> list[tuple[float, float]]:
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    out = []
    for x, y in poly:
        px = min(max(float(x), 0.0), 1.0) * max_x
        py = min(max(float(y), 0.0), 1.0) * max_y
        out.append((px, py))
    return out


def _polygon_bbox(points: list[tuple[float, float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def semantic_mask_file(mask_dir: Path, image_stem: str, class_id: int) -> Path:
    return mask_dir / f"{image_stem}__cid_{class_id}.npy"


def build_semantic_artifacts(project: dict, image_name: str, annotations: list[dict]) -> tuple[dict[int, np.ndarray], dict, list[dict]]:
    """Build per-class semantic masks + COCO-like sidecar, enforcing semantic constraints."""
    img_path = Path(project["image_dir"]) / image_name
    with Image.open(img_path) as im:
        width, height = im.size

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")

    total_pixels = width * height
    coverage = np.zeros((height, width), dtype=bool)
    class_ids = np.full((height, width), fill_value=-1, dtype=np.int32)

    by_class = defaultdict(list)
    class_area = defaultdict(int)
    class_bbox = {}
    kept_annotations = []

    for ann_idx, ann in enumerate(annotations):
        poly = ann.get("polygon") or []
        if len(poly) < 3:
            raise ValueError(f"Polygon #{ann_idx + 1} has fewer than 3 points")

        cid = labels_to_cid(ann.get("labels", {}), project)
        px_poly = _poly_norm_to_pixels(poly, width, height)

        temp = Image.new("1", (width, height), 0)
        ImageDraw.Draw(temp).polygon(px_poly, fill=1)
        poly_mask = np.asarray(temp, dtype=bool)

        # Semantic retreat: existing (earlier) mask keeps ownership.
        new_pixels = poly_mask & ~coverage
        contributed_pixels = int(np.count_nonzero(new_pixels))

        if contributed_pixels <= 0:
            continue

        coverage |= new_pixels
        class_ids[new_pixels] = cid
        class_area[cid] += contributed_pixels

        by_class[cid].append(px_poly)
        bb = _polygon_bbox(px_poly)
        if cid not in class_bbox:
            class_bbox[cid] = bb
        else:
            x0 = min(class_bbox[cid][0], bb[0])
            y0 = min(class_bbox[cid][1], bb[1])
            x1 = max(class_bbox[cid][0] + class_bbox[cid][2], bb[0] + bb[2])
            y1 = max(class_bbox[cid][1] + class_bbox[cid][3], bb[1] + bb[3])
            class_bbox[cid] = [x0, y0, x1 - x0, y1 - y0]
        kept_annotations.append({
            "class_id": cid,
            "labels": ann.get("labels", {}),
            "polygon": ann.get("polygon", []),
        })

    covered = int(np.count_nonzero(coverage))
    uncovered = total_pixels - covered
    if uncovered:
        pct = (uncovered / total_pixels) * 100.0
        raise ValueError(f"Semantic masks must tile the image. Uncovered pixels: {uncovered} ({pct:.2f}%)")

    cats = composite_classes(project)
    sorted_class_ids = sorted(int(cid) for cid in cats.keys())
    class_masks = {cid: (class_ids == cid) for cid in sorted_class_ids}
    category_names = {cid: "_".join(v.values()) for cid, v in cats.items()}
    stem = Path(image_name).stem

    anns_out = []
    ann_id = 1
    for cid, polys in sorted(by_class.items()):
        anns_out.append({
            "id": ann_id,
            "category_id": cid,
            "category_name": category_names.get(cid, str(cid)),
            "iscrowd": 0,
            "area": class_area[cid],
            "bbox": [round(v, 2) for v in class_bbox.get(cid, [0, 0, 0, 0])],
            # COCO-style list of polygon rings in absolute pixel coordinates.
            "segmentation": [
                [round(coord, 2) for xy in poly for coord in xy]
                for poly in polys
            ],
        })
        ann_id += 1

    sidecar = {
        "format": "semantic_maskrcnn_v1",
        "image": {
            "file_name": image_name,
            "width": width,
            "height": height,
        },
        "categories": [
            {"id": cid, "name": category_names.get(cid, str(cid)), "labels": combo}
            for cid, combo in sorted(cats.items())
        ],
        "annotations": anns_out,
        "mask_encoding": "per_class_npy_hw",
        "mask_shape": [height, width],
        "mask_files": {
            str(cid): semantic_mask_file(Path("."), stem, cid).name
            for cid in sorted_class_ids
        },
    }
    return class_masks, sidecar, kept_annotations


def write_data_yaml(project: dict):
    ann_dir = Path(project["annotation_dir"])
    ann_dir.mkdir(parents=True, exist_ok=True)
    cmap = composite_classes(project)
    names = {i: "_".join(v.values()) for i, v in cmap.items()}
    content = {
        "path": project["image_dir"],
        "train": ".",
        "val": ".",
        "names": names,
        "nc": (max(names.keys()) + 1) if names else 0,
        "class_ids": sorted(names.keys()),
        "label_groups": project["label_groups"],
    }
    if project["task"] == "skeleton":
        content["kpt_shape"] = [len(project.get("keypoint_names", [])), 3]
        content["keypoint_names"] = project.get("keypoint_names", [])
        content["skeleton_edges"] = project.get("skeleton_edges", [])
    elif project["task"] == "semantic_segmentation":
        content["mask_dir"] = "masks"
        content["mask_format"] = "npy_per_class_hw"
    (ann_dir / "data.yaml").write_text(yaml.safe_dump(content, sort_keys=False))


def _prune_annotations_for_deleted_classes(project: dict, deleted_class_ids: set[int]) -> dict:
    """Delete references to removed classes in saved annotations and semantic mask files."""
    stats = {"files_rewritten": 0, "annotations_removed": 0, "images_touched": 0}
    if not deleted_class_ids:
        return stats

    task = project.get("task")
    for image_name in image_files(project["image_dir"]):
        ap = annotation_path(project, image_name)
        changed = False
        removed_here = 0

        if ap.exists():
            raw = ap.read_text().strip()

            if task == "classification":
                if raw:
                    cid = int(raw.split()[0])
                    if cid in deleted_class_ids:
                        ap.write_text("")
                        changed = True
                        removed_here = 1

            elif task == "semantic_segmentation":
                if raw:
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = None

                    if isinstance(payload, dict):
                        anns = payload.get("annotations", [])
                        kept = [a for a in anns if int(a.get("class_id", -1)) not in deleted_class_ids]
                        removed_here = len(anns) - len(kept)
                        if removed_here:
                            payload["annotations"] = kept
                            ap.write_text(json.dumps(payload, indent=2))
                            changed = True
                    else:
                        lines = [ln for ln in raw.splitlines() if ln.strip()]
                        kept_lines = []
                        for line in lines:
                            parts = line.split()
                            if not parts:
                                continue
                            if int(parts[0]) in deleted_class_ids:
                                removed_here += 1
                            else:
                                kept_lines.append(line)
                        if removed_here:
                            ap.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
                            changed = True

            else:
                if raw:
                    lines = [ln for ln in raw.splitlines() if ln.strip()]
                    kept_lines = []
                    for line in lines:
                        parts = line.split()
                        if not parts:
                            continue
                        if int(parts[0]) in deleted_class_ids:
                            removed_here += 1
                        else:
                            kept_lines.append(line)
                    if removed_here:
                        ap.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
                        changed = True

        if task == "semantic_segmentation":
            stem = Path(image_name).stem
            mask_dir = semantic_mask_dir(project, mkdir=True)
            legacy_npz = mask_dir / f"{stem}.npz"
            if legacy_npz.exists():
                legacy_npz.unlink(missing_ok=True)
                changed = True
            for cid in deleted_class_ids:
                mp = semantic_mask_file(mask_dir, stem, cid)
                if mp.exists():
                    mp.unlink(missing_ok=True)
                    changed = True

            sidecar_path = mask_dir / f"{stem}.coco.json"
            if sidecar_path.exists():
                try:
                    sidecar = json.loads(sidecar_path.read_text())
                except json.JSONDecodeError:
                    sidecar = None
                if isinstance(sidecar, dict):
                    anns = sidecar.get("annotations", [])
                    cats = sidecar.get("categories", [])
                    kept_anns = [a for a in anns if int(a.get("category_id", -1)) not in deleted_class_ids]
                    kept_cats = [c for c in cats if int(c.get("id", -1)) not in deleted_class_ids]
                    new_mask_files = {
                        k: v for k, v in (sidecar.get("mask_files") or {}).items()
                        if int(k) not in deleted_class_ids
                    }
                    if len(kept_anns) != len(anns) or len(kept_cats) != len(cats) or len(new_mask_files) != len(sidecar.get("mask_files") or {}):
                        sidecar["annotations"] = kept_anns
                        sidecar["categories"] = kept_cats
                        sidecar["mask_files"] = new_mask_files
                        sidecar_path.write_text(json.dumps(sidecar, indent=2))
                        changed = True

        if changed:
            stats["files_rewritten"] += 1
            stats["annotations_removed"] += removed_here
            stats["images_touched"] += 1

    return stats


def _run_settings_prune_job(job_id: str, project_snapshot: dict, deleted_class_ids: list[int]):
    settings_prune_status[job_id] = {
        "status": "running",
        "deleted_class_ids": deleted_class_ids,
        "stats": None,
        "error": None,
    }
    try:
        stats = _prune_annotations_for_deleted_classes(project_snapshot, set(deleted_class_ids))
        settings_prune_status[job_id] = {
            "status": "done",
            "deleted_class_ids": deleted_class_ids,
            "stats": stats,
            "error": None,
        }
    except Exception as e:
        settings_prune_status[job_id] = {
            "status": "error",
            "deleted_class_ids": deleted_class_ids,
            "stats": None,
            "error": str(e),
        }


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    projects = load_projects()
    for pid, p in projects.items():
        imgs = image_files(p["image_dir"]) if os.path.isdir(p["image_dir"]) else []
        labeled = sum(1 for f in imgs if annotation_path(p, f).exists())
        p["_total"] = len(imgs)
        p["_labeled"] = labeled
    return render_template("index.html", projects=projects, task_display=TASK_DISPLAY)


@app.route("/create", methods=["POST"])
def create_project():
    data = request.get_json()
    task = data.get("task")
    if task not in TASK_KEYS:
        return jsonify(error=f"Unsupported task: {task}"), 400
    pid = uuid.uuid4().hex[:12]
    raw_groups = data.get("label_groups", [])
    norm_groups = _normalize_label_groups(raw_groups, [])
    project = {
        "name": data["name"],
        "image_dir": data["image_dir"],
        "annotation_dir": data["annotation_dir"],
        "task": task,
        "label_groups": norm_groups,
        "class_catalog": _build_class_catalog(norm_groups, []),
        "keypoint_names": data.get("keypoint_names", []),
        "skeleton_edges": data.get("skeleton_edges", []),
    }
    if not os.path.isdir(project["image_dir"]):
        return jsonify(error="Image directory does not exist"), 400
    Path(project["annotation_dir"]).mkdir(parents=True, exist_ok=True)
    projects = load_projects()
    projects[pid] = project
    save_projects(projects)
    write_data_yaml(project)
    return jsonify(id=pid)


@app.route("/project/<pid>")
def project_home(pid):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    p = projects[pid]
    imgs = image_files(p["image_dir"])
    if not imgs:
        return "No images found in the directory.", 404
    return redirect(url_for("annotate_page", pid=pid, filename=imgs[0]))


@app.route("/project/<pid>/annotate/<filename>")
def annotate_page(pid, filename):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    p = projects[pid]
    imgs = image_files(p["image_dir"])
    idx = imgs.index(filename) if filename in imgs else 0
    return render_template(
        "annotate.html",
        project=p,
        pid=pid,
        images=imgs,
        current=filename,
        current_idx=idx,
        task_display=TASK_DISPLAY,
    )


@app.route("/api/project/<pid>/image/<filename>")
def serve_image(pid, filename):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    img_path = Path(projects[pid]["image_dir"]) / filename
    if not img_path.exists():
        abort(404)
    return send_file(img_path)


@app.route("/api/project/<pid>/image_size/<filename>")
def image_size(pid, filename):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    img_path = Path(projects[pid]["image_dir"]) / filename
    with Image.open(img_path) as im:
        w, h = im.size
    return jsonify(width=w, height=h)


@app.route("/api/project/<pid>/annotation/<filename>", methods=["GET"])
def get_annotation(pid, filename):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    p = projects[pid]
    ap = annotation_path(p, filename)
    if not ap.exists():
        return jsonify(annotations=[], mask=None)

    task = p["task"]
    if task == "classification":
        text = ap.read_text().strip()
        if text:
            cid = int(text.split()[0])
            return jsonify(annotations=[{"class_id": cid, "labels": cid_to_labels(cid, p)}])
        return jsonify(annotations=[])

    if task == "semantic_segmentation":
        raw = ap.read_text().strip()
        if not raw:
            return jsonify(annotations=[])
        try:
            payload = json.loads(raw)
            anns = payload.get("annotations", [])
            for ann in anns:
                ann.setdefault("labels", cid_to_labels(ann.get("class_id", 0), p))
            return jsonify(annotations=anns)
        except json.JSONDecodeError:
            # Backward compatibility for projects saved before semantic JSON format.
            anns = []
            for line in raw.splitlines():
                parts = line.split()
                if not parts:
                    continue
                cid = int(parts[0])
                coords = list(map(float, parts[1:]))
                points = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
                anns.append({"class_id": cid, "labels": cid_to_labels(cid, p), "polygon": points})
            return jsonify(annotations=anns)

    annotations_out = []
    for line in ap.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        labels = cid_to_labels(cid, p)
        if task == "detection":
            cx, cy, w, h = map(float, parts[1:5])
            annotations_out.append({"class_id": cid, "labels": labels,
                                    "bbox": [cx, cy, w, h]})
        elif task == "instance_segmentation":
            coords = list(map(float, parts[1:]))
            points = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            annotations_out.append({"class_id": cid, "labels": labels,
                                    "polygon": points})
        elif task == "skeleton":
            cx, cy, w, h = map(float, parts[1:5])
            kpts_flat = list(map(float, parts[5:]))
            kpts = []
            for i in range(0, len(kpts_flat), 3):
                kpts.append({"x": kpts_flat[i], "y": kpts_flat[i + 1],
                             "v": int(kpts_flat[i + 2])})
            annotations_out.append({"class_id": cid, "labels": labels,
                                    "bbox": [cx, cy, w, h], "keypoints": kpts})
    return jsonify(annotations=annotations_out)


@app.route("/api/project/<pid>/annotation/<filename>", methods=["POST"])
def save_annotation(pid, filename):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    p = projects[pid]
    data = request.get_json()
    task = p["task"]
    ap = annotation_path(p, filename, mkdir=True)


    if task == "classification":
        anns = data.get("annotations", [])
        if anns:
            cid = labels_to_cid(anns[0].get("labels", {}), p)
            ap.write_text(str(cid) + "\n")
        else:
            ap.write_text("")
        return jsonify(ok=True)

    if task == "semantic_segmentation":
        anns = data.get("annotations", [])
        try:
            class_masks, sidecar, kept_anns = build_semantic_artifacts(p, filename, anns)
        except ValueError as e:
            msg = str(e)
            code = "semantic_uncovered_pixels" if msg.startswith("Semantic masks must tile the image") else "semantic_invalid"
            return jsonify(error=msg, error_code=code), 400

        # Editor source-of-truth (normalized polygons + labels)
        editor_payload = {
            "format": "semantic_editor_v1",
            "annotations": kept_anns,
        }
        ap.write_text(json.dumps(editor_payload, indent=2))

        stem = Path(filename).stem
        mask_dir = semantic_mask_dir(p, mkdir=True)
        (mask_dir / f"{stem}.npz").unlink(missing_ok=True)
        for old_mask in mask_dir.glob(f"{stem}__cid_*.npy"):
            old_mask.unlink(missing_ok=True)
        for cid, mask in class_masks.items():
            np.save(semantic_mask_file(mask_dir, stem, int(cid)), mask.astype(np.uint8))
        (mask_dir / f"{stem}.coco.json").write_text(json.dumps(sidecar, indent=2))
        return jsonify(ok=True)

    lines = []
    for ann in data.get("annotations", []):
        cid = labels_to_cid(ann.get("labels", {}), p)
        if task == "detection":
            b = ann["bbox"]
            lines.append(f"{cid} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
        elif task == "instance_segmentation":
            pts = ann["polygon"]
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{cid} {coords}")
        elif task == "skeleton":
            b = ann["bbox"]
            kpts = ann["keypoints"]
            kstr = " ".join(f"{k['x']:.6f} {k['y']:.6f} {int(k['v'])}" for k in kpts)
            lines.append(f"{cid} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {kstr}")
    ap.write_text("\n".join(lines) + "\n" if lines else "")
    return jsonify(ok=True)


@app.route("/api/project/<pid>/delete", methods=["POST"])
def delete_project(pid):
    projects = load_projects()
    if pid in projects:
        del projects[pid]
        save_projects(projects)
    return jsonify(ok=True)


@app.route("/api/browse_folders")
def browse_folders():
    """Return directories under a given path for the folder picker.
    Special value '__HOME__' resolves to the current user's home directory."""
    root = request.args.get("path", "")

    # Resolve special token or empty to user home
    if not root or root == "__HOME__":
        root = str(Path.home())

    if not root:
        # Fallback for platforms where Path.home() might return empty
        if sys.platform == "win32":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                dp = f"{letter}:\\"
                if os.path.isdir(dp):
                    drives.append(dp)
            return jsonify(path="", dirs=drives, parent=None)
        else:
            root = "/"

    root = os.path.abspath(root)

    # If the given path is not a valid directory, walk up to nearest valid ancestor
    original = root
    while root and not os.path.isdir(root):
        parent = os.path.dirname(root)
        if parent == root:
            # Reached filesystem root and it's still not valid
            root = str(Path.home())
            break
        root = parent

    if not os.path.isdir(root):
        return jsonify(error="Cannot resolve directory"), 400

    dirs = []
    try:
        for entry in sorted(os.scandir(root), key=lambda e: e.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append(entry.name)
    except PermissionError:
        pass

    parent = os.path.dirname(root) if root != os.path.dirname(root) else None
    return jsonify(path=root, dirs=dirs, parent=parent)


@app.route("/api/create_folder", methods=["POST"])
def create_folder():
    """Create a new subfolder in the selected parent directory."""
    data = request.get_json() or {}
    parent_path = str(data.get("parent_path", "")).strip()
    folder_name = str(data.get("folder_name", "")).strip()

    if not parent_path:
        parent_path = str(Path.home())
    parent_path = os.path.abspath(parent_path)

    if not os.path.isdir(parent_path):
        return jsonify(error="Parent directory does not exist"), 400

    # Disallow path traversal or nested paths from the name field.
    if (not folder_name or folder_name in {".", ".."}
            or os.path.sep in folder_name
            or (os.path.altsep and os.path.altsep in folder_name)):
        return jsonify(error="Invalid folder name"), 400

    new_path = os.path.abspath(os.path.join(parent_path, folder_name))
    if os.path.commonpath([parent_path, new_path]) != parent_path:
        return jsonify(error="Invalid folder path"), 400

    try:
        Path(new_path).mkdir(parents=False, exist_ok=True)
    except OSError as e:
        return jsonify(error=f"Cannot create folder: {e}"), 400

    return jsonify(ok=True, path=new_path)


@app.route("/api/project/<pid>/settings", methods=["GET"])
def get_project_settings(pid):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    p = projects[pid]
    return jsonify(
        label_groups=p.get("label_groups", []),
        keypoint_names=p.get("keypoint_names", []),
        skeleton_edges=p.get("skeleton_edges", []),
    )


@app.route("/api/project/<pid>/settings", methods=["PUT"])
def update_project_settings(pid):
    """Update mutable project settings (label groups, skeleton config).
    Directory and task type are intentionally immutable."""
    projects = load_projects()
    if pid not in projects:
        abort(404)
    data = request.get_json() or {}
    p = projects[pid]

    old_groups = p.get("label_groups", [])
    old_catalog = p.get("class_catalog", [])
    new_groups = _normalize_label_groups(data.get("label_groups", []), old_groups)
    new_catalog = _build_class_catalog(new_groups, old_catalog)

    old_ids = {int(c.get("class_id", -1)) for c in old_catalog}
    new_ids = {int(c.get("class_id", -1)) for c in new_catalog}
    deleted_class_ids = {cid for cid in old_ids if cid not in new_ids and cid >= 0}
    confirmed = bool(data.get("confirm_deletion", False))

    removed = [f"class_id {cid}" for cid in sorted(deleted_class_ids)]

    # Preflight: require explicit confirmation before applying destructive class deletions.
    if deleted_class_ids and not confirmed:
        return jsonify(
            ok=False,
            requires_confirmation=True,
            removed=removed,
            deleted_class_ids=sorted(deleted_class_ids),
        )

    # Apply
    p["label_groups"] = new_groups
    p["class_catalog"] = new_catalog
    if p["task"] == "skeleton":
        if "keypoint_names" in data:
            p["keypoint_names"] = data["keypoint_names"]
        if "skeleton_edges" in data:
            p["skeleton_edges"] = data["skeleton_edges"]

    save_projects(projects)
    write_data_yaml(p)

    prune_job_id = None
    if deleted_class_ids:
        prune_job_id = uuid.uuid4().hex
        ids = sorted(deleted_class_ids)
        project_snapshot = json.loads(json.dumps(p))
        settings_prune_status[prune_job_id] = {
            "status": "pending",
            "deleted_class_ids": ids,
            "stats": None,
            "error": None,
        }
        executor.submit(_run_settings_prune_job, prune_job_id, project_snapshot, ids)

    return jsonify(
        ok=True,
        removed=removed,
        deleted_class_ids=sorted(deleted_class_ids),
        prune_job_id=prune_job_id,
        prune_status=(settings_prune_status.get(prune_job_id, {}).get("status") if prune_job_id else None),
    )


@app.route("/api/project/<pid>/settings/prune_status/<job_id>", methods=["GET"])
def settings_prune_job_status(pid, job_id):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    info = settings_prune_status.get(job_id)
    if info is None:
        return jsonify(error="Unknown prune job"), 404
    return jsonify(job_id=job_id, **info)


def find_available_port(start_port: int, tries: int = 100):
    """
    Find the first available port from {start_port} to {start_port + tries}
    :param start_port: The port that the program starts to scan, if it's occupied, the
    program will scan {start_port + 1}. If it's occupied again, try the next one...
    :param tries: Default 100, the maximum trying times from the start port.
    :return:
    """
    for i in range(tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", start_port + i))
            s.close()
            return start_port + i
        except OSError:
            pass
    raise Exception(f"Tried {tries} times, no available port from {start_port} to "
                    f"{start_port + tries}.")


# Ref: batch labeled-status endpoint to avoid N requests on page load
@app.route("/api/project/<pid>/labeled_status")
def labeled_status(pid):
    projects = load_projects()
    if pid not in projects:
        abort(404)
    p = projects[pid]
    imgs = image_files(p["image_dir"])
    result = {}
    for f in imgs:
        result[f] = annotation_path(p, f).exists()
    return jsonify(result)


@app.route("/api/sam/status")
def sam_status():
    """Report whether SAM is available."""
    return jsonify(
        available=sam_engine.available,
        device=sam_engine.device if sam_engine.available else None,
        model_type=sam_engine.model_type,
    )


def _compute_embedding_bg(pid, filename, img_path, cache_key):
    """Background worker for embedding computation."""
    try:
        import cv2
        image = cv2.imread(img_path)
        if image is None:
            sam_embed_status[cache_key] = "error"
            return
        sam_engine.compute_embedding(image, cache_key)
        sam_embed_status[cache_key] = "ready"
    except Exception as e:
        app.logger.error("SAM embedding error: %s", e)
        sam_embed_status[cache_key] = "error"


@app.route("/api/project/<pid>/sam/embed/<path:filename>", methods=["POST"])
def sam_embed(pid, filename):
    """Start async embedding computation for an image."""
    projects = load_projects()
    if pid not in projects:
        abort(404)
    p = projects[pid]
    img_path = os.path.join(p["image_dir"], filename)
    if not os.path.isfile(img_path):
        abort(404)
    # Cache key = hash of project id + filename + file mtime
    mtime = str(os.path.getmtime(img_path))
    cache_key = hashlib.md5(f"{pid}:{filename}:{mtime}".encode()).hexdigest()
    status = sam_embed_status.get(cache_key)
    if status == "ready" or cache_key in sam_engine._embed_cache:
        sam_embed_status[cache_key] = "ready"
        return jsonify(status="ready", cache_key=cache_key)
    if status == "pending":
        return jsonify(status="pending", cache_key=cache_key)
    sam_embed_status[cache_key] = "pending"
    executor.submit(_compute_embedding_bg, pid, filename, img_path, cache_key)
    return jsonify(status="pending", cache_key=cache_key)


@app.route("/api/project/<pid>/sam/embed_status/<cache_key>")
def sam_embed_status_check(pid, cache_key):
    """Poll embedding computation status."""
    status = sam_embed_status.get(cache_key, "unknown")
    if cache_key in sam_engine._embed_cache:
        status = "ready"
    return jsonify(status=status, cache_key=cache_key)


@app.route("/api/project/<pid>/sam/predict", methods=["POST"])
def sam_predict(pid):
    """Run SAM prediction given prompt points on a cached embedding."""
    data = request.get_json()
    cache_key = data.get("cache_key")
    points = data.get("points", [])          # [[x,y], ...]  pixel coords
    point_labels = data.get("point_labels", [])  # [1, 0, ...]
    if not cache_key or not points:
        return jsonify(error="cache_key and points required"), 400
    try:
        polygons, score = sam_engine.predict(
            cache_key, points, point_labels,
            simplify_tolerance=float(data.get("simplify", 1.5)),
        )
        return jsonify(polygons=polygons, score=score)
    except KeyError:
        return jsonify(error="Embedding not found. Re-embed the image."), 404
    except Exception as e:
        app.logger.error("SAM predict error: %s", e)
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    port = find_available_port(1024)
    webbrowser.open_new_tab(f'http://localhost:{port}')
    app.run(port=port)

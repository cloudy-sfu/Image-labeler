"""
Flask-based image annotation tool with YOLO-format output.
Supports: classification, detection, instance segmentation,
skeleton/pose annotation.
Ref: https://docs.ultralytics.com/datasets/
"""
import hashlib
import json
import os
import socket
import sys
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cart_product
from pathlib import Path

from PIL import Image
from flask import (
    Flask, render_template, request, jsonify,
    send_file, redirect, url_for, abort
)

from sam_engine import sam_engine

executor = ThreadPoolExecutor(max_workers=2)
sam_embed_status = {}
app = Flask(__name__)
app.secret_key = "labeler-secret"

PROJECTS_FILE = Path("projects.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
IGNORE_FILES = {".DS_Store", "desktop.ini", "Thumbs.db", ".gitkeep"}

# Front-end display names for tasks
TASK_DISPLAY = {
    "classification": "Image Classification",
    "detection": "Object Detection (Bounding Box)",
    "instance_segmentation": "Instance Segmentation (Polygon)",
    "skeleton": "Keypoint / Pose Estimation",
}
TASK_KEYS = list(TASK_DISPLAY.keys())


# ── helpers ──────────────────────────────────────────────────────────────────

def load_projects() -> dict:
    if PROJECTS_FILE.exists():
        return json.loads(PROJECTS_FILE.read_text())
    return {}


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


def composite_classes(label_groups: list[dict]) -> dict[int, dict]:
    """Cartesian product of label groups → {class_id: {group: label}}."""
    if not label_groups:
        return {}
    names = [g["name"] for g in label_groups]
    values = [g["labels"] for g in label_groups]
    return {
        i: dict(zip(names, combo))
        for i, combo in enumerate(cart_product(*[
            [lb["name"] for lb in v] for v in values
        ]))
    }


def labels_to_cid(labels: dict, label_groups: list[dict]) -> int:
    for cid, combo in composite_classes(label_groups).items():
        if combo == labels:
            return cid
    return 0


def cid_to_labels(cid: int, label_groups: list[dict]) -> dict:
    return composite_classes(label_groups).get(cid, {})


def annotation_path(project: dict, image_name: str, mkdir=False) -> Path:
    ann_dir = Path(project["annotation_dir"])
    stem = Path(image_name).stem
    d = ann_dir / "labels"
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d / (stem + ".txt")


def write_data_yaml(project: dict):
    ann_dir = Path(project["annotation_dir"])
    ann_dir.mkdir(parents=True, exist_ok=True)
    cmap = composite_classes(project["label_groups"])
    names = {i: "_".join(v.values()) for i, v in cmap.items()}
    content = {
        "path": project["image_dir"],
        "train": ".",
        "val": ".",
        "names": names,
        "nc": len(names),
        "label_groups": project["label_groups"],
    }
    if project["task"] == "skeleton":
        content["kpt_shape"] = [len(project.get("keypoint_names", [])), 3]
        content["keypoint_names"] = project.get("keypoint_names", [])
        content["skeleton_edges"] = project.get("skeleton_edges", [])
    (ann_dir / "data.yaml").write_text(json.dumps(content, indent=2))


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
    project = {
        "name": data["name"],
        "image_dir": data["image_dir"],
        "annotation_dir": data["annotation_dir"],
        "task": task,
        "label_groups": data.get("label_groups", []),
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
    lg = p["label_groups"]

    if task == "classification":
        text = ap.read_text().strip()
        if text:
            cid = int(text.split()[0])
            return jsonify(annotations=[{"class_id": cid, "labels": cid_to_labels(cid, lg)}])
        return jsonify(annotations=[])

    annotations_out = []
    for line in ap.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        labels = cid_to_labels(cid, lg)
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
    lg = p["label_groups"]
    ap = annotation_path(p, filename, mkdir=True)


    if task == "classification":
        anns = data.get("annotations", [])
        if anns:
            cid = labels_to_cid(anns[0].get("labels", {}), lg)
            ap.write_text(str(cid) + "\n")
        else:
            ap.write_text("")
        return jsonify(ok=True)

    lines = []
    for ann in data.get("annotations", []):
        cid = labels_to_cid(ann.get("labels", {}), lg)
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
    data = request.get_json()
    p = projects[pid]

    # ── Detect removed labels and warn caller ──
    old_groups = {g["name"]: {l["name"] for l in g["labels"]}
                  for g in p.get("label_groups", [])}
    new_groups_raw = data.get("label_groups", [])
    removed = []
    for g in p.get("label_groups", []):
        new_g = next((ng for ng in new_groups_raw if ng["name"] == g["name"]), None)
        if new_g is None:
            removed.append(f"group '{g['name']}' (entire group)")
        else:
            new_names = {l["name"] for l in new_g["labels"]}
            for l in g["labels"]:
                if l["name"] not in new_names:
                    removed.append(f"label '{l['name']}' in group '{g['name']}'")

    # Apply
    p["label_groups"] = new_groups_raw
    if p["task"] == "skeleton":
        if "keypoint_names" in data:
            p["keypoint_names"] = data["keypoint_names"]
        if "skeleton_edges" in data:
            p["skeleton_edges"] = data["skeleton_edges"]

    save_projects(projects)
    write_data_yaml(p)
    return jsonify(ok=True, removed=removed)


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

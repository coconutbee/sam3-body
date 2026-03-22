import argparse
import csv
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from notebook.utils import setup_sam_3d_body
    HAS_SAM3D = True
except ImportError:
    HAS_SAM3D = False

IDX_NOSE = 0
IDX_LEFT_EYE = 1
IDX_RIGHT_EYE = 2
IDX_LEFT_EAR = 3
IDX_RIGHT_EAR = 4
IDX_LEFT_SHOULDER_SAM = 5
IDX_RIGHT_SHOULDER_SAM = 6
IDX_NECK = 69

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

EPS = 1e-8
FACE_CONF_MIN = 0.03
SHOULDER_CONF_MIN = 0.03
PITCH_OFFSET_DEG = 25.0


def limit_angle(angle):
    while angle < -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def _safe_float(v):
    if v is None:
        return ""
    try:
        return float(v)
    except Exception:
        return ""


def _unit_vec(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def _is_valid_point(pt):
    if pt is None:
        return False
    arr = np.asarray(pt, dtype=np.float32)
    return arr.shape[0] >= 3 and np.all(np.isfinite(arr[:3]))


def _signed_angle_yaw_deg(vec_head, vec_body, eps=1e-8):
    head_xz = np.array([vec_head[0], vec_head[2]], dtype=np.float32)
    body_xz = np.array([vec_body[0], vec_body[2]], dtype=np.float32)

    norm_h = np.linalg.norm(head_xz)
    norm_b = np.linalg.norm(body_xz)
    if norm_h < eps or norm_b < eps:
        return 0.0

    head_xz = head_xz / norm_h
    body_xz = body_xz / norm_b

    det = body_xz[0] * head_xz[1] - body_xz[1] * head_xz[0]
    dot = np.clip(body_xz[0] * head_xz[0] + body_xz[1] * head_xz[1], -1.0, 1.0)
    return float(np.degrees(np.arctan2(det, dot)))


def _yaw_deg_from_vec(vec):
    return float(np.degrees(np.arctan2(vec[0], vec[2] + EPS)))


def _rotate_y(vec, deg):
    rad = np.radians(deg)
    c = np.cos(rad)
    s = np.sin(rad)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([
        c * x + s * z,
        y,
        -s * x + c * z,
    ], dtype=np.float32)


def _clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


def _orient_normal_for_consistency(unit_vec):
    if unit_vec[2] < 0:
        return -unit_vec
    return unit_vec


def compute_sam3d_angles(image_path, estimator):
    if not image_path or not os.path.exists(image_path):
        return {
            "head_body_yaw": None,
            "head_pitch": None,
            "status": "Not_Found",
            "person_count": 0,
        }

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {
                "head_body_yaw": None,
                "head_pitch": None,
                "status": "Read_Failed",
                "person_count": 0,
            }

        outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        person_count = len(outputs) if outputs is not None else 0
        if not outputs:
            return {
                "head_body_yaw": None,
                "head_pitch": None,
                "status": "No_Person",
                "person_count": person_count,
            }

        person_output = outputs[0]
        kpts_3d = person_output["pred_keypoints_3d"]

        nose = kpts_3d[IDX_NOSE][:3]
        left_eye = kpts_3d[IDX_LEFT_EYE][:3]
        right_eye = kpts_3d[IDX_RIGHT_EYE][:3]
        left_ear = kpts_3d[IDX_LEFT_EAR][:3]
        right_ear = kpts_3d[IDX_RIGHT_EAR][:3]
        left_shoulder = kpts_3d[IDX_LEFT_SHOULDER_SAM][:3]
        right_shoulder = kpts_3d[IDX_RIGHT_SHOULDER_SAM][:3]
        neck = kpts_3d[IDX_NECK][:3]

        nose_to_left_eye = left_eye - nose
        nose_to_right_eye = right_eye - nose
        cross_face = np.cross(nose_to_left_eye, nose_to_right_eye)

        neck_to_right_shoulder = right_shoulder - neck
        neck_to_left_shoulder = left_shoulder - neck
        cross_shoulder = np.cross(neck_to_right_shoulder, neck_to_left_shoulder)

        face_mag = float(np.linalg.norm(cross_face))
        shoulder_mag = float(np.linalg.norm(cross_shoulder))

        face_denom = float(np.linalg.norm(nose_to_left_eye) * np.linalg.norm(nose_to_right_eye) + EPS)
        shoulder_denom = float(np.linalg.norm(neck_to_right_shoulder) * np.linalg.norm(neck_to_left_shoulder) + EPS)

        confidence_face = _clamp01(face_mag / face_denom)
        confidence_body = _clamp01(shoulder_mag / shoulder_denom)

        if confidence_face < FACE_CONF_MIN:
            return {
                "head_body_yaw": None,
                "head_pitch": None,
                "status": "Degenerate_Face_Geometry",
                "person_count": person_count,
            }

        if confidence_body < SHOULDER_CONF_MIN:
            return {
                "head_body_yaw": None,
                "head_pitch": None,
                "status": "Degenerate_Shoulder_Geometry",
                "person_count": person_count,
            }

        head_center = (nose + left_eye + right_eye) / 3.0
        body_center = (neck + left_shoulder + right_shoulder) / 3.0

        use_ears_for_pitch = _is_valid_point(left_ear) and _is_valid_point(right_ear)
        if use_ears_for_pitch:
            ref_left = left_ear
            ref_right = right_ear
            status = "OK"
        else:
            ref_left = left_eye
            ref_right = right_eye
            status = "Fallback_Eyes_For_Pitch"

        head_lr = ref_right - ref_left
        head_up = nose - head_center
        body_lr = right_shoulder - left_shoulder
        body_up = neck - body_center

        head_vec = np.cross(head_lr, head_up)
        body_vec = np.cross(body_lr, body_up)

        head_vec = _orient_normal_for_consistency(_unit_vec(head_vec))
        body_vec = _orient_normal_for_consistency(_unit_vec(body_vec))

        head_body_yaw = _signed_angle_yaw_deg(head_vec, body_vec)

        body_yaw_abs = _yaw_deg_from_vec(body_vec)
        head_aligned = _rotate_y(head_vec, -body_yaw_abs)
        body_aligned = _rotate_y(body_vec, -body_yaw_abs)

        ear_center = (ref_left + ref_right) / 2.0
        nose_from_ears = nose - ear_center
        nose_from_ears_aligned = _rotate_y(nose_from_ears, -body_yaw_abs)

        head_pitch_abs = float(np.degrees(np.arctan2(-nose_from_ears_aligned[1], np.abs(nose_from_ears_aligned[2]) + EPS)))
        body_pitch_abs = float(
            np.degrees(
                np.arctan2(-body_aligned[1], np.sqrt(body_aligned[0] ** 2 + body_aligned[2] ** 2) + EPS)
            )
        )
        pitch_deg = head_pitch_abs - body_pitch_abs
        pitch_deg = pitch_deg + PITCH_OFFSET_DEG

        head_body_yaw = limit_angle(head_body_yaw)
        pitch_deg = limit_angle(pitch_deg)

        return {
            "head_body_yaw": head_body_yaw,
            "head_pitch": pitch_deg,
            "status": status,
            "person_count": person_count,
        }
    except Exception as e:
        return {
            "head_body_yaw": None,
            "head_pitch": None,
            "status": f"Error: {e}",
            "person_count": 0,
        }


def collect_image_paths(input_dir, recursive=True):
    image_paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS:
                    image_paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(input_dir):
            full_path = os.path.join(input_dir, filename)
            if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS:
                image_paths.append(full_path)
    image_paths.sort()
    return image_paths


def run_sam3d_inference(input_dir, output_csv, sam3d_repo_id, recursive=True, max_images=0):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not HAS_SAM3D:
        raise ImportError("Cannot import notebook.utils.setup_sam_3d_body. Ensure sam-3d-body env/path is available.")

    image_paths = collect_image_paths(input_dir, recursive=recursive)
    if not image_paths:
        print(f"⚠️ No images found in {input_dir}")
        return 0

    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]
        print(f"🧪 Testing mode: only first {len(image_paths)} images")

    fieldnames = [
        "image_path",
        "sam3d_head_body_yaw",
        "sam3d_head_pitch",
        "sam3d_status",
        "sam3d_person_count",
    ]

    done_paths = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "image_path" in reader.fieldnames:
                    for row in reader:
                        path = row.get("image_path", "")
                        if path:
                            done_paths.add(path)
        except Exception as e:
            print(f"⚠️ Failed to read existing CSV for resume: {e}")

    total_candidates = len(image_paths)
    remaining_paths = [p for p in image_paths if p not in done_paths]
    already_done = total_candidates - len(remaining_paths)

    print(
        f"📊 Inference plan -> total: {total_candidates}, already inferred: {already_done}, remaining: {len(remaining_paths)}"
    )

    if len(remaining_paths) == 0:
        print("✅ Nothing left to infer. Skip inference.")
        return 0

    estimator = setup_sam_3d_body(hf_repo_id=sam3d_repo_id)

    rows = []
    for image_path in tqdm(remaining_paths, desc="Running SAM3D pose inference"):
        res = compute_sam3d_angles(image_path, estimator)
        rows.append(
            {
                "image_path": image_path,
                "sam3d_head_body_yaw": _safe_float(res["head_body_yaw"]),
                "sam3d_head_pitch": _safe_float(res["head_pitch"]),
                "sam3d_status": res["status"],
                "sam3d_person_count": int(res["person_count"]),
            }
        )

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = (not os.path.exists(output_csv)) or (len(done_paths) == 0)
    mode = "a" if os.path.exists(output_csv) and len(done_paths) > 0 else "w"
    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3D-only pose angle inference and export CSV")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image folder")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--sam3d_repo_id", type=str, default="facebook/sam-3d-body-dinov3", help="SAM-3D-Body HF repo id")
    parser.add_argument("--non_recursive", action="store_true", help="Only process files in input_dir")
    parser.add_argument("--max_images", type=int, default=0, help="Only test first N images (0 means all)")
    args = parser.parse_args()

    total = run_sam3d_inference(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        sam3d_repo_id=args.sam3d_repo_id,
        recursive=not args.non_recursive,
        max_images=args.max_images,
    )

    print("\n" + "=" * 80)
    print(f"✅ SAM3D done. Processed images: {total}")
    print(f"📄 CSV saved to: {args.output_csv}")
    print("=" * 80)

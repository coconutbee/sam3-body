import cv2
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

SELECTED_KEYPOINTS = {
	0: "nose",
	1: "left_eye",
	2: "right_eye",
	5: "left_shoulder",
	6: "right_shoulder",
	69: "neck",
}


def draw_selected_3d_keypoints_panel(img_bgr, outputs):
	panel = img_bgr.copy()
	person_colors = [
		(0, 255, 255),
		(255, 0, 255),
		(255, 255, 0),
		(0, 165, 255),
		(0, 255, 0),
		(255, 0, 0),
	]
	line_height = 22
	start_y = 24
	arrow_len = 60

	def _fmt_vec(vec):
		return f"({vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f})"

	def _unit_vec(vec, eps=1e-8):
		norm = np.linalg.norm(vec)
		if norm < eps:
			return np.zeros_like(vec)
		return vec / norm

	def _angle_deg(vec_a, vec_b, eps=1e-8):
		norm_a = np.linalg.norm(vec_a)
		norm_b = np.linalg.norm(vec_b)
		if norm_a < eps or norm_b < eps:
			return 0.0
		cos_theta = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
		return float(np.degrees(np.arccos(cos_theta)))

	def _draw_unit_arrow(panel_img, start_xy, unit_vec_3d, color):
		vec_2d = np.array([unit_vec_3d[0], -unit_vec_3d[1]], dtype=np.float32)
		vec_norm = np.linalg.norm(vec_2d)
		if vec_norm < 1e-8:
			return
		vec_2d = vec_2d / vec_norm

		x0, y0 = int(round(start_xy[0])), int(round(start_xy[1]))
		x1 = int(round(x0 + vec_2d[0] * arrow_len))
		y1 = int(round(y0 + vec_2d[1] * arrow_len))

		cv2.arrowedLine(
			panel_img,
			(x0, y0),
			(x1, y1),
			color,
			2,
			cv2.LINE_AA,
			tipLength=0.25,
		)

	def _project_3d_to_2d(pt_3d, cam_t, focal_length, img_w, img_h, eps=1e-8):
		pt_cam = pt_3d + cam_t
		z = float(pt_cam[2])
		if z <= eps:
			return None
		u = float(focal_length * (pt_cam[0] / z) + img_w / 2.0)
		v = float(focal_length * (pt_cam[1] / z) + img_h / 2.0)
		return np.array([u, v], dtype=np.float32)

	def _draw_xyz_axes_at_point(panel_img, origin_3d, cam_t, focal_length, axis_len, pid):
		img_h, img_w = panel_img.shape[:2]
		origin_2d = _project_3d_to_2d(origin_3d, cam_t, focal_length, img_w, img_h)
		if origin_2d is None:
			return

		axes = [
			("+X", np.array([axis_len, 0.0, 0.0], dtype=np.float32), (0, 0, 255)),
			("+Y", np.array([0.0, axis_len, 0.0], dtype=np.float32), (0, 255, 0)),
			("+Z", np.array([0.0, 0.0, axis_len], dtype=np.float32), (255, 0, 0)),
		]

		o = (int(round(origin_2d[0])), int(round(origin_2d[1])))
		cv2.circle(panel_img, o, 4, (255, 255, 255), -1)
		cv2.putText(
			panel_img,
			f"P{pid} axis@nose",
			(o[0] + 6, max(14, o[1] - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)

		for axis_name, delta_3d, axis_color in axes:
			end_2d = _project_3d_to_2d(
				origin_3d + delta_3d, cam_t, focal_length, img_w, img_h
			)
			if end_2d is None:
				continue

			e = (int(round(end_2d[0])), int(round(end_2d[1])))
			cv2.arrowedLine(
				panel_img,
				o,
				e,
				axis_color,
				2,
				cv2.LINE_AA,
				tipLength=0.25,
			)
			cv2.putText(
				panel_img,
				axis_name,
				(e[0] + 4, e[1] + 4),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				axis_color,
				1,
				cv2.LINE_AA,
			)

	for pid, person_output in enumerate(outputs):
		color = person_colors[pid % len(person_colors)]
		kpts_2d = person_output["pred_keypoints_2d"]
		kpts_3d = person_output["pred_keypoints_3d"]
		cam_t = person_output["pred_cam_t"]
		focal_length = person_output["focal_length"]

		for kpt_idx, kpt_name in SELECTED_KEYPOINTS.items():
			x_2d, y_2d = kpts_2d[kpt_idx][:2]
			x_3d, y_3d, z_3d = kpts_3d[kpt_idx][:3]

			x_px, y_px = int(round(x_2d)), int(round(y_2d))
			cv2.circle(panel, (x_px, y_px), 5, color, -1)

			label = f"P{pid} {kpt_name}: ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})"
			text_y = max(20, y_px - 8)
			cv2.putText(
				panel,
				label,
				(min(x_px + 8, panel.shape[1] - 5), text_y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				color,
				1,
				cv2.LINE_AA,
			)

		nose = kpts_3d[0][:3]
		left_eye = kpts_3d[1][:3]
		right_eye = kpts_3d[2][:3]
		left_shoulder = kpts_3d[5][:3]
		right_shoulder = kpts_3d[6][:3]
		neck = kpts_3d[69][:3]

		nose_to_left_eye = left_eye - nose
		nose_to_right_eye = right_eye - nose
		cross_face = np.cross(nose_to_left_eye, nose_to_right_eye)

		neck_to_right_shoulder = right_shoulder - neck
		neck_to_left_shoulder = left_shoulder - neck
		cross_shoulder = np.cross(neck_to_right_shoulder, neck_to_left_shoulder)

		face_unit = _unit_vec(cross_face)
		shoulder_unit = _unit_vec(cross_shoulder)

		# Yaw: angle between face normal and body normal
		yaw_deg = _angle_deg(face_unit, shoulder_unit)

		# Pitch: signed angle with z-axis (using yz-plane), constrained to [-90, 90].
		# Face normal direction can flip (n and -n are equivalent), so we use |z|
		# to avoid 180-degree ambiguity while preserving up/down sign from y.
		pitch_deg = float(np.degrees(np.arctan2(-face_unit[1], np.abs(face_unit[2]) + 1e-8)))

		nose_2d = kpts_2d[0][:2]
		neck_2d = kpts_2d[69][:2]
		_draw_unit_arrow(panel, nose_2d, face_unit, color)
		_draw_unit_arrow(panel, neck_2d, shoulder_unit, color)
		_draw_xyz_axes_at_point(
			panel,
			nose,
			cam_t,
			focal_length,
			axis_len=0.12,
			pid=pid,
		)

		info_y = start_y + pid * 7 * line_height
		cv2.putText(
			panel,
			f"P{pid} face_cross: {_fmt_vec(cross_face)}",
			(10, info_y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			panel,
			f"P{pid} shoulder_cross: {_fmt_vec(cross_shoulder)}",
			(10, info_y + line_height),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			panel,
			f"P{pid} face_unit@nose: {_fmt_vec(face_unit)}",
			(10, info_y + 2 * line_height),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			panel,
			f"P{pid} shoulder_unit@neck: {_fmt_vec(shoulder_unit)}",
			(10, info_y + 3 * line_height),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			panel,
			f"P{pid} yaw(face-body): {yaw_deg:.2f} deg",
			(10, info_y + 4 * line_height),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			panel,
			f"P{pid} pitch(face up/down): {pitch_deg:.2f} deg",
			(10, info_y + 5 * line_height),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)

	return panel

# Set up the estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

# Load and process image
img_bgr = cv2.imread("/media/ee303/5090_disk2/JACK/ECCV_DATA/Infinity_pose_only/69792_An Asian girl in a shirt, head turned to her right, portrait.jpg")
img_bgr = cv2.imread("/media/ee303/5090_disk2/JACK/ECCV_DATA/Infinity_pose_only/63551_A Caucasian boy in an outfit, head turned to his right over the shoulder and tilted up, portrait.jpg")

outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
print('#'*30)
# 1. 先取出第一個偵測結果
res = outputs[0]

# 2. 抓出各個變數的結果
kpts_3d = res['pred_keypoints_3d']    # 3D 關鍵點座標
kpts_2d = res['pred_keypoints_2d']    # 2D 關鍵點座標
verts   = res['pred_vertices']        # 3D 人體模型頂點
pose    = res['body_pose_params']     # 身體姿勢參數 (SMPL 格式)

# 3. 統一列印資訊
# print(f"{'Variable':<20} | {'Shape':<15}")
# print("-" * 40)
# print(f"{'pred_keypoints_3d':<20} | {kpts_3d.shape}")
# print(f"{'pred_keypoints_2d':<20} | {kpts_2d.shape}")
# print(f"{'pred_vertices':<20} | {verts.shape}")
# print(f"{'body_pose_params':<20} | {pose.shape}")
# print('#'*30)
# Visualize and save results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
r_panel = draw_selected_3d_keypoints_panel(img_bgr, outputs)
rend_img = np.concatenate([rend_img, r_panel], axis=1)
cv2.imwrite("outpuuu.jpg", rend_img.astype(np.uint8))
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import torch
import smplx
from keypoints_map import load_keypoints3d_map


def create_smpl_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("SMPL model not found: {}".format(model_path))
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        model = smplx.create(
            str(model_path),
            model_type="smpl",
            batch_size=1,
            gender="neutral",
        )
    return model.eval()


# COCO face keypoints taken from SMPL mesh vertices (same vertex ids as smplx's extra joints).
FACE_VERTEX_IDS = {"nose": 332, "right_eye": 6260, "left_eye": 2800, "right_ear": 4071, "left_ear": 583}


def get_face_points_for_frame(model, person_data, frame_idx, use_world: bool = False):
    """COCO nose/eyes/ears for one frame, in the same frame as get_3d_joints_for_frame."""
    pose_key = "pose_world" if use_world and "pose_world" in person_data else "pose"
    trans_key = "trans_world" if use_world and "trans_world" in person_data else "trans"
    pose = torch.tensor(np.ascontiguousarray(person_data[pose_key][frame_idx:frame_idx + 1]), dtype=torch.float32)
    trans = torch.tensor(np.ascontiguousarray(person_data[trans_key][frame_idx:frame_idx + 1]), dtype=torch.float32)
    betas = torch.tensor(np.ascontiguousarray(person_data["betas"][frame_idx:frame_idx + 1]), dtype=torch.float32)
    with torch.no_grad():
        verts = model(betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=trans).vertices[0].cpu().numpy()
    return {name: [float(v) for v in verts[vid]] for name, vid in FACE_VERTEX_IDS.items()}


def get_3d_joints_for_frame(model, person_data, frame_idx, regressor_path, map_path, use_world: bool = False):
    pose_key = "pose_world" if use_world and "pose_world" in person_data else "pose"
    trans_key = "trans_world" if use_world and "trans_world" in person_data else "trans"
    pose = person_data[pose_key][frame_idx:frame_idx + 1]
    trans = person_data[trans_key][frame_idx:frame_idx + 1]
    curr_betas = person_data["betas"][frame_idx:frame_idx + 1]

    curr_pose_t = torch.tensor(np.ascontiguousarray(pose), dtype=torch.float32)
    curr_trans_t = torch.tensor(np.ascontiguousarray(trans), dtype=torch.float32)
    curr_betas_t = torch.tensor(np.ascontiguousarray(curr_betas), dtype=torch.float32)

    with torch.no_grad():
        output = model(
            betas=curr_betas_t,
            global_orient=curr_pose_t[:, :3],
            body_pose=curr_pose_t[:, 3:],
            transl=curr_trans_t,
        )

    vertices = output.vertices[0].cpu().numpy()
    j_reg = np.load(regressor_path)
    if j_reg.shape[1] != 6890:
        raise ValueError(f"Expected (_, 6890) for SMPL vertices, got {j_reg.shape}")

    joints_all = j_reg @ vertices

    map_data = load_keypoints3d_map(map_path)
    result = {}
    for kp in map_data["keypoints"]:
        x, y, z = joints_all[kp["regressor_index"]]
        result[kp["name"]] = [float(x), float(y), float(z)]
    return result


def get_3d_joints_and_face_sequence(
    model,
    person_data,
    regressor_path,
    map_path,
    use_world: bool = False,
    export_face: bool = True,
    chunk_size: int = 128,
):
    """Batched SMPL forward returning (joints_list, face_list) from one pass per chunk."""
    # ponytail: single batched SMPL pass + single regressor load instead of 4 batch=1 passes/frame
    pose_key = "pose_world" if use_world and "pose_world" in person_data else "pose"
    trans_key = "trans_world" if use_world and "trans_world" in person_data else "trans"
    pose = np.ascontiguousarray(person_data[pose_key], dtype=np.float32)
    trans = np.ascontiguousarray(person_data[trans_key], dtype=np.float32)
    betas = np.ascontiguousarray(person_data["betas"], dtype=np.float32)
    n = int(pose.shape[0])
    if n == 0:
        return [], ([] if export_face else None)
    if betas.ndim == 1:
        betas = np.repeat(betas[None, :], n, axis=0)

    j_reg = np.load(regressor_path)
    if j_reg.shape[1] != 6890:
        raise ValueError(f"Expected (_, 6890) for SMPL vertices, got {j_reg.shape}")
    map_data = load_keypoints3d_map(map_path)
    names = [kp["name"] for kp in map_data["keypoints"]]
    reg_sub = np.ascontiguousarray(j_reg[[kp["regressor_index"] for kp in map_data["keypoints"]]], dtype=np.float32)
    face_items = list(FACE_VERTEX_IDS.items())

    joints_out = []
    face_out = [] if export_face else None
    step = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, n, step):
            end = min(n, start + step)
            p_t = torch.from_numpy(pose[start:end])
            t_t = torch.from_numpy(trans[start:end])
            b_t = torch.from_numpy(betas[start:end])
            verts = model(betas=b_t, global_orient=p_t[:, :3], body_pose=p_t[:, 3:], transl=t_t).vertices.cpu().numpy()
            joints_batch = np.einsum("jv,tvc->tjc", reg_sub, verts)
            for t_i in range(end - start):
                joints_out.append({name: [float(x) for x in joints_batch[t_i, j_i]] for j_i, name in enumerate(names)})
                if export_face:
                    face_out.append({fname: [float(x) for x in verts[t_i, vid]] for fname, vid in face_items})
    return joints_out, face_out

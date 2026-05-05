import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is available in assignment env
    torch = None


ArrayLike = Union[np.ndarray, "torch.Tensor"]


@dataclass
class BVHSkeleton:
    joint_names: List[str]
    joint_parents: List[int]
    root_index: int = 0

    def children(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {i: [] for i in range(len(self.joint_names))}
        for i, parent in enumerate(self.joint_parents):
            if parent >= 0:
                out[parent].append(i)
        return out


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_positions(motion: Union[ArrayLike, Dict[str, ArrayLike]], batch_index: int = 0) -> np.ndarray:
    if isinstance(motion, dict):
        if "positions" not in motion:
            raise KeyError("Expected motion dict to contain a 'positions' key.")
        positions = _to_numpy(motion["positions"])
    else:
        positions = _to_numpy(motion)

    if positions.ndim == 4:
        if not (0 <= batch_index < positions.shape[0]):
            raise IndexError(f"batch_index {batch_index} out of range for shape {positions.shape}")
        positions = positions[batch_index]

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "Expected positions shape [T, J, 3] or [B, T, J, 3]. "
            f"Got shape {positions.shape}"
        )

    return positions.astype(np.float64, copy=False)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = _normalize(a.reshape(1, 3), eps=eps)[0]
    b = _normalize(b.reshape(1, 3), eps=eps)[0]

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < eps:
        if c > 0.0:
            return np.eye(3)

        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(a, axis).reshape(1, 3))[0]
        x, y, z = axis
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + 2.0 * (K @ K)

    vx, vy, vz = v
    K = np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _kabsch_rotation(src_vectors: np.ndarray, dst_vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if src_vectors.shape[0] == 0:
        return np.eye(3)

    src = _normalize(src_vectors, eps=eps)
    dst = _normalize(dst_vectors, eps=eps)

    if src.shape[0] == 1:
        return _rotation_from_a_to_b(src[0], dst[0], eps=eps)

    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def _matrix_to_euler_zxy_degrees_candidates(R: np.ndarray) -> List[np.ndarray]:
    x = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    cx = np.cos(x)

    if abs(cx) > 1e-8:
        y = np.arctan2(-R[2, 0], R[2, 2])
        z = np.arctan2(-R[0, 1], R[1, 1])
        alt_x = np.pi - x
        alt_y = np.arctan2(R[2, 0], -R[2, 2])
        alt_z = np.arctan2(R[0, 1], -R[1, 1])
    else:
        y = 0.0
        z = np.arctan2(R[1, 0], R[0, 0])
        alt_x = x
        alt_y = y
        alt_z = z

    primary = np.degrees(np.array([z, x, y], dtype=np.float64))
    alternate = np.degrees(np.array([alt_z, alt_x, alt_y], dtype=np.float64))
    return [primary, alternate]


def _closest_continuous_euler(candidates: List[np.ndarray], prev: Optional[np.ndarray]) -> np.ndarray:
    if prev is None:
        return candidates[0]

    best = candidates[0]
    best_err = np.inf

    for cand in candidates:
        for kz in (-1.0, 0.0, 1.0):
            for kx in (-1.0, 0.0, 1.0):
                for ky in (-1.0, 0.0, 1.0):
                    shifted = cand + 360.0 * np.array([kz, kx, ky], dtype=np.float64)
                    err = float(np.sum((shifted - prev) ** 2))
                    if err < best_err:
                        best = shifted
                        best_err = err
    return best


def _bidirectional_ema(data: np.ndarray, alpha: float) -> np.ndarray:
    T = data.shape[0]
    if T <= 2:
        return data.copy()

    fwd = data.copy()
    for t in range(1, T):
        fwd[t] = alpha * data[t] + (1.0 - alpha) * fwd[t - 1]

    bwd = data.copy()
    for t in range(T - 2, -1, -1):
        bwd[t] = alpha * data[t] + (1.0 - alpha) * bwd[t + 1]

    return 0.5 * (fwd + bwd)


def _smooth_positions_temporal(
    positions: np.ndarray,
    smoothing_alpha: float,
    smoothing_passes: int,
    root_index: int,
    keep_root_trajectory: bool = True,
) -> np.ndarray:
    alpha = float(np.clip(smoothing_alpha, 1e-4, 1.0))
    passes = max(1, int(smoothing_passes))

    smoothed = positions.copy()
    root_traj = positions[:, root_index, :].copy()

    for _ in range(passes):
        smoothed = _bidirectional_ema(smoothed, alpha)

    if keep_root_trajectory:
        smoothed[:, root_index, :] = root_traj

    return smoothed


def _sanitize_name(name: str) -> str:
    s = str(name).strip().replace(" ", "_")
    return s if s else "joint"


def _traversal_order(children: Dict[int, List[int]], root: int) -> List[int]:
    order: List[int] = []

    def dfs(j: int) -> None:
        order.append(j)
        for c in children.get(j, []):
            dfs(c)

    dfs(root)
    return order


def _hierarchy_lines(
    skeleton: BVHSkeleton,
    rest_offsets: np.ndarray,
    end_site_scale: float = 0.3,
) -> List[str]:
    children = skeleton.children()
    names = [_sanitize_name(n) for n in skeleton.joint_names]

    def joint_block(j: int, indent: int) -> List[str]:
        pad = "\t" * indent
        lines: List[str] = []

        if j == skeleton.root_index:
            lines.append(f"{pad}ROOT {names[j]}")
        else:
            lines.append(f"{pad}JOINT {names[j]}")

        lines.append(f"{pad}{{")
        off = rest_offsets[j]
        lines.append(f"{pad}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")

        if j == skeleton.root_index:
            lines.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}\tCHANNELS 3 Zrotation Xrotation Yrotation")

        joint_children = children[j]
        if joint_children:
            for c in joint_children:
                lines.extend(joint_block(c, indent + 1))
        else:
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            if skeleton.joint_parents[j] >= 0:
                parent_offset = rest_offsets[j]
                norm = np.linalg.norm(parent_offset)
                if norm > 1e-8:
                    end_offset = parent_offset / norm * norm * end_site_scale
                else:
                    end_offset = np.array([0.0, end_site_scale, 0.0], dtype=np.float64)
            else:
                end_offset = np.array([0.0, end_site_scale, 0.0], dtype=np.float64)
            lines.append(
                f"{pad}\t\tOFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}"
            )
            lines.append(f"{pad}\t}}")

        lines.append(f"{pad}}}")
        return lines

    return ["HIERARCHY", *joint_block(skeleton.root_index, 0)]


def _compute_frame_rotations(
    positions_t: np.ndarray,
    rest_vectors: List[np.ndarray],
    skeleton: BVHSkeleton,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    num_joints = positions_t.shape[0]
    children = skeleton.children()

    global_rot = [np.eye(3) for _ in range(num_joints)]
    local_rot = [np.eye(3) for _ in range(num_joints)]

    order = _traversal_order(children, skeleton.root_index)

    for j in order:
        ch = children[j]
        if not ch:
            local_rot[j] = np.eye(3)
            parent = skeleton.joint_parents[j]
            if parent >= 0:
                global_rot[j] = global_rot[parent] @ local_rot[j]
            continue

        src = np.stack([rest_vectors[c] for c in ch], axis=0)
        dst_global = np.stack([positions_t[c] - positions_t[j] for c in ch], axis=0)

        parent = skeleton.joint_parents[j]
        if parent >= 0:
            dst = (global_rot[parent].T @ dst_global.T).T
            R_local = _kabsch_rotation(src, dst)
            local_rot[j] = R_local
            global_rot[j] = global_rot[parent] @ R_local
        else:
            R_global = _kabsch_rotation(src, dst_global)
            local_rot[j] = R_global
            global_rot[j] = R_global

    root_pos = positions_t[skeleton.root_index]
    return root_pos, local_rot


def export_motion_to_bvh(
    motion: Union[ArrayLike, Dict[str, ArrayLike]],
    output_path: str,
    joint_names: Sequence[str],
    joint_parents: Sequence[int],
    joint_offsets: Optional[ArrayLike] = None,
    fps: float = 30.0,
    root_index: int = 0,
    batch_index: int = 0,
    temporal_smoothing_alpha: Optional[float] = None,
    temporal_smoothing_passes: int = 2,
    max_angular_step_deg: Optional[float] = None,
) -> str:
    """Export global joint positions to a BVH file for Blender.

    Args:
        motion: Joint positions as [T, J, 3] or [B, T, J, 3], or dict with key 'positions'.
        output_path: Destination .bvh file path.
        joint_names: Joint names matching axis 1 in motion.
        joint_parents: Parent indices (-1 for root), matching joint_names.
        joint_offsets: Optional canonical local offsets [J, 3] from the source skeleton.
            When provided, these are used directly for BVH hierarchy stability.
        fps: Frames per second for BVH timing.
        root_index: Root joint index in the arrays.
        batch_index: Batch index when motion has shape [B, T, J, 3].
        temporal_smoothing_alpha: Optional temporal smoothing strength in (0, 1].
            Lower values smooth more aggressively. Set None to disable.
        temporal_smoothing_passes: Number of bidirectional smoothing passes.
        max_angular_step_deg: Optional per-frame Euler step clamp in degrees.
            Set None to disable.

    Returns:
        Absolute path to the written BVH file.
    """
    raw_positions = _ensure_positions(motion, batch_index=batch_index)
    positions = raw_positions
    if temporal_smoothing_alpha is not None:
        positions = _smooth_positions_temporal(
            positions,
            smoothing_alpha=float(temporal_smoothing_alpha),
            smoothing_passes=temporal_smoothing_passes,
            root_index=int(root_index),
            keep_root_trajectory=True,
        )
    T, J, D = positions.shape
    if D != 3:
        raise ValueError(f"Expected last dimension to be 3, got {D}")

    if len(joint_names) != J or len(joint_parents) != J:
        raise ValueError(
            "joint_names and joint_parents must match the joint dimension. "
            f"Got J={J}, len(names)={len(joint_names)}, len(parents)={len(joint_parents)}"
        )

    skeleton = BVHSkeleton(list(joint_names), [int(p) for p in joint_parents], root_index=int(root_index))
    children = skeleton.children()

    if joint_offsets is not None:
        rest_offsets = _to_numpy(joint_offsets).astype(np.float64, copy=False)
        if rest_offsets.shape != (J, 3):
            raise ValueError(
                "joint_offsets must have shape [J, 3]. "
                f"Got {rest_offsets.shape}, expected ({J}, 3)"
            )
    else:
        rest_offsets = np.zeros((J, 3), dtype=np.float64)
        for j, p in enumerate(skeleton.joint_parents):
            if p >= 0:
                # Build offsets from stable bone lengths + frame-0 direction.
                # Taking a median of full world-space vectors can collapse toward zero
                # when the character rotates over time, which breaks BVH hierarchy.
                bone_vecs = raw_positions[:, j] - raw_positions[:, p]
                bone_lens = np.linalg.norm(bone_vecs, axis=-1)
                median_len = float(np.median(bone_lens))

                base_dir = bone_vecs[0]
                base_norm = float(np.linalg.norm(base_dir))
                if base_norm < 1e-8:
                    # Fallback to the longest observed bone vector when frame 0 is degenerate.
                    best_idx = int(np.argmax(bone_lens))
                    base_dir = bone_vecs[best_idx]
                    base_norm = float(np.linalg.norm(base_dir))

                if base_norm < 1e-8:
                    base_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    base_norm = 1.0

                rest_offsets[j] = base_dir / base_norm * median_len

    rest_vectors = [np.zeros(3, dtype=np.float64) for _ in range(J)]
    for j, p in enumerate(skeleton.joint_parents):
        if p >= 0:
            rest_vectors[j] = rest_offsets[j]

    hierarchy = _hierarchy_lines(skeleton, rest_offsets)

    traversal = _traversal_order(children, skeleton.root_index)
    non_root = [j for j in traversal if j != skeleton.root_index]

    motion_lines: List[str] = ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / float(fps):.8f}"]

    prev_eulers: List[Optional[np.ndarray]] = [None for _ in range(J)]

    for t in range(T):
        root_pos, local_rot = _compute_frame_rotations(positions[t], rest_vectors, skeleton)
        eulers = np.zeros((J, 3), dtype=np.float64)
        for j in range(J):
            candidates = _matrix_to_euler_zxy_degrees_candidates(local_rot[j])
            chosen = _closest_continuous_euler(candidates, prev_eulers[j])
            if prev_eulers[j] is not None and max_angular_step_deg is not None:
                max_step = float(max_angular_step_deg)
                chosen = prev_eulers[j] + np.clip(chosen - prev_eulers[j], -max_step, max_step)
            eulers[j] = chosen
            prev_eulers[j] = chosen

        channel_values: List[float] = []

        root_euler = eulers[skeleton.root_index]
        channel_values.extend([float(root_pos[0]), float(root_pos[1]), float(root_pos[2])])
        channel_values.extend([float(root_euler[0]), float(root_euler[1]), float(root_euler[2])])

        for j in non_root:
            e = eulers[j]
            channel_values.extend([float(e[0]), float(e[1]), float(e[2])])

        motion_lines.append(" ".join(f"{v:.6f}" for v in channel_values))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    out_abs = os.path.abspath(output_path)

    with open(out_abs, "w", encoding="utf-8") as f:
        for line in hierarchy:
            f.write(line + "\n")
        for line in motion_lines:
            f.write(line + "\n")

    return out_abs

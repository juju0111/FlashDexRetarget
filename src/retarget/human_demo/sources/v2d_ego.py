from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from retarget.human_demo.sources.base import Demo, Hand, Source as Base, _mano


class Source(Base):
    """V2D egocentric reconstruction bundles (nvidia-isaac/video_to_data result bundles, post-processed).

    <raw>/<clip>/result.npz + mesh.obj (+ manifest.json). The bundle world is already gravity-aligned,
    levelled and metric (hand-scale rescale), at the video rate (fps.json or 20 Hz). MANO parameters are
    stored in the camera frame (HaMeR convention: global_orient + 45-d finger pose, transl = MANO
    translation); they are re-posed into the world with camera_to_world. The left hand is stored in
    right-hand parameter space and is mirrored back; import with sides=right for single-hand clips.
    One rigid object per clip, mesh.obj in metres."""

    mano = {"use_pca": False, "flat_hand_mean": True}

    def units(self) -> list[str]:
        return sorted(d.name for d in self.raw.iterdir() if (d / "result.npz").exists() and (d / "mesh.obj").exists())

    def _fps(self, clip_dir: Path) -> float:
        meta = clip_dir / "fps.json"
        if meta.exists():
            return float(json.load(open(meta))["fps"])
        return float(self.cfg.get("fps", 20.0))

    def demos(self, unit: str) -> list[Demo | str]:
        clip_dir = self.raw / unit
        d = dict(np.load(clip_dir / "result.npz", allow_pickle=False))
        c2w = d["camera_to_world_transform"].astype(np.float64)
        T = len(c2w)
        hands = {}
        for side in ("right", "left"):
            if not d[f"hand_{side}_is_valid"].astype(bool).all():
                continue
            orient = d[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)
            fpose = d[f"hand_{side}_finger_pose"].astype(np.float64).reshape(T, 45)
            transl = d[f"hand_{side}_wrist_trans_in_camera"].astype(np.float64)
            betas = d[f"hand_{side}_betas"].astype(np.float64).reshape(10)
            if side == "left":  # stored in right-hand space: mirror into the native left model
                orient = orient.copy(); orient[:, 1:] *= -1.0
                fpose = fpose.reshape(T, 15, 3).copy(); fpose[:, :, 1:] *= -1.0; fpose = fpose.reshape(T, 45)
            # MANO root joint in the camera frame = J0(betas) + transl (HaMeR transl convention)
            model = _mano(self.mano_dir, side, tuple(sorted(self.mano.items())))
            import torch
            with torch.no_grad():
                j0 = model(betas=torch.tensor(betas, dtype=torch.float32)[None]).joints[0, 0].numpy().astype(np.float64)
            root_cam = j0 + transl
            root_world = np.einsum("tij,tj->ti", c2w[:, :3, :3], root_cam) + c2w[:, :3, 3]
            rot_world = Rotation.from_matrix(np.einsum("tij,tjk->tik", c2w[:, :3, :3], Rotation.from_rotvec(orient).as_matrix())).as_rotvec()
            hands[side] = Hand(rot_world, fpose, betas, wrist=root_world)
        oid = unit
        poses = {oid: d["object_to_world_transform"].astype(np.float64)}
        objects = {s: oid for s in hands}
        return [Demo(unit, self._fps(clip_dir), hands, objects, poses)]

    def load_mesh(self, obj_id: str) -> trimesh.Trimesh:
        return trimesh.load(self.raw / obj_id / "mesh.obj", force="mesh", process=False, skip_materials=True)

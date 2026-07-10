"""
Main CAST pipeline — ties together all stages from a single RGB image
to a physically coherent 3D scene.

Usage:
    from pipeline import CASTPipeline

    pipe = CASTPipeline(config)          # see config.py
    scene = pipe.reconstruct(image_path) # path or np.ndarray
    scene.export(output_dir)             # writes meshes + scene graph
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import open3d as o3d

from config import CASTConfig
from scene_analysis import (analyze_scene, SceneAnalysisResult, ObjectInfo,
                            depth_to_point_cloud, build_relation_graph_vlm)
from sam3d_wrapper import SAM3DWrapper
from pose_adapter import PoseAdapter
from iterative_procedure import iterative_generate
from physics_correction import optimize_poses, compute_sdf_for_mesh, create_ground_plane


# ============================================================================
# Scene representation
# ============================================================================

class CASTScene:
    """Holds the reconstructed scene: meshes with transforms + metadata."""

    def __init__(self):
        self.objects: List[dict] = []       # [{mesh, R, t, s, name, ...}]
        self.background: Optional[o3d.geometry.TriangleMesh] = None
        self.camera_intrinsics: Optional[np.ndarray] = None
        self.camera_pose: Optional[np.ndarray] = None
        self.relation_graph: Dict = {}

    def add_object(self, mesh: o3d.geometry.TriangleMesh,
                   R: np.ndarray, t: np.ndarray, s: float,
                   name: str = "object"):
        self.objects.append({
            'mesh': mesh,
            'R': R, 't': t, 's': s,
            'name': name,
        })

    def get_scene_mesh(self) -> o3d.geometry.TriangleMesh:
        """Combine all objects into one scene mesh."""
        combined = o3d.geometry.TriangleMesh()
        for obj in self.objects:
            m = o3d.geometry.TriangleMesh(obj['mesh'])
            # Apply similarity transform: scale, rotate, translate
            m.scale(obj['s'], center=(0, 0, 0))
            m.rotate(obj['R'].T, center=(0, 0, 0))  # inverse because rotate() applies R^T
            m.translate(obj['t'])
            combined += m
        return combined

    def export(self, output_dir: str):
        """Write all meshes and a scene-graph JSON to disk."""
        os.makedirs(output_dir, exist_ok=True)
        scene_graph = {'objects': [], 'camera': {}}

        for i, obj in enumerate(self.objects):
            name = obj.get('name', f'object_{i:03d}')
            mesh_path = os.path.join(output_dir, f'{name}.obj')
            o3d.io.write_triangle_mesh(mesh_path, obj['mesh'])

            scene_graph['objects'].append({
                'name': name,
                'mesh': f'{name}.obj',
                'transform': {
                    'rotation': obj['R'].tolist(),
                    'translation': obj['t'].tolist(),
                    'scale': float(obj['s']),
                },
            })

        if self.camera_intrinsics is not None:
            scene_graph['camera']['intrinsics'] = self.camera_intrinsics.tolist()
        if self.camera_pose is not None:
            scene_graph['camera']['pose'] = self.camera_pose.tolist()

        if self.relation_graph:
            # Convert tuple keys to strings for JSON serialization
            serializable_graph = {
                'nodes': self.relation_graph.get('nodes', []),
                'contact_edges': [
                    list(e) for e in self.relation_graph.get('contact_edges', [])
                ],
                'support_edges': [
                    list(e) for e in self.relation_graph.get('support_edges', [])
                ],
            }
            scene_graph['relation_graph'] = serializable_graph

        with open(os.path.join(output_dir, 'scene_graph.json'), 'w') as f:
            json.dump(scene_graph, f, indent=2)

        print(f"[CAST] Scene exported to {output_dir}/")
        print(f"       {len(self.objects)} objects written.")


# ============================================================================
# Main pipeline
# ============================================================================

class CASTPipeline:
    """
    CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image.

    Example (Qwen VLM):
        config = CASTConfig(
            device="cuda",
            vlm_provider="qwen",
            qwen_api_key="sk-xxxxx",
        )
        pipe   = CASTPipeline(config)
        scene  = pipe.reconstruct("my_room.jpg")
        scene.export("./output/my_room")

    Example (GPT-4V):
        config = CASTConfig(
            device="cuda",
            vlm_provider="openai",
            openai_api_key="sk-xxxxx",
        )
        pipe   = CASTPipeline(config)
        scene  = pipe.reconstruct("my_room.jpg")
    """

    def __init__(self, config: CASTConfig = None):
        self.config = config or CASTConfig()
        self.device = self.config.device

        # Lazy-init
        self._sam3d: Optional[SAM3DWrapper] = None
        self._pose_adapter: Optional[PoseAdapter] = None

    # ------------------------------------------------------------------
    @property
    def sam3d(self) -> SAM3DWrapper:
        if self._sam3d is None:
            self._sam3d = SAM3DWrapper(
                model_id=self.config.sam3d_model_id,
                device=self.device,
                use_fp16=self.config.sam3d_use_fp16,
                offline=self.config.sam3d_offline,
            )
        return self._sam3d

    @property
    def pose_adapter(self) -> PoseAdapter:
        if self._pose_adapter is None:
            self._pose_adapter = PoseAdapter(
                refinement=self.config.pose_refinement,
                camera_intrinsics=None,  # set after scene analysis
                icp_max_distance=self.config.icp_max_distance,
                verbose=True,
            )
        return self._pose_adapter

    # ------------------------------------------------------------------
    def reconstruct(self,
                    image: Union[str, np.ndarray],
                    output_dir: Optional[str] = None) -> CASTScene:
        """
        Full CAST reconstruction pipeline.

        Args:
            image:      path to RGB image, or (H, W, 3) uint8 array
            output_dir: optional output directory for intermediate results

        Returns:
            CASTScene with all generated meshes and transforms.
        """
        # 0. Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        print(f"[CAST] Input image: {img.shape[1]}x{img.shape[0]}")

        # ---- Stage 1: Scene Analysis (Section 3) ----
        print("\n" + "="*60)
        print("Stage 1/4: Scene Analysis")
        print("="*60)
        analysis = analyze_scene(
            img,
            openai_api_key=self.config.openai_api_key,
            openai_base_url=self.config.openai_base_url,
            gpt_model=self.config.gpt_model,
            vlm_provider=self.config.vlm_provider,
            vlm_api_key=self.config.get_vlm_api_key(),
            vlm_base_url=self.config.get_vlm_base_url(),
            vlm_model=self.config.get_vlm_model(),
            vlm_ensemble_trials=self.config.vlm_ensemble_trials,
            vlm_max_tokens=self.config.vlm_max_tokens,
            vlm_temperature=self.config.vlm_temperature,
        )
        print(f"  Detected {len(analysis.objects)} objects")
        for obj in analysis.objects:
            print(f"    [{obj.id}] {obj.name}: {obj.point_cloud.shape[0]} points")

        # ---- Stage 2: Per-object 3D Generation + Alignment (Section 4) ----
        print("\n" + "="*60)
        print("Stage 2/4: Per-Object Generation + Alignment")
        print("="*60)
        scene = CASTScene()
        scene.camera_intrinsics = analysis.camera_intrinsics
        scene.camera_pose = analysis.camera_pose

        for obj in analysis.objects:
            print(f"\n  Processing [{obj.id}] {obj.name} ...")

            # Crop image to object bounding box
            x1, y1, x2, y2 = obj.bbox
            crop = img[y1:y2, x1:x2]
            mask_crop = obj.mask[y1:y2, x1:x2]

            # Run SAM 3D generation + PoseAdapter refinement
            mesh, R, t, s = iterative_generate(
                image=crop,
                mask=mask_crop,
                scene_pc=obj.point_cloud,
                sam3d=self.sam3d,
                pose_adapter=self.pose_adapter,
                verbose=True,
            )

            scene.add_object(mesh, R, t, s, name=obj.name)

        # ---- Stage 3: Relation graph (Section 5.3) ----
        print("\n" + "="*60)
        print("Stage 3/4: Scene Relation Graph")
        print("="*60)

        scene.relation_graph = build_relation_graph_vlm(
            img, analysis.objects,
            vlm_provider=self.config.vlm_provider,
            api_key=self.config.get_vlm_api_key(),
            base_url=self.config.get_vlm_base_url(),
            model=self.config.get_vlm_model(),
            ensemble_trials=self.config.vlm_ensemble_trials,
            max_tokens=self.config.vlm_max_tokens,
            temperature=self.config.vlm_temperature,
        )
        print(f"  Contact edges:  {len(scene.relation_graph.get('contact_edges', []))}")
        print(f"  Support edges:  {len(scene.relation_graph.get('support_edges', []))}")

        # ---- Stage 4: Physics-Aware Correction (Section 5) ----
        print("\n" + "="*60)
        print("Stage 4/4: Physics-Aware Correction")
        print("="*60)

        if self.config.enable_physics_correction and (
            scene.relation_graph.get('contact_edges') or
            scene.relation_graph.get('support_edges')
        ):
            meshes = [obj['mesh'] for obj in scene.objects]
            object_infos = analysis.objects

            optimized = optimize_poses(
                meshes=meshes,
                objects=object_infos,
                relation_graph=scene.relation_graph,
                steps=self.config.physics_optim_steps,
                lr=self.config.physics_lr,
                sigma=self.config.physics_sdf_sigma,
                num_samples=self.config.physics_surface_samples,
                verbose=True,
            )

            # Apply optimized transforms
            for obj_id, (R_opt, t_opt, s_opt) in optimized.items():
                if obj_id < len(scene.objects):
                    scene.objects[obj_id]['R'] = R_opt
                    scene.objects[obj_id]['t'] = t_opt
                    # scale is kept from generation; physics only optimizes R, t
        else:
            print("  Physics correction skipped (no edges or disabled).")

        # Write output
        out = output_dir or self.config.output_dir
        scene.export(out)
        print(f"\n[CAST] Reconstruction complete! ({len(scene.objects)} objects)")
        return scene

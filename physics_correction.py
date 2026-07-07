"""
Physics-aware correction — Section 5 of CAST.

Motivated by rigid-body simulation, CAST formulates a simplified
optimisation problem over object poses to enforce two physical
constraint types extracted from the relation graph:

  - Contact  : bidirectional, no penetration + at least one contact point
  - Support  : unilateral (supporter static, supported optimised)

The cost functions are defined via Signed Distance Fields (SDF) computed
by Open3D, with gradients obtained through PyTorch autodiff.

Key design choices (Section 5.1):
  - Custom "simulation" rather than off-the-shelf rigid-body solver, because:
      1. Partial scene (some objects may be missing)
      2. Imperfect geometries (convex decomposition is brittle)
      3. Initial penetrations (destabilise standard solvers)
  - Only optimises R, t (not full dynamics); snapshots at current timestep
    need to be physically plausible, not dynamically stable over time.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import open3d as o3d


# ============================================================================
# 1. SDF computation via Open3D
# ============================================================================

def compute_sdf_for_mesh(mesh: o3d.geometry.TriangleMesh,
                         query_points: np.ndarray) -> np.ndarray:
    """
    Compute signed distances from query points to a mesh surface.

    Uses Open3D's ray-casting-based SDF. Positive = outside, negative = inside.

    Args:
        mesh:         Open3D TriangleMesh
        query_points: (N, 3) world-space points

    Returns:
        (N,) SDF values
    """
    if len(mesh.vertices) == 0:
        return np.full(len(query_points), 1e6, dtype=np.float32)

    # Ensure mesh is watertight-ish for SDF
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    queries = o3d.core.Tensor(query_points.astype(np.float32))
    sdf = scene.compute_signed_distance(queries)
    return sdf.numpy()


def sample_surface_points(mesh: o3d.geometry.TriangleMesh,
                          num_points: int = 2048) -> np.ndarray:
    """Uniformly sample points on mesh surface (rest pose)."""
    if len(mesh.vertices) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)


# ============================================================================
# 2. Transform helpers
# ============================================================================

def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation (Zhou et al. 2019) to 3x3 matrix.
    Stable, continuous representation suitable for gradient-based optimisation.

    Args:
        r6d: (B, 6) or (6,)
    Returns:
        R:   (B, 3, 3) or (3, 3)
    """
    if r6d.dim() == 1:
        r6d = r6d.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    a1 = r6d[:, :3]
    a2 = r6d[:, 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)

    if squeeze:
        R = R.squeeze(0)
    return R


def transform_points(pts: torch.Tensor,
                     R: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
    """
    Apply rigid transform: out = pts @ R.T + t

    Args:
        pts: (N, 3) or (B, N, 3)
        R:   (3, 3) or (B, 3, 3)
        t:   (3,)   or (B, 3)
    Returns:
        (N, 3) or (B, N, 3)
    """
    if pts.dim() == 2 and R.dim() == 2:
        return pts @ R.T + t
    elif pts.dim() == 3 and R.dim() == 3:
        return torch.bmm(pts, R.transpose(1, 2)) + t.unsqueeze(1)
    else:
        raise ValueError(f"Shape mismatch: pts {pts.shape}, R {R.shape}, t {t.shape}")


# ============================================================================
# 3. Cost functions (Eq. 9, 10, 11)
# ============================================================================

def contact_cost(surface_i: torch.Tensor,
                 sdf_i_fn,           # callable: (Tensor[N,3]) -> Tensor[N] SDF of mesh i
                 surface_j: torch.Tensor,
                 sdf_j_fn,           # callable: (Tensor[N,3]) -> Tensor[N] SDF of mesh j
                 R_i: torch.Tensor, t_i: torch.Tensor,
                 R_j: torch.Tensor, t_j: torch.Tensor,
                 sigma: float = 0.05) -> torch.Tensor:
    """
    Bilateral contact cost (Eq. 9). Penalises penetration and rewards
    at least one near-contact point between the two objects.

    C(T_i, T_j) = C(i→j) + C(j→i)

    where C(i→j) = - Σ D_i(p_j) * I(D_i<0) / Σ I(D_i<0)
                    + max(min D_i(p_j), 0)

    Args:
        surface_i: (N_i, 3) surface points of object i (rest pose)
        sdf_i_fn:  SDF function for mesh i in world space
        surface_j: (N_j, 3) surface points of object j (rest pose)
        sdf_j_fn:  SDF function for mesh j in world space
        R_i, t_i:  transform for object i
        R_j, t_j:  transform for object j
        sigma:     near-surface threshold (paper Eq. 11)
    Returns:
        scalar cost
    """
    # Transform surfaces to world space
    pts_i = transform_points(surface_i, R_i, t_i)
    pts_j = transform_points(surface_j, R_j, t_j)

    # SDF of i at points of j
    d_i_at_j = sdf_i_fn(pts_j)   # (N_j,)
    # SDF of j at points of i
    d_j_at_i = sdf_j_fn(pts_i)   # (N_i,)

    def _directional(d_at_other):
        penetration = d_at_other[d_at_other < 0]
        if penetration.numel() == 0:
            pen_term = torch.tensor(0.0, device=d_at_other.device)
        else:
            pen_term = -penetration.mean()
        # Minimum distance (if > 0, objects are separated → penalty)
        sep_term = torch.clamp(d_at_other.min(), min=0)
        return pen_term + sep_term

    return _directional(d_i_at_j) + _directional(d_j_at_i)


def support_cost(surface_supported: torch.Tensor,
                 sdf_supporter_fn,     # static object SDF
                 R_supported: torch.Tensor,
                 t_supported: torch.Tensor,
                 sigma: float = 0.05) -> torch.Tensor:
    """
    Unilateral support cost (Eq. 10). The supporter is static; only the
    supported object is optimised.

    For flat surfaces like ground, additionally regularises near-surface
    SDF values (Eq. 11) to encourage close contact.
    """
    pts = transform_points(surface_supported, R_supported, t_supported)
    d = sdf_supporter_fn(pts)   # (N,)

    # Penetration penalty
    penetration = d[d < 0]
    pen_term = -penetration.mean() if penetration.numel() > 0 else torch.tensor(0.0, device=d.device)

    # Separation penalty — but only within sigma band
    near_surface = d[(d > 0) & (d < sigma)]
    if near_surface.numel() > 0:
        close_contact = near_surface.mean()
    else:
        # If nothing is close, penalise minimum distance
        close_contact = d.min()

    return pen_term + close_contact


# ============================================================================
# 4. Constraint graph → optimisable cost graph
# ============================================================================

def build_constraint_graph(relation_graph: Dict,
                           objects: List,
                           meshes: List[o3d.geometry.TriangleMesh],
                           surface_samples: Dict[int, torch.Tensor]) -> Dict:
    """
    Convert the relation graph (from GPT-4V) into an optimisation-ready
    constraint graph that maps each edge to a specific cost function.

    Args:
        relation_graph:  {'nodes', 'contact_edges', 'support_edges'}
        objects:         list of ObjectInfo
        meshes:          list of Open3D meshes (one per object)
        surface_samples: {obj_id: Tensor (N, 3)} pre-sampled surface points

    Returns:
        {
            'variables':  {id: {'R_6d': Tensor, 't': Tensor}},  # optimisable
            'contacts':   [(i, j, cost_fn), ...],
            'supports':   [(supporter, supported, cost_fn), ...],
        }
    """
    constraints = {
        'variables': {},
        'contacts': [],
        'supports': [],
    }

    # Initialise optimisable variables from current estimates
    for obj in objects:
        # Convert initial R (3x3) → 6D representation
        # If no initial transform, use identity
        R_6d = torch.zeros(6, dtype=torch.float32)
        R_6d[0] = 1.0  # identity → axis-aligned
        R_6d[4] = 1.0
        t = torch.from_numpy(obj.point_cloud.mean(axis=0)).float() \
            if len(obj.point_cloud) > 0 else torch.zeros(3)

        constraints['variables'][obj.id] = {
            'R_6d': R_6d.clone().detach().requires_grad_(True),
            't': t.clone().detach().requires_grad_(True),
        }

    # Build SDF caches (computed once per optimisation step)
    # For each edge, we wrap SDF computation in a closure
    for (i, j) in relation_graph.get('contact_edges', []):
        if i not in surface_samples or j not in surface_samples:
            continue
        si = surface_samples[i]
        sj = surface_samples[j]

        def make_contact_fn(_i, _j, _si, _sj):
            def fn(vars):
                R_i = rotation_6d_to_matrix(vars[_i]['R_6d'])
                R_j = rotation_6d_to_matrix(vars[_j]['R_6d'])
                # SDF closures
                def sdf_i(q): return compute_sdf_for_mesh(meshes[_i],
                                        q.detach().cpu().numpy())
                def sdf_j(q): return compute_sdf_for_mesh(meshes[_j],
                                        q.detach().cpu().numpy())
                return contact_cost(_si, sdf_i, _sj, sdf_j,
                                    R_i, vars[_i]['t'], R_j, vars[_j]['t'])
            return fn

        constraints['contacts'].append((i, j, make_contact_fn(i, j, si, sj)))

    for (supporter, supported) in relation_graph.get('support_edges', []):
        if supported not in surface_samples or supporter not in surface_samples:
            continue
        s_sup = surface_samples[supported]

        def make_support_fn(_supp, _supd, _ss):
            def fn(vars):
                R = rotation_6d_to_matrix(vars[_supd]['R_6d'])
                def sdf_supp(q): return compute_sdf_for_mesh(meshes[_supp],
                                        q.detach().cpu().numpy())
                return support_cost(_ss, sdf_supp, R, vars[_supd]['t'])
            return fn

        constraints['supports'].append((supporter, supported,
                                        make_support_fn(supporter, supported, s_sup)))

    return constraints


# ============================================================================
# 5. Optimisation loop (Eq. 8)
# ============================================================================

def optimize_poses(meshes: List[o3d.geometry.TriangleMesh],
                   objects: List,
                   relation_graph: Dict,
                   steps: int = 200,
                   lr: float = 0.01,
                   sigma: float = 0.05,
                   num_samples: int = 2048,
                   verbose: bool = True) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Run the physics-aware correction optimisation (Sec. 5.2–5.4).

    Args:
        meshes:         list of Open3D meshes (canonical space, one per object)
        objects:        list of ObjectInfo
        relation_graph: output of GPT-4V relation reasoning
        steps:          optimisation steps
        lr:             learning rate
        sigma:          near-surface threshold
        num_samples:    surface points per object
        verbose:        print progress

    Returns:
        {obj_id: (R, t, s)} optimised transforms (scale s is fixed to 1.0
        for the physics correction; scale comes from AlignGen).
    """
    if not relation_graph.get('contact_edges') and not relation_graph.get('support_edges'):
        if verbose:
            print("[PhysicsCorrection] No relation edges — skipping optimisation.")
        return {i: (np.eye(3), np.zeros(3), 1.0) for i in range(len(objects))}

    # 1. Pre-sample surface points at rest pose
    surface_samples = {}
    for i, mesh in enumerate(meshes):
        pts = sample_surface_points(mesh, num_samples)
        surface_samples[i] = torch.from_numpy(pts).float()

    # 2. Build constraint graph
    constraints = build_constraint_graph(relation_graph, objects, meshes, surface_samples)

    # 3. Set up optimiser
    params = []
    for v in constraints['variables'].values():
        params.append(v['R_6d'])
        params.append(v['t'])
    optim = torch.optim.Adam(params, lr=lr)

    # 4. Optimisation loop
    for step in range(steps):
        optim.zero_grad()
        total_loss = torch.tensor(0.0)

        for (i, j, cost_fn) in constraints['contacts']:
            total_loss = total_loss + cost_fn(constraints['variables'])

        for (supporter, supported, cost_fn) in constraints['supports']:
            total_loss = total_loss + cost_fn(constraints['variables'])

        if torch.is_tensor(total_loss) and total_loss > 0:
            total_loss.backward()
            optim.step()

        if verbose and step % 50 == 0 and torch.is_tensor(total_loss):
            print(f"  [Physics] step {step:4d}/{steps}  loss={total_loss.item():.6f}")

    # 5. Extract final transforms
    results = {}
    for obj_id, var in constraints['variables'].items():
        R = rotation_6d_to_matrix(var['R_6d']).detach().cpu().numpy()
        t = var['t'].detach().cpu().numpy()
        results[obj_id] = (R, t, 1.0)

    return results


# ============================================================================
# 6. Ground / floor regularisation
# ============================================================================

def create_ground_plane(ground_height: float = 0.0) -> o3d.geometry.TriangleMesh:
    """Create a large ground plane for floor-support regularisation."""
    ground = o3d.geometry.TriangleMesh.create_box(width=20.0, height=0.02, depth=20.0)
    ground.translate([-10.0, ground_height - 0.02, -10.0])
    return ground

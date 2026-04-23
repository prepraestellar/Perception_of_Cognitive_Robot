import numpy as np
from scipy.optimize import least_squares


def normalize_angle(angle):
    """Normalize angle to [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PoseGraphEdge:
    """Represents a constraint between two poses or a pose and an observation."""
    def __init__(self, pose_i_idx, pose_j_idx, measurement, information_matrix, edge_type="odometry"):
        """
        Args:
            pose_i_idx: Index of first pose in pose list
            pose_j_idx: Index of second pose (None for loop closure observations)
            measurement: Measured relative pose (dx, dy, dtheta) or observation vector
            information_matrix: Inverse covariance (weights the constraint)
            edge_type: "odometry" or "loop_closure"
        """
        self.pose_i_idx = pose_i_idx
        self.pose_j_idx = pose_j_idx
        self.measurement = np.array(measurement)
        self.information_matrix = np.array(information_matrix)
        self.edge_type = edge_type


class GraphSLAM:
    """
    Pose-graph SLAM using scipy.optimize.least_squares for nonlinear optimization.
    Optimizes robot trajectory given odometry constraints and loop closures.
    """

    def __init__(self):
        self.poses = []  # List of [x, y, theta] poses
        self.edges = []  # List of PoseGraphEdge constraints
        self.optimized = False

    def add_pose(self, x, y, theta):
        """Add a new pose to the graph (typically from odometry prediction)."""
        self.poses.append(np.array([x, y, theta], dtype=np.float64))
        self.optimized = False

    def add_odometry_edge(self, pose_i_idx, pose_j_idx, dx, dy, dtheta, information_matrix=None):
        """
        Add odometry constraint between consecutive poses.
        
        Args:
            pose_i_idx: Index of first pose
            pose_j_idx: Index of second pose (typically i+1)
            dx, dy, dtheta: Measured relative motion
            information_matrix: 3x3 information matrix (default: identity)
        """
        if information_matrix is None:
            information_matrix = np.eye(3)
        
        edge = PoseGraphEdge(
            pose_i_idx, pose_j_idx,
            [dx, dy, dtheta],
            information_matrix,
            edge_type="odometry"
        )
        self.edges.append(edge)
        self.optimized = False

    def add_loop_closure_edge(self, pose_i_idx, pose_j_idx, dx, dy, dtheta, information_matrix=None):
        """
        Add loop closure constraint (detected revisit of previous location).
        
        Args:
            pose_i_idx: Index of current pose
            pose_j_idx: Index of previous pose (likely far away)
            dx, dy, dtheta: Estimated relative pose based on visual/map similarity
            information_matrix: 3x3 information matrix (default: identity * loop_closure_confidence)
        """
        if information_matrix is None:
            # Loop closures are typically less certain, so lower weight
            information_matrix = np.eye(3) * 0.5
        
        edge = PoseGraphEdge(
            pose_i_idx, pose_j_idx,
            [dx, dy, dtheta],
            information_matrix,
            edge_type="loop_closure"
        )
        self.edges.append(edge)
        self.optimized = False

    def _relative_pose_error(self, pose_i, pose_j, measurement):
        """
        Compute residual for relative pose constraint.
        
        Args:
            pose_i: [x, y, theta] of first pose
            pose_j: [x, y, theta] of second pose
            measurement: [dx, dy, dtheta] expected relative pose
        
        Returns:
            3-element residual vector
        """
        # Transform pose_j into pose_i's frame
        dx_actual = pose_j[0] - pose_i[0]
        dy_actual = pose_j[1] - pose_i[1]
        dtheta_actual = normalize_angle(pose_j[2] - pose_i[2])
        
        # Rotate into pose_i's frame
        cos_theta = np.cos(pose_i[2])
        sin_theta = np.sin(pose_i[2])
        dx_local = cos_theta * dx_actual + sin_theta * dy_actual
        dy_local = -sin_theta * dx_actual + cos_theta * dy_actual
        
        # Residual: expected minus actual
        residual = np.array([
            measurement[0] - dx_local,
            measurement[1] - dy_local,
            normalize_angle(measurement[2] - dtheta_actual)
        ])
        
        return residual

    def _residuals_for_optimization(self, pose_vector):
        """
        Compute all residuals from edges for scipy.optimize.least_squares.
        
        Args:
            pose_vector: Flattened vector [x0, y0, theta0, x1, y1, theta1, ..., xN, yN, thetaN]
        
        Returns:
            Weighted residual vector
        """
        n_poses = len(self.poses)
        all_residuals = []
        
        for edge in self.edges:
            # Reshape pose vector into pose list
            pose_i = pose_vector[edge.pose_i_idx * 3:(edge.pose_i_idx + 1) * 3]
            pose_j = pose_vector[edge.pose_j_idx * 3:(edge.pose_j_idx + 1) * 3]
            
            # Compute residual
            residual = self._relative_pose_error(pose_i, pose_j, edge.measurement)
            
            # Weight by information matrix (Cholesky decomposition for numerical stability)
            try:
                L = np.linalg.cholesky(edge.information_matrix)
                weighted_residual = L @ residual
            except np.linalg.LinAlgError:
                # If not positive definite, use sqrt of diagonal
                weighted_residual = np.sqrt(np.diag(edge.information_matrix)) * residual
            
            all_residuals.extend(weighted_residual)
        
        return np.array(all_residuals)

    
    def optimize(self, max_iterations=20, verbose=False):
        """
        Optimize pose graph using scipy.optimize.least_squares.
        Always optimizes regardless of graph size (uses tight iteration limits for speed).
        
        Args:
            max_iterations: Maximum optimizer iterations (default: 20 for fast remapping)
            verbose: Print optimization progress
        
        Returns:
            Optimization result
        """
        if len(self.poses) < 2:
            if verbose:
                print("[GraphSLAM] Not enough poses to optimize (need ≥2)")
            return None
        
        if len(self.edges) == 0:
            if verbose:
                print("[GraphSLAM] No edges to optimize")
            return None
        
        x0 = np.array([p for pose in self.poses for p in pose])
        
        # Optimize with very tight limits for fast convergence
        result = least_squares(
            self._residuals_for_optimization,
            x0,
            max_nfev=max_iterations,
            ftol=1e-2,
            xtol=1e-2,
            gtol=1e-2,
            verbose=0
        )
        
        # Update poses with optimized values
        for i, pose in enumerate(self.poses):
            self.poses[i] = result.x[i * 3:(i + 1) * 3]
        
        self.optimized = True
        
        if verbose:
            print(f"[GraphSLAM] Optimized: cost={result.cost:.4f}, poses={len(self.poses)}, edges={len(self.edges)}")
        
        return result

    def get_pose(self, idx):
        """Get pose at index."""
        if 0 <= idx < len(self.poses):
            return tuple(self.poses[idx])
        return None

    def get_all_poses(self):
        """Return list of all optimized poses."""
        return [tuple(p) for p in self.poses]

    def get_trajectory(self):
        """Return trajectory as list of (x, y, theta) tuples."""
        return self.get_all_poses()

    def predict_pose(self, prev_pose, control_input):
        """
        Predict next pose from odometry.
        
        Args:
            prev_pose: (x, y, theta) tuple
            control_input: (v, omega, dt) - linear velocity, angular velocity, time step
        
        Returns:
            (x, y, theta) predicted pose
        """
        v, omega, dt = control_input
        dx = v * np.cos(prev_pose[2]) * dt
        dy = v * np.sin(prev_pose[2]) * dt
        dtheta = omega * dt
        return (
            prev_pose[0] + dx,
            prev_pose[1] + dy,
            normalize_angle(prev_pose[2] + dtheta)
        )

    def num_poses(self):
        """Return number of poses in graph."""
        return len(self.poses)

    def num_edges(self):
        """Return number of edges in graph."""
        return len(self.edges)

    def is_optimized(self):
        """Return whether graph has been optimized."""
        return self.optimized

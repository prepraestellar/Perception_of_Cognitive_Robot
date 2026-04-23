"""
Graph-based map representation and A* pathfinding.
"""
import math
import heapq


class GraphMap:
    """
    Dictionary-based occupancy map.
    Key   = (grid_x, grid_y) integer tuple
    Value = occupancy flag (0.0 = free, 1.0 = wall)
    Grows in any direction — no fixed size.
    """

    def __init__(self, resolution=0.05):
        self.nodes = {}
        self.resolution = resolution

    def world_to_grid(self, wx, wy):
        """Convert world coordinates to grid indices."""
        gx = int(math.floor(wx / self.resolution))
        gy = int(math.floor(wy / self.resolution))
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        """Convert grid indices to world coordinates (center of cell)."""
        wx = gx * self.resolution + self.resolution / 2
        wy = gy * self.resolution + self.resolution / 2
        return (wx, wy)

    def add_to_map(self, wx, wy):
        """Mark a world location as occupied."""
        key = self.world_to_grid(wx, wy)
        self.nodes[key] = 1.0

    def is_occupied(self, gx, gy, threshold=0.8):
        """
        Check if a grid cell is occupied.
        Returns: True = wall, False = free, None = unknown
        """
        val = self.nodes.get((gx, gy))
        if val is None:
            return None
        return val >= threshold


def a_star(start, goal, graph_map, threshold=0.8, max_iter=20000):
    """
    Finds a path from start to goal on the graph map using A* algorithm.
    
    Args:
        start: (gx, gy) tuple
        goal: (gx, gy) tuple
        graph_map: GraphMap instance
        threshold: occupancy threshold for considering a cell as wall
        max_iter: maximum iterations before giving up
    
    Returns:
        List of (gx, gy) waypoints from start to goal, or None if unreachable.
    """
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    iterations = 0

    while open_set:
        iterations += 1
        if iterations > max_iter:
            return None

        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if graph_map.is_occupied(neighbor[0], neighbor[1], threshold) is True:
                continue  # wall — skip
            tentative_g = g_score[current] + math.dist(current, neighbor)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + math.dist(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None  # no path found

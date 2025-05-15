import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import heapq
from matplotlib.colors import ListedColormap, BoundaryNorm

# -------------------------------
# A* Algorithm Implementation
# -------------------------------
def astar(room, start, goal):
    """
    Uses A* search on a 2D grid.
    Parameters:
      room: 2D numpy array where free cells are 0; obstacles are 1 (or 2)
      start, goal: tuples (x, y)
    Returns:
      A list of grid cells (including start and goal) representing the planned path,
      or None if no path is found.
    """
    width, height = room.shape

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        est, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                # Only free cells (value 0) are allowed.
                if room[neighbor] == 0 and neighbor not in visited:
                    new_cost = cost + 1
                    est_new = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (est_new, new_cost, neighbor, path + [neighbor]))
    return None

# -------------------------------
# generate_room Function (Updated)
# -------------------------------
def generate_room(size, num_obstacles=30):
    """
    Generates a room as a grid with:
      - Free cells as 0.
      - Random single-cell obstacles (value 1).
      - Closed (hollow) objects added:
          • Their border is filled with obstacles (value 1).
          • Their interior is marked with 2 (inaccessible).

    The robot will only plan in cells equal to 0.

    Parameters:
      size: Tuple (width, height) of the room.
      num_obstacles: Number of random single-cell obstacles.

    Returns:
      A 2D numpy array representing the room.
    """
    width, height = size
    room = np.zeros((width, height), dtype=int)

    # Add random single-cell obstacles (avoid borders).
    for _ in range(num_obstacles):
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
        room[x, y] = 1

    # Add closed (hollow) objects.
    num_closed_objects = 2  # Adjust as desired.
    for _ in range(num_closed_objects):
        # Choose random dimensions between 3x3 and 6x6.
        obj_width = random.randint(3, 6)
        obj_height = random.randint(3, 6)
        # Make sure the object fits inside the room (leaving a margin).
        max_x = width - obj_height - 1
        max_y = height - obj_width - 1
        if max_x < 1 or max_y < 1:
            continue  # Skip if the object would not fit.
        top_left_x = random.randint(1, max_x)
        top_left_y = random.randint(1, max_y)

        # Draw the object's border (value 1).
        room[top_left_x, top_left_y:top_left_y+obj_width] = 1                     # Top border.
        room[top_left_x + obj_height - 1, top_left_y:top_left_y+obj_width] = 1        # Bottom border.
        room[top_left_x:top_left_x+obj_height, top_left_y] = 1                        # Left border.
        room[top_left_x:top_left_x+obj_height, top_left_y+obj_width - 1] = 1          # Right border.

        # Mark interior as inaccessible (value 2) if the object is large enough.
        if obj_width > 2 and obj_height > 2:
            room[top_left_x+1:top_left_x+obj_height-1, top_left_y+1:top_left_y+obj_width-1] = 2

    return room

# -------------------------------
# Robot & SLAM Class with Built Map
# -------------------------------
class Robot:
    def __init__(self, room, start):
        self.room = room
        self.x, self.y = start             # Current cell (must be free: value 0).
        self.path = [start]                # Actual (ground-truth) path.
        self.visited = np.zeros(room.shape, dtype=bool)
        self.visited[start] = True
        self.planned_path = []             # Current planned route (list of cells).
        # Built occupancy grid map from LiDAR scans (-1: unknown, 0: free, 1: obstacle).
        # It starts from an unknown state (all cells -1) and is updated cumulatively.
        self.built_map = np.full(room.shape, -1, dtype=int)
        self.lidar_range = 5               # Maximum LiDAR range.

        # Optional: pose graph (for illustration).
        self.pose_graph = nx.Graph()
        self.pose_counter = 0
        self.pose_graph.add_node(self.pose_counter, pos=start)

    def get_free_neighbors(self):
        """
        Returns a list of 4-connected neighboring cells that are free (room value == 0).
        """
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx_cell = self.x + dx
            ny_cell = self.y + dy
            if (0 <= nx_cell < self.room.shape[0] and 0 <= ny_cell < self.room.shape[1] and
                self.room[nx_cell, ny_cell] == 0):
                neighbors.append((nx_cell, ny_cell))
        return neighbors

    def move_to(self, next_pos):
        """
        Moves the robot to the given cell and updates its state.
        """
        self.x, self.y = next_pos
        self.path.append(next_pos)
        self.visited[next_pos] = True
        self.pose_counter += 1
        self.pose_graph.add_node(self.pose_counter, pos=next_pos)
        self.pose_graph.add_edge(self.pose_counter - 1, self.pose_counter, weight=1)

    def plan_next_goal(self):
        """
        Chooses the next target cell:
          - Prefer unvisited free cells (value 0).
          - Otherwise, pick a random free cell (excluding the current cell).
        """
        width, height = self.room.shape
        candidates = [(i, j) for i in range(width) for j in range(height)
                      if self.room[i, j] == 0 and not self.visited[i, j]]
        if candidates:
            curr = (self.x, self.y)
            candidates.sort(key=lambda cell: abs(cell[0] - curr[0]) + abs(cell[1] - curr[1]))
            return candidates[0]
        free_cells = [(i, j) for i in range(width) for j in range(height)
                      if self.room[i, j] == 0 and (i, j) != (self.x, self.y)]
        if free_cells:
            return random.choice(free_cells)
        return None

    def plan_path_to_goal(self, goal):
        """
        Uses A* to compute a collision-free path from the current position to the goal.
        """
        start = (self.x, self.y)
        return astar(self.room, start, goal)

    def follow_planned_path(self):
        """
        Follows a planned path (if available) or replans if needed.
        If no valid planned path is found, the robot moves to a random free neighbor.
        This method is called continuously during simulation.
        """
        if self.planned_path and len(self.planned_path) > 1:
            self.planned_path.pop(0)
            next_pos = self.planned_path[0]
            self.move_to(next_pos)
            return

        goal = self.plan_next_goal()
        if goal is not None:
            new_path = self.plan_path_to_goal(goal)
            if new_path is not None and len(new_path) > 1:
                self.planned_path = new_path
                self.planned_path.pop(0)
                next_pos = self.planned_path[0]
                self.move_to(next_pos)
                return

        # Fall back: choose a random free neighbor.
        neighbors = self.get_free_neighbors()
        if neighbors:
            next_pos = random.choice(neighbors)
            self.planned_path = [(self.x, self.y), next_pos]
            self.move_to(next_pos)

    def scan_environment(self):
        """
        Simulates LiDAR scans in the four cardinal directions.
        For each ray:
          - Marks cells as free (0) until an obstacle is encountered (cells with room value != 0).
          - Marks the first encountered obstacle as 1.
        Updates the built_map accordingly in a cumulative way (from -1, unknown, onwards).
        Returns a list of detected obstacle points.
        """
        measurements = []
        angles = [0, 90, 180, 270]
        for angle in angles:
            hit_obstacle = False
            for r in range(1, self.lidar_range + 1):
                lx = self.x + int(round(np.cos(np.radians(angle)) * r))
                ly = self.y + int(round(np.sin(np.radians(angle)) * r))
                if 0 <= lx < self.room.shape[0] and 0 <= ly < self.room.shape[1]:
                    # Any cell not equal to 0 is treated as an obstacle.
                    if self.room[lx, ly] != 0:
                        self.built_map[lx, ly] = 1
                        measurements.append((lx, ly))
                        hit_obstacle = True
                        break
                    else:
                        self.built_map[lx, ly] = 0
                else:
                    break
            if not hit_obstacle:
                for r in range(1, self.lidar_range + 1):
                    lx = self.x + int(round(np.cos(np.radians(angle)) * r))
                    ly = self.y + int(round(np.sin(np.radians(angle)) * r))
                    if 0 <= lx < self.room.shape[0] and 0 <= ly < self.room.shape[1]:
                        self.built_map[lx, ly] = 0
        return measurements

# -------------------------------
# Visualization Function
# -------------------------------
def visualize(room, robot):
    """
    Creates two side-by-side plots:
      - Left: The physical room (with obstacles) and the robot's trajectory.
      - Right: The built occupancy grid map (from LiDAR scans).

    In the room grid:
      0 = free space,
      1 = obstacle (walls, random obstacles),
      2 = inaccessible closed room interior.
    For visualization, both 1 and 2 are treated as obstacles.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Display room layout and robot path.
    axs[0].imshow(room.T, cmap="gray_r", origin='lower', 
                  extent=[0, room.shape[0], 0, room.shape[1]])
    rp = np.array(robot.path)
    if rp.shape[0] > 0:
        axs[0].plot(rp[:, 0] + 0.5, rp[:, 1] + 0.5, 'b.-', label="Robot Path", linewidth=2)
    axs[0].scatter(robot.x + 0.5, robot.y + 0.5, c="red", s=80, label="Robot")
    if robot.planned_path and len(robot.planned_path) > 1:
        pp = np.array(robot.planned_path)
        axs[0].plot(pp[:, 0] + 0.5, pp[:, 1] + 0.5, 'k--', label="Planned Route", linewidth=2)
    axs[0].set_xlim(0, room.shape[0])
    axs[0].set_ylim(0, room.shape[1])
    axs[0].set_title("Room Map & Robot Trajectory")
    axs[0].legend(loc="upper right")

    # Right: Display the built occupancy grid.
    cmap_built = ListedColormap(['lightblue', 'white', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap_built.N)
    axs[1].imshow(robot.built_map.T, cmap=cmap_built, norm=norm, origin='lower',
                   extent=[0, room.shape[0], 0, room.shape[1]])
    axs[1].set_title("Built Map via LiDAR Scans")

    plt.tight_layout()
    plt.show()

# -------------------------------
# Main Simulation Loop
# -------------------------------
def main():
    room_size = (20, 20)
    room = generate_room(room_size, num_obstacles=30)

    # Determine a safe starting point: choose a cell that is free (value 0) and not inside any closed object.
    default_start = (room_size[0] // 2, room_size[1] // 2)
    if room[default_start] != 0:
        free_cells = [(i, j) for i in range(room_size[0]) for j in range(room_size[1]) if room[i, j] == 0]
        if free_cells:
            start = random.choice(free_cells)
        else:
            start = default_start
    else:
        start = default_start

    print(f"Robot starting at {start}")
    robot = Robot(room, start)

    # Compute the number of free cells (reachable areas) in the room.
    total_free = np.sum(room == 0)

    # Run until all reachable free cells have been visited.
    iteration = 0
    max_iterations = 10000  # Safety net in case of unforeseen issues.
    while np.sum(np.logical_and(robot.visited, room == 0)) < total_free and iteration < max_iterations:
        robot.follow_planned_path()
        robot.scan_environment()
        iteration += 1
        if iteration % 100 == 0:
            visited_count = np.sum(np.logical_and(robot.visited, room == 0))
            print(f"Iteration {iteration}: {visited_count}/{total_free} free cells visited.")

    print(f"Mapping complete after {iteration} iterations.")
    visualize(room, robot)

if __name__ == '__main__':
    main()

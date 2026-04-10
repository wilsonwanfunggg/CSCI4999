import pybullet as p
import numpy as np
import random
import math

class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent
        self.cost = 0.0 

class RRTPlanner:
    def __init__(self, robot_id, end_effector_index, lower_limits, upper_limits, obstacle_ids=[], step_size=0.05, max_iter=2000, planar_constraint=False, margin=0.035):
        self.robot_id = robot_id
        self.ee_idx = end_effector_index
        self.lower = np.array(lower_limits)
        self.upper = np.array(upper_limits)
        self.obstacle_ids = obstacle_ids 
        self.step_size = step_size
        self.max_iter = max_iter
        self.planar_constraint = planar_constraint
        self.margin = margin
        
        self.search_radius = self.step_size * 2.0
        
        # --- FIX 1: Increased Probe Radius (Safety Buffer) ---
        # Was 0.04. Increased to 0.06 to enforce a strict gap between cube and wall.
        sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        self.probe_id = p.createMultiBody(0, sphere_col, -1, [0, 0, -10])

    def is_state_valid(self, pos):
        if np.any(pos < self.lower) or np.any(pos > self.upper): return False
        p.resetBasePositionAndOrientation(self.probe_id, pos, [0,0,0,1])
        p.performCollisionDetection()
        contact_points = p.getContactPoints(bodyA=self.probe_id)
        for contact in contact_points:
            if contact[2] in self.obstacle_ids: return False
        return True

    def visualize_path(self, path):
        if path is None or len(path) < 2: return
        for i in range(len(path) - 1):
            p.addUserDebugLine(path[i], path[i+1], [0, 1, 0], 3, 10.0)

    def plan(self, start_pos, goal_pos):
        start_pos = np.array(start_pos)
        goal_pos = np.array(goal_pos)
        best_goal_node = None
        min_cost_to_goal = float('inf')

        try:
            if not self.is_state_valid(start_pos):
                for _ in range(20): 
                    jitter = np.random.uniform(-0.05, 0.05, 3)
                    if self.planar_constraint: jitter[2] = 0 
                    if self.is_state_valid(start_pos + jitter):
                        start_pos = start_pos + jitter
                        break
                else: return None 

            if not self.is_state_valid(goal_pos): return None

            if self.is_state_valid_path(start_pos, goal_pos):
                path = [start_pos, goal_pos]
                self.visualize_path(path)
                return path

            start_node = Node(start_pos); start_node.cost = 0.0
            self.node_list = [start_node]

            for i in range(self.max_iter):
                if random.random() < 0.1: rnd_pos = goal_pos
                else:
                    rnd_pos = np.random.uniform(self.lower, self.upper)
                    if self.planar_constraint: rnd_pos[2] = start_pos[2]

                nearest_node = self.get_nearest(self.node_list, rnd_pos)
                new_node = self.steer(nearest_node, rnd_pos)
                
                if not self.is_state_valid(new_node.pos): continue
                if not self.is_state_valid_path(nearest_node.pos, new_node.pos): continue

                neighbors = self.get_neighbors(new_node.pos, self.search_radius)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + np.linalg.norm(new_node.pos - nearest_node.pos)
                
                for neighbor in neighbors:
                    d = np.linalg.norm(new_node.pos - neighbor.pos)
                    if (neighbor.cost + d < new_node.cost):
                        if self.is_state_valid_path(neighbor.pos, new_node.pos):
                            new_node.parent = neighbor; new_node.cost = neighbor.cost + d

                self.node_list.append(new_node)

                for neighbor in neighbors:
                    if neighbor == new_node.parent: continue
                    d = np.linalg.norm(new_node.pos - neighbor.pos)
                    new_cost = new_node.cost + d
                    if new_cost < neighbor.cost:
                        if self.is_state_valid_path(new_node.pos, neighbor.pos):
                            neighbor.parent = new_node; neighbor.cost = new_cost
                            self.propagate_cost_updates(neighbor)

                dist_to_goal = np.linalg.norm(new_node.pos - goal_pos)
                if dist_to_goal <= self.step_size:
                    if self.is_state_valid_path(new_node.pos, goal_pos):
                        potential_cost = new_node.cost + dist_to_goal
                        if potential_cost < min_cost_to_goal:
                            min_cost_to_goal = potential_cost
                            best_goal_node = Node(goal_pos, parent=new_node)
                            best_goal_node.cost = potential_cost

            if best_goal_node:
                final_path = self.construct_path(best_goal_node)
                self.visualize_path(final_path)
                return final_path
            return None
        finally:
            p.resetBasePositionAndOrientation(self.probe_id, [0, 0, -100], [0, 0, 0, 1])

    def get_nearest(self, node_list, point):
        point_arr = np.array(point)
        return min(node_list, key=lambda node: np.linalg.norm(node.pos - point_arr))

    def get_neighbors(self, point, radius):
        point_arr = np.array(point)
        neighbors = []
        for node in self.node_list:
            if np.linalg.norm(node.pos - point_arr) <= radius: neighbors.append(node)
        return neighbors

    def steer(self, from_node, to_point):
        start_pos = np.array(from_node.pos); target_pos = np.array(to_point)
        direction = target_pos - start_pos; length = np.linalg.norm(direction)
        if length < 1e-5: return Node(start_pos, parent=from_node)
        step = min(self.step_size, length)
        new_pos = start_pos + (direction / length) * step
        return Node(new_pos, parent=from_node)

    def is_state_valid_path(self, pos1, pos2):
        p1 = np.array(pos1); p2 = np.array(pos2)
        dist = np.linalg.norm(p2 - p1)
        
        # --- FIX 2: Higher Resolution Check ---
        # Was 0.01 (1cm). Changed to 0.005 (5mm) to catch grazing collisions.
        steps = int(dist / 0.005) + 1
        
        for i in range(steps):
            alpha = i / steps
            interp_pos = p1 * (1 - alpha) + p2 * alpha
            if not self.is_state_valid(interp_pos): return False
        return True

    def propagate_cost_updates(self, node):
        for n in self.node_list:
            if n.parent == node:
                dist = np.linalg.norm(n.pos - node.pos)
                n.cost = node.cost + dist
                self.propagate_cost_updates(n)

    def construct_path(self, goal_node):
        path = []
        curr = goal_node
        while curr is not None: path.append(curr.pos); curr = curr.parent
        path.reverse()
        return self.smooth_path(path)

    def smooth_path(self, path, iterations=50): 
        if len(path) < 3: return path
        smoothed = list(path)
        for _ in range(iterations):
            if len(smoothed) < 3: break
            i = random.randint(0, len(smoothed) - 2)
            j = random.randint(i + 1, len(smoothed) - 1)
            if j - i <= 1: continue
            p1 = np.array(smoothed[i]); p2 = np.array(smoothed[j])
            if self.is_state_valid_path(p1, p2):
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed
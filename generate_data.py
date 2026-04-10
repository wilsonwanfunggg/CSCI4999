import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
import random
import math
import time

from planner import RRTPlanner

# config 
DATA_DIR = "behavioral_cloning_data"
NUM_EPISODES = 500
STEPS_PER_EPISODE = 20

IMG_WIDTH, IMG_HEIGHT = 224, 224
CUBE_SCALE = 0.05

FULL_LIMITS_LOWER =[-1.5, -1.5, 0.01] 
MAZE_LIMITS_UPPER =[1.5, 1.5, 0.40]

# Geometry Constants
MAZE_REF_START = -1.3 
MAZE_REF_END = 1.3
WALL_FULL_THICKNESS = 0.06 
VISUAL_GAP = 0.00 

def get_random_non_yellow_color():
    while True:
        r = np.random.uniform(0.2, 0.9)
        g = np.random.uniform(0.2, 0.9)
        b = np.random.uniform(0.2, 0.9)
        if r > 0.6 and g > 0.6 and b < 0.4: continue 
        return[r, g, b, 1.0]

# --- CRITICAL FIX 1: Returning Unit Vectors for the AI ---
def get_optimal_maze_vector(current_pos, path_future, planner, max_dist=0.08):
    best_target = None
    for pt in path_future:
        pt_arr = np.array(pt)
        dist = np.linalg.norm(pt_arr[:2] - np.array(current_pos)[:2])
        if dist > max_dist: break 
        if planner.is_state_valid_path(current_pos, pt):
            best_target = pt_arr
        else:
            break 
            
    if best_target is not None:
        vec = best_target[:2] - np.array(current_pos)[:2]
        norm = np.linalg.norm(vec)
        if norm > 1e-5:
            return vec / norm  # UNIT VECTOR
    return None

def get_safe_vector_for_command(current_pos, command, planner, max_dist=0.15):
    directions = {
        "push forward":[1, 0], 
        "push backward": [-1, 0],
        "push left":     [0, 1],  
        "push right":    [0, -1]
    }
    if command not in directions: return None
    
    unit_vec = np.array(directions[command])
    start_pos = np.array(current_pos)
    start_pos[2] = 0.025 
    
    safe_dist = 0.0
    step_size = 0.01 
    
    for d in np.arange(step_size, max_dist + step_size, step_size):
        test_pos = start_pos.copy()
        test_pos[0] += unit_vec[0] * d
        test_pos[1] += unit_vec[1] * d
        if planner.is_state_valid(test_pos): safe_dist = d
        else: break
            
    if safe_dist < 0.03: return None
    return unit_vec  # UNIT VECTOR

# ============================================================================
# BASE GEOMETRY
# ============================================================================
def create_arc_wall_mesh(radius_center, start_ang, end_ang, thickness, height, color):
    num_segments = 60
    angle_step = (end_ang - start_ang) / num_segments
    vertices =[]
    indices =[]
    r_in = radius_center - (thickness / 2.0)
    r_out = radius_center + (thickness / 2.0)
    
    for i in range(num_segments + 1):
        theta = start_ang + (i * angle_step)
        c, s = math.cos(theta), math.sin(theta)
        vertices.extend([[r_in * c, r_in * s, 0], [r_in * c, r_in * s, height],[r_out * c, r_out * s, 0],[r_out * c, r_out * s, height]])

    for i in range(num_segments):
        base = i * 4
        indices.extend([base+2, base+6, base+3, base+3, base+6, base+7])
        indices.extend([base+0, base+1, base+4, base+1, base+5, base+4])
        indices.extend([base+1, base+3, base+5, base+3, base+7, base+5])
        indices.extend([base+0, base+4, base+2, base+2, base+4, base+6])

    indices.extend([0, 1, 2, 1, 3, 2])
    last = num_segments * 4
    indices.extend([last+0, last+2, last+1, last+1, last+2, last+3])

    col_shape = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=color, specularColor=[0,0,0])
    return p.createMultiBody(0, col_shape, vis_shape, [0,0,0])

def create_exclusion_cylinder(radius_exact):
    height = 0.5
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius_exact, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius_exact, length=height, rgbaColor=[1, 1, 1, 0])
    return p.createMultiBody(0, col, vis,[0, 0, height/2], [0, 0, 0, 1])

def create_invisible_end_cap(angle, r_inner, r_outer):
    height = 0.5
    mid_r = (r_inner + r_outer) / 2.0
    x, y = mid_r * math.cos(angle), mid_r * math.sin(angle)
    length = (r_outer - r_inner) + 0.05
    thickness = 0.02
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, height/2], rgbaColor=[1, 1, 1, 0])
    return p.createMultiBody(0, col, vis,[x, y, height/2], p.getQuaternionFromEuler([0, 0, angle]))

def create_world_boundaries(r_outer):
    ids =[]
    box_half_len, thickness, height, offset = 2.0, 0.1, 0.5, r_outer + 0.2 
    positions = [[offset + thickness, 0, height/2],[-(offset + thickness), 0, height/2],
                 [0, offset + thickness, height/2],[0, -(offset + thickness), height/2]]
    orientations = [[0, 0, 0, 1],[0, 0, 0, 1], p.getQuaternionFromEuler([0,0,1.57]), p.getQuaternionFromEuler([0,0,1.57])]
    for pos, orn in zip(positions, orientations):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, box_half_len, height])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, box_half_len, height], rgbaColor=[0,0,1,0.0])
        ids.append(p.createMultiBody(0, col, vis, pos, orn))
    return ids

# ============================================================================
# CUHK MAZE ELEMENTS (Directly from your older version)
# ============================================================================
def create_3d_letter_C(center_x, center_y, scale=0.06, height=0.04, color=[0.7, 0.2, 0.2, 1.0]):
    ids =[]
    t = scale * 0.3  
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x - scale/2 + t/2, center_y, height/2]))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x, center_y + scale/2 - t/2, height/2]))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x, center_y - scale/2 + t/2, height/2]))
    return ids

def create_3d_letter_U(center_x, center_y, scale=0.06, height=0.04, color=[0.2, 0.6, 0.2, 1.0]):
    ids =[]
    t = scale * 0.3
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis, [center_x - scale/2 + t/2, center_y, height/2]))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x + scale/2 - t/2, center_y, height/2]))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis, [center_x, center_y - scale/2 + t/2, height/2]))
    return ids

def create_3d_letter_H(center_x, center_y, scale=0.06, height=0.04, color=[0.2, 0.2, 0.7, 1.0]):
    ids =[]
    t = scale * 0.3
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x - scale/2 + t/2, center_y, height/2]))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x + scale/2 - t/2, center_y, height/2]))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[scale/2, t/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x, center_y, height/2]))
    return ids

def create_3d_letter_K(center_x, center_y, scale=0.06, height=0.04, color=[0.7, 0.5, 0.1, 1.0]):
    ids =[]
    t = scale * 0.3
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t/2, scale/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis, [center_x - scale/2 + t/2, center_y, height/2]))
    diag_len = scale * 0.5
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[diag_len/2, t/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[diag_len/2, t/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x + scale*0.1, center_y + scale*0.25, height/2],
                                  p.getQuaternionFromEuler([0, 0, 0.7])))
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[diag_len/2, t/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[diag_len/2, t/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis,[center_x + scale*0.1, center_y - scale*0.25, height/2],
                                  p.getQuaternionFromEuler([0, 0, -0.7])))
    return ids

def create_flat_box(cx, cy, half_x, half_y, height=0.035, color=[0.3, 0.3, 0.3, 1.0], angle=0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_x, half_y, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_x, half_y, height/2], rgbaColor=color)
    orn = p.getQuaternionFromEuler([0, 0, angle])
    return p.createMultiBody(0, col, vis, [cx, cy, height/2], orn)

def create_oval_outline(cx, cy, radius_x, radius_y, stroke=0.012, height=0.035, color=[0.3, 0.7, 0.3, 1.0], num_segments=24):
    ids =[]
    for i in range(num_segments):
        ang1 = (i / num_segments) * 2 * math.pi
        ang2 = ((i + 1) / num_segments) * 2 * math.pi
        x1, y1 = cx + radius_x * math.cos(ang1), cy + radius_y * math.sin(ang1)
        x2, y2 = cx + radius_x * math.cos(ang2), cy + radius_y * math.sin(ang2)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        seg_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        seg_ang = math.atan2(y2 - y1, x2 - x1)
        ids.append(create_flat_box(mid_x, mid_y, seg_len/2, stroke/2, height, color, seg_ang))
    return ids

def create_gate_silhouette(cx, cy, angle=0, size=0.12, height=0.01, color=[0.25, 0.25, 0.25, 1.0]):
    ids =[]
    pillar_width, pillar_len, gap = 0.028, size * 0.5, size * 0.45
    beam_thickness, beam_len = 0.022, gap + pillar_width * 4
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    perp_cos, perp_sin = math.cos(angle + math.pi/2), math.sin(angle + math.pi/2)

    beam_x = cx - (pillar_len/2 + beam_thickness/2) * cos_a
    beam_y = cy - (pillar_len/2 + beam_thickness/2) * sin_a
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[beam_thickness/2, beam_len/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[beam_thickness/2, beam_len/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis, [beam_x, beam_y, height/2], p.getQuaternionFromEuler([0, 0, angle])))

    lp_x = cx + (-gap/2 - pillar_width/2) * perp_cos; lp_y = cy + (-gap/2 - pillar_width/2) * perp_sin
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[pillar_len/2, pillar_width/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[pillar_len/2, pillar_width/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis, [lp_x, lp_y, height/2], p.getQuaternionFromEuler([0, 0, angle])))

    rp_x = cx + (gap/2 + pillar_width/2) * perp_cos; rp_y = cy + (gap/2 + pillar_width/2) * perp_sin
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[pillar_len/2, pillar_width/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[pillar_len/2, pillar_width/2, height/2], rgbaColor=color)
    ids.append(p.createMultiBody(0, col, vis, [rp_x, rp_y, height/2], p.getQuaternionFromEuler([0, 0, angle])))
    return ids

def create_pavilion_silhouette(cx, cy, angle=0, size=0.12, height=0.01):
    ids = []
    green_color, red_color =[0.1, 0.55, 0.1, 1.0],[0.8, 0.15, 0.15, 1.0]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    perp_cos, perp_sin = math.cos(angle + math.pi/2), math.sin(angle + math.pi/2)

    roof_height, base_width, leg_len, leg_stroke = size * 0.5, size * 0.75, size * 0.45, 0.015
    apex_x, apex_y = cx - roof_height * cos_a, cy - roof_height * sin_a

    num_slices = 10
    for i in range(num_slices):
        t = (i + 0.5) / num_slices
        slice_x = apex_x + t * (cx - apex_x); slice_y = apex_y + t * (cy - apex_y)
        slice_width = t * base_width
        if slice_width > 0.008:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[roof_height/num_slices, slice_width/2, height/2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[roof_height/num_slices, slice_width/2, height/2], rgbaColor=green_color)
            ids.append(p.createMultiBody(0, col, vis, [slice_x, slice_y, height/2], p.getQuaternionFromEuler([0, 0, angle])))

    ball_radius = 0.015
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=ball_radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ball_radius, length=height, rgbaColor=green_color)
    ids.append(p.createMultiBody(0, col, vis, [apex_x, apex_y, height/2]))

    leg_positions =[-base_width/2 + 0.02, -base_width/5, base_width/5, base_width/2 - 0.02]
    for pos in leg_positions:
        leg_base_x = cx + pos * perp_cos; leg_base_y = cy + pos * perp_sin
        leg_cx = leg_base_x + (leg_len/2) * cos_a; leg_cy = leg_base_y + (leg_len/2) * sin_a
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[leg_len/2, leg_stroke/2, height/2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[leg_len/2, leg_stroke/2, height/2], rgbaColor=red_color)
        ids.append(p.createMultiBody(0, col, vis,[leg_cx, leg_cy, height/2], p.getQuaternionFromEuler([0, 0, angle])))
    return ids

def create_full_width_zigzag(center_ang, radius_inner, radius_outer, height=0.04, color=[0.15, 0.15, 0.15, 1.0]):
    ids =[]
    lane_width = 0.18
    arm_t = 0.03
    gap_ang = 0.15
    block_ang = 0.12

    ang_start = center_ang - (gap_ang / 2) - block_ang
    ang_mid_left = center_ang - (gap_ang / 2)
    ang_mid_right = center_ang + (gap_ang / 2)
    ang_end = center_ang + (gap_ang / 2) + block_ang

    r_in_top_arm, r_out_top_arm = radius_outer - arm_t, radius_outer
    ids.append(create_arc_wall_mesh((r_in_top_arm + r_out_top_arm) / 2.0, ang_start, ang_end, arm_t, height, color))
    
    r_in_top_block, r_out_top_block = radius_inner + arm_t + lane_width, radius_outer - arm_t
    ids.append(create_arc_wall_mesh((r_in_top_block + r_out_top_block) / 2.0, ang_mid_right, ang_end, (r_out_top_block - r_in_top_block), height, color))

    r_in_bot_arm, r_out_bot_arm = radius_inner, radius_inner + arm_t
    ids.append(create_arc_wall_mesh((r_in_bot_arm + r_out_bot_arm) / 2.0, ang_start, ang_end, arm_t, height, color))
    
    r_in_bot_block, r_out_bot_block = radius_inner + arm_t, radius_outer - arm_t - lane_width
    ids.append(create_arc_wall_mesh((r_in_bot_block + r_out_bot_block) / 2.0, ang_start, ang_mid_left, (r_out_bot_block - r_in_bot_block), height, color))
    return ids

def create_uc_water_tower_silhouette(cx, cy, angle=0, size=0.16, height=0.01, color=[0.55, 0.55, 0.55, 1.0]):
    ids =[]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    shaft_len, shaft_width = size * 0.55, size * 0.25
    flare_len, flare_width = size * 0.15, size * 0.45
    top_len, top_width     = size * 0.30, size * 0.70
    
    c1 = -0.225 * size
    x1, y1 = cx + c1 * cos_a, cy + c1 * sin_a
    ids.append(create_flat_box(x1, y1, shaft_len/2, shaft_width/2, height, color, angle))
    
    c2 = 0.125 * size
    x2, y2 = cx + c2 * cos_a, cy + c2 * sin_a
    ids.append(create_flat_box(x2, y2, flare_len/2, flare_width/2, height, color, angle))
    
    c3 = 0.350 * size
    x3, y3 = cx + c3 * cos_a, cy + c3 * sin_a
    ids.append(create_flat_box(x3, y3, top_len/2, top_width/2, height, color, angle))
    return ids

def create_track_silhouette(cx, cy, radius_x=0.10, radius_y=0.07, height=0.003):
    ids = []
    green_color =[0.2, 0.6, 0.2, 1.0]
    track_color =[0.7, 0.4, 0.3, 1.0]
    red_line_color =[0.7, 0.15, 0.15, 1.0]
    white_color =[0.9, 0.9, 0.9, 1.0]

    field_rx = radius_x * 0.65; field_ry = radius_y * 0.65
    track_inner_rx = radius_x * 0.68; track_inner_ry = radius_y * 0.68

    num_rows = 12
    for i in range(num_rows):
        y_offset = (i - num_rows/2 + 0.5) / (num_rows/2) * field_ry
        if abs(y_offset) < field_ry:
            x_half_width = field_rx * math.sqrt(1 - (y_offset/field_ry)**2)
            if x_half_width > 0.005:
                row_x, row_y = cx, cy + y_offset
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[x_half_width, field_ry/num_rows, height/2])
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[x_half_width, field_ry/num_rows, height/2], rgbaColor=green_color)
                ids.append(p.createMultiBody(0, col, vis, [row_x, row_y, height/2]))

    track_width = radius_x * 0.28
    num_track_segments = 28
    for i in range(num_track_segments):
        ang = (i / num_track_segments) * 2 * math.pi
        mid_rx = (track_inner_rx + radius_x) / 2
        mid_ry = (track_inner_ry + radius_y) / 2
        seg_x, seg_y = cx + mid_rx * math.cos(ang), cy + mid_ry * math.sin(ang)
        seg_radial_len = track_width * 0.9
        seg_arc_width = 2 * math.pi * mid_rx / num_track_segments * 1.15
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[seg_radial_len/2, seg_arc_width/2, height/2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[seg_radial_len/2, seg_arc_width/2, height/2], rgbaColor=track_color)
        ids.append(p.createMultiBody(0, col, vis,[seg_x, seg_y, height/2], p.getQuaternionFromEuler([0, 0, ang])))

    line_stroke = 0.008
    ids.extend(create_oval_outline(cx, cy, radius_x, radius_y, line_stroke, height + 0.001, red_line_color, num_segments=32))
    
    lane1_rx = track_inner_rx + track_width * 0.35; lane1_ry = track_inner_ry + track_width * 0.35
    ids.extend(create_oval_outline(cx, cy, lane1_rx, lane1_ry, 0.004, height + 0.001, white_color, num_segments=28))
    
    lane2_rx = track_inner_rx + track_width * 0.7; lane2_ry = track_inner_ry + track_width * 0.7
    ids.extend(create_oval_outline(cx, cy, lane2_rx, lane2_ry, 0.004, height + 0.001, white_color, num_segments=28))

    return ids

def add_cuhk_maze(radius_inner, radius_outer, rotation_offset=0, include_invisible_walls=True, randomize_layout=True):
    obstacle_ids =[]
    WALL_HEIGHT = 0.06
    rail_color =[0.1, 0.1, 0.1, 1.0]

    start_angle = MAZE_REF_START + rotation_offset
    end_angle = MAZE_REF_END + rotation_offset

    obstacle_ids.append(create_arc_wall_mesh(radius_inner, start_angle, end_angle, WALL_FULL_THICKNESS, WALL_HEIGHT, rail_color))
    obstacle_ids.append(create_arc_wall_mesh(radius_outer, start_angle, end_angle, WALL_FULL_THICKNESS, WALL_HEIGHT, rail_color))

    if include_invisible_walls:
        obstacle_ids.append(create_invisible_end_cap(start_angle, radius_inner, radius_outer))
        obstacle_ids.append(create_invisible_end_cap(end_angle, radius_inner, radius_outer))
        obstacle_ids.append(create_exclusion_cylinder(radius_inner + 0.04))

    path_r = (radius_inner + radius_outer) / 2.0
    path_width = radius_outer - radius_inner
    total_angle = end_angle - start_angle

    if randomize_layout:
        obstacle_placements =[]
        num_segments = 8
        segment_size = total_angle / num_segments
        segment_indices = list(range(num_segments))
        np.random.shuffle(segment_indices)
        current_segment = 0
        
        # 1. LETTERS
        letter_scale = np.random.uniform(0.028, 0.045)
        letter_height = np.random.uniform(0.018, 0.032)
        letters =['C', 'U', 'H', 'K']
        np.random.shuffle(letters) 
        
        seg_idx = segment_indices[current_segment]
        letter_start = start_angle + seg_idx * segment_size + np.random.uniform(0, segment_size * 0.3)
        letter_spacing = np.random.uniform(0.11, 0.17)
        current_segment += 1
        
        for i, letter in enumerate(letters):
            ang = letter_start + i * letter_spacing
            if i % 2 == 0: r = radius_inner + path_width * np.random.uniform(0.23, 0.37)
            else: r = radius_outer - path_width * np.random.uniform(0.23, 0.37)
            
            x, y = r * math.cos(ang), r * math.sin(ang)
            
            if letter == 'C': obstacle_ids.extend(create_3d_letter_C(x, y, letter_scale, letter_height))
            elif letter == 'U': obstacle_ids.extend(create_3d_letter_U(x, y, letter_scale, letter_height))
            elif letter == 'H': obstacle_ids.extend(create_3d_letter_H(x, y, letter_scale, letter_height))
            elif letter == 'K': obstacle_ids.extend(create_3d_letter_K(x, y, letter_scale, letter_height))
        
        # 2. GATE
        seg_idx = segment_indices[current_segment]
        gate_ang = start_angle + seg_idx * segment_size + np.random.uniform(0, segment_size * 0.8)
        gate_r = radius_outer - path_width * np.random.uniform(0.53, 0.67)
        gate_x, gate_y = gate_r * math.cos(gate_ang), gate_r * math.sin(gate_ang)
        obstacle_ids.extend(create_gate_silhouette(gate_x, gate_y, angle=gate_ang, size=np.random.uniform(0.08, 0.14), height=0.01, color=[0.25, 0.25, 0.25, 1.0]))
        current_segment += 1
        
        # 3. PAVILION
        seg_idx = segment_indices[current_segment]
        pav_ang = start_angle + seg_idx * segment_size + np.random.uniform(0, segment_size * 0.8)
        pav_r = radius_inner + path_width * np.random.uniform(0.58, 0.72)
        pav_x, pav_y = pav_r * math.cos(pav_ang), pav_r * math.sin(pav_ang)
        obstacle_ids.extend(create_pavilion_silhouette(pav_x, pav_y, angle=pav_ang, size=np.random.uniform(0.07, 0.13)))
        current_segment += 1
        
        # 4. ZIGZAG
        seg_idx = segment_indices[current_segment]
        zz_ang = start_angle + seg_idx * segment_size + np.random.uniform(0, segment_size * 0.8)
        obstacle_ids.extend(create_full_width_zigzag(zz_ang, radius_inner, radius_outer, height=0.04, color=[0.2, 0.2, 0.2, 1.0]))
        current_segment += 1
        
        # 5. WATER TOWER
        seg_idx = segment_indices[current_segment]
        tower_ang = start_angle + seg_idx * segment_size + np.random.uniform(0, segment_size * 0.8)
        tower_r = radius_outer - path_width * np.random.uniform(0.55, 0.69)
        tower_x, tower_y = tower_r * math.cos(tower_ang), tower_r * math.sin(tower_ang)
        obstacle_ids.extend(create_uc_water_tower_silhouette(tower_x, tower_y, angle=tower_ang + math.pi, size=np.random.uniform(0.06, 0.11), height=0.01, color=[0.55, 0.55, 0.55, 1.0]))
        current_segment += 1
        
        # 6. SPORTS GROUND
        seg_idx = segment_indices[current_segment]
        sports_ang = start_angle + seg_idx * segment_size + np.random.uniform(0, segment_size * 0.8)
        sports_r = path_r + np.random.uniform(-0.03, 0.03)
        sports_x, sports_y = sports_r * math.cos(sports_ang), sports_r * math.sin(sports_ang)
        obstacle_ids.extend(create_track_silhouette(sports_x, sports_y, radius_x=np.random.uniform(0.06, 0.10), radius_y=np.random.uniform(0.04, 0.067), height=0.003))
        current_segment += 1
    
    else:
        # FIXED STANDARD LAYOUT
        letter_scale = 0.035
        letter_height = 0.025

        letter_spacing = 0.14
        letter_start = start_angle + total_angle * 0.08
        letters = ['C', 'U', 'H', 'K'] 
        for i, letter in enumerate(letters):
            ang = letter_start + i * letter_spacing
            if i % 2 == 0: r = radius_inner + path_width * 0.30
            else: r = radius_outer - path_width * 0.30
            if letter == 'H': ang += 0.05

            x, y = r * math.cos(ang), r * math.sin(ang)

            if letter == 'C': obstacle_ids.extend(create_3d_letter_C(x, y, letter_scale, letter_height))
            elif letter == 'U': obstacle_ids.extend(create_3d_letter_U(x, y, letter_scale, letter_height))
            elif letter == 'H': obstacle_ids.extend(create_3d_letter_H(x, y, letter_scale, letter_height))
            elif letter == 'K': obstacle_ids.extend(create_3d_letter_K(x, y, letter_scale, letter_height))

        gate_ang = start_angle + total_angle * 0.33
        gate_r = radius_outer - path_width * 0.60
        gate_x, gate_y = gate_r * math.cos(gate_ang), gate_r * math.sin(gate_ang)
        obstacle_ids.extend(create_gate_silhouette(gate_x, gate_y, angle=gate_ang, size=0.11, height=0.01, color=[0.25, 0.25, 0.25, 1.0]))

        pav_ang = start_angle + total_angle * 0.44
        pav_r = radius_inner + path_width * 0.65
        pav_x, pav_y = pav_r * math.cos(pav_ang), pav_r * math.sin(pav_ang)
        obstacle_ids.extend(create_pavilion_silhouette(pav_x, pav_y, angle=pav_ang, size=0.10, height=0.01))

        zz_ang = start_angle + total_angle * 0.62
        obstacle_ids.extend(create_full_width_zigzag(zz_ang, radius_inner, radius_outer, height=0.04, color=[0.2, 0.2, 0.2, 1.0]))

        tower_ang = start_angle + total_angle * 0.78
        tower_r = radius_outer - path_width * 0.62
        tower_x, tower_y = tower_r * math.cos(tower_ang), tower_r * math.sin(tower_ang)
        obstacle_ids.extend(create_uc_water_tower_silhouette(tower_x, tower_y, angle=tower_ang + math.pi, size=0.08, height=0.01, color=[0.55, 0.55, 0.55, 1.0]))

        sports_ang = start_angle + total_angle * 0.90
        sports_r = path_r
        sports_x, sports_y = sports_r * math.cos(sports_ang), sports_r * math.sin(sports_ang)
        obstacle_ids.extend(create_track_silhouette(sports_x, sports_y, radius_x=0.08, radius_y=0.053, height=0.003))

    return obstacle_ids

def get_valid_pos_unified(planner, ang_min, ang_max, r_inner_center, r_outer_center, test_cube_id):
    wall_half_thick = WALL_FULL_THICKNESS / 2.0
    padding = 0.08 
    
    safe_r_min = r_inner_center + wall_half_thick + padding
    safe_r_max = r_outer_center - wall_half_thick - padding
    
    if safe_r_min >= safe_r_max: return None
    
    for _ in range(1000):
        r = np.random.uniform(safe_r_min, safe_r_max)
        theta = np.random.uniform(min(ang_min,ang_max) + 0.1, max(ang_min,ang_max) - 0.1)
        x = r * np.cos(theta); y = r * np.sin(theta)
        z = 0.03 
        
        ray_from =[x, y, 1.0]; ray_to = [x, y, 0.0]
        results = p.rayTest(ray_from, ray_to)
        if results[0][0] > 0 and results[0][0] != test_cube_id: continue
        if not planner.is_state_valid([x, y, 0.025]): continue
            
        p.resetBasePositionAndOrientation(test_cube_id, [x, y, z], [0,0,0,1])
        p.resetBaseVelocity(test_cube_id, [0,0,0],[0,0,0])
        p.performCollisionDetection()
        
        contact_points = p.getContactPoints(bodyA=test_cube_id)
        if any(pt[2] != 0 for pt in contact_points): continue

        for _ in range(15): p.stepSimulation()
            
        final_pos, _ = p.getBasePositionAndOrientation(test_cube_id)
        if final_pos[2] > 0.03: continue

        final_r = np.linalg.norm(final_pos[:2])
        if final_r <= (r_inner_center + 0.04) or final_r >= (r_outer_center - 0.04): continue
            
        final_theta = math.atan2(final_pos[1], final_pos[0])
        check_min = min(ang_min, ang_max) - 0.05
        check_max = max(ang_min, ang_max) + 0.05
        if final_theta < check_min or final_theta > check_max: continue
        
        contact_points = p.getContactPoints(bodyA=test_cube_id)
        if any(pt[2] != 0 for pt in contact_points): continue
            
        return np.array(final_pos)
            
    return None

def get_start_and_goal(planner, r_inner, r_outer, ang_start, ang_end, test_cube_id):
    total_angle = ang_end - ang_start
    sector_size = total_angle * 0.30
    sector_head = (ang_start, ang_start + sector_size)
    sector_tail = (ang_end - sector_size, ang_end)
    
    if random.random() < 0.5: s_ang, g_ang = sector_head, sector_tail
    else: s_ang, g_ang = sector_tail, sector_head
        
    start_pos = get_valid_pos_unified(planner, s_ang[0], s_ang[1], r_inner, r_outer, test_cube_id)
    goal_pos = get_valid_pos_unified(planner, g_ang[0], g_ang[1], r_inner, r_outer, test_cube_id)
    return start_pos, goal_pos

class PhysicsRRTPlanner(RRTPlanner):
    def __init__(self, *args, **kwargs):
        if 'step_size' in kwargs: kwargs['step_size'] = 0.01 
        else: kwargs['step_size'] = 0.01
        if 'margin' in kwargs: kwargs['margin'] = 0.05
        else: kwargs['margin'] = 0.05
        super().__init__(*args, **kwargs)
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
    def __del__(self):
        try: p.removeBody(self.probe_id)
        except: pass

def get_robot_perspective_image(robot_id):
    robot_pos_old, robot_orn_old = p.getBasePositionAndOrientation(robot_id)
    p.resetBasePositionAndOrientation(robot_id,[100, 100, 100], robot_orn_old)
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.6, 0.0, 1.76], 
        cameraTargetPosition=[0.6, 0.0, 0.0], 
        cameraUpVector=[1, 0, 0]
    )
    projection_matrix = p.computeProjectionMatrixFOV(60.0, 1.0, 0.1, 3.1)
    w, h, rgb, depth, seg = p.getCameraImage(IMG_WIDTH, IMG_HEIGHT, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER, shadow=0)
    p.resetBasePositionAndOrientation(robot_id, robot_pos_old, robot_orn_old)
    return rgb, seg

def interpolate_path(path, step_size=0.03):
    if not path or len(path) < 2: return path
    dense_path =[]
    for i in range(len(path) - 1):
        start = np.array(path[i]); end = np.array(path[i+1])
        dist = np.linalg.norm(end - start)
        num_steps = int(max(dist / step_size, 1))
        for t in range(num_steps):
            alpha = t / num_steps
            dense_path.append(start * (1 - alpha) + end * alpha)
    dense_path.append(path[-1])
    return dense_path

# ============================================================================
# MAIN
# ============================================================================
def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR); os.makedirs(os.path.join(DATA_DIR, "images"))
        os.makedirs(os.path.join(DATA_DIR, "masks")); os.makedirs(os.path.join(DATA_DIR, "vectors"))
        os.makedirs(os.path.join(DATA_DIR, "prompts"))

    p.connect(p.DIRECT) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    global_sample_count = 0
    print(f"--- Starting Final Generation (CUHK Theme + Vector AI Logic) ---")

    for episode in range(NUM_EPISODES):
        print(f"\n[Episode {episode}/{NUM_EPISODES}] Starting new episode...")
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], useFixedBase=True)
        robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

        rotation = np.random.uniform(-0.5, 0.5)
        radius_inner = np.random.uniform(0.45, 0.55)
        radius_outer = np.random.uniform(0.85, 0.95)
        
        use_random = np.random.random() < 0.5
        # Call the exact CUHK generator you provided
        obs_ids = add_cuhk_maze(radius_inner, radius_outer, rotation, include_invisible_walls=True, randomize_layout=use_random)
        
        if not use_random:
            print(f"  [Episode {episode}] Using FIXED standardized layout (for test exposure)")
        
        # Planner with margin suitable for passing through CUHK elements
        planner = PhysicsRRTPlanner(
            robot_id=None, end_effector_index=None,
            lower_limits=FULL_LIMITS_LOWER, upper_limits=MAZE_LIMITS_UPPER,
            obstacle_ids=obs_ids, step_size=0.015, max_iter=8000,
            planar_constraint=True, margin=0.08  
        )

        cube_id = p.loadURDF("cube.urdf", basePosition=[0,0,-10], globalScaling=CUBE_SCALE)
        p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0.8, 0, 1]) 

        s_ang_base = MAZE_REF_START + rotation; e_ang_base = MAZE_REF_END + rotation
        
        print(f"  [Episode {episode}] Finding start/goal positions...")
        start_pos, goal_pos = get_start_and_goal(
            planner, radius_inner, radius_outer, s_ang_base, e_ang_base, cube_id
        )
        
        if start_pos is None or goal_pos is None:
            print(f"  [Episode {episode}] FAILED: Could not find valid start/goal positions. Retrying...")
            continue
            
        print(f"  [Episode {episode}] Start/goal found! Planning path...")
        p.resetBasePositionAndOrientation(cube_id, start_pos, [0,0,0,1])
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
        p.createMultiBody(0, -1, goal_vis, basePosition=goal_pos)
        
        path = planner.plan(start_pos, goal_pos)
        if path is None:
            print(f"  [Episode {episode}] FAILED: Path planning failed. Retrying...")
            continue
        
        print(f"  [Episode {episode}] Path found with {len(path)} waypoints. Generating data...")

        dense_path = interpolate_path(path, step_size=0.04)
        
        for i in range(len(dense_path) - 1):
            curr_pos = dense_path[i]
            
            # --- CRITICAL FIX 2: JITTER / RECOVERY TRAINING ---
            if random.random() < 0.6:
                jitter = np.random.uniform(-0.04, 0.04, 3)
                jitter[2] = 0 
                test_pos = curr_pos + jitter
                if planner.is_state_valid(test_pos):
                    curr_pos = test_pos

            random_yaw = np.random.uniform(-np.pi, np.pi)
            p.resetBasePositionAndOrientation(cube_id, curr_pos, p.getQuaternionFromEuler([0, 0, random_yaw]))
            p.performCollisionDetection()
            
            rgb, seg = get_robot_perspective_image(robot_id)
            mask = np.where(seg == cube_id, 255, 0).astype(np.uint8)
            if np.sum(mask) < 50: continue
            bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)

            future_path = dense_path[i+1:]
            
            maze_vector = get_optimal_maze_vector(curr_pos, future_path, planner)
            
            if maze_vector is not None:
                idx = global_sample_count
                cv2.imwrite(os.path.join(DATA_DIR, "images", f"{idx}.png"), bgr_img)
                cv2.imwrite(os.path.join(DATA_DIR, "masks", f"{idx}.png"), mask)
                np.save(os.path.join(DATA_DIR, "vectors", f"{idx}.npy"), maze_vector)
                with open(os.path.join(DATA_DIR, "prompts", f"{idx}.txt"), "w") as f: 
                    f.write("solve the maze")
                global_sample_count += 1
            
            commands = ["push forward", "push backward", "push left", "push right"]
            random.shuffle(commands)
            
            for cmd in commands:
                safe_vec = get_safe_vector_for_command(curr_pos, cmd, planner)
                if safe_vec is not None:
                    idx = global_sample_count
                    cv2.imwrite(os.path.join(DATA_DIR, "images", f"{idx}.png"), bgr_img)
                    cv2.imwrite(os.path.join(DATA_DIR, "masks", f"{idx}.png"), mask)
                    np.save(os.path.join(DATA_DIR, "vectors", f"{idx}.npy"), safe_vec)
                    with open(os.path.join(DATA_DIR, "prompts", f"{idx}.txt"), "w") as f: 
                        f.write(cmd)
                    global_sample_count += 1
            
        print(f"  [Episode {episode}] ✓ Complete! Total Samples: {global_sample_count}")

    p.disconnect()
    print("--- Generation Complete ---")

if __name__ == "__main__":
    main()
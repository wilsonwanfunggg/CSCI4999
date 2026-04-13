import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import time
from threading import Thread, Lock
import os
import math
from transformers import AutoTokenizer, AutoModel

from model import LanguageConditionedUNet
from planner import RRTPlanner
# Import all the beautiful CUHK geometry functions needed to build the exam maze
from generate_data import (add_cuhk_maze, create_arc_wall_mesh, create_exclusion_cylinder, 
                           create_invisible_end_cap, create_world_boundaries, 
                           MAZE_REF_START, MAZE_REF_END, WALL_FULL_THICKNESS)

TEST_MODE = "MODEL_C"  # Change to "MODEL_A" to test Phase 1 baseline

# config
if torch.cuda.is_available(): DEVICE = "cuda"
elif torch.backends.mps.is_available(): DEVICE = "mps"
else: DEVICE = "cpu"

MODEL_PATH = "best_multitask_model.pth"
INITIAL_PROMPT = None

IMG_WIDTH, IMG_HEIGHT = 224, 224
CUBE_SCALE = 0.05
FRANKA_PANDA_ENDEFFECTOR_INDEX = 11
SIM_SLEEP_TIME = 1./240.  

# Robot Control Constants
PRE_PUSH_OFFSET = 0.08  
MIN_SAFE_Z = 0.01
HOVER_Z = 0.30  # High enough to clear the pavilion and water tower!
PUSH_Z = 0.020  
MAX_PLANNING_RETRIES = 3  

# Workspace Limits - WIDENED to safely cover the entire CUHK maze span
FULL_LIMITS_LOWER = [-0.2, -0.9, 0.01] 
ENTRY_LIMITS_UPPER =[1.0, 0.9, 0.85] 

# "Ready" Pose
HOME_JOINT_ANGLES =[0.0, -0.7, 0.0, -2.356, 0.0, 1.571, 0.785] 

# Visual Markers - Kept exact as your older version
START_POS_DEFAULT = [0.2, -0.6, 0.025] 
GOAL_POS_VISUAL =[0.25,  0.77, 0.03] 

class RobotState: IDLE = 0; PERCEIVING_AND_PLANNING = 1; EXECUTING = 2

# helper classes 
class UserInputHandler:
    def __init__(self):
        self.instruction = INITIAL_PROMPT
        self.new_instruction_event = (INITIAL_PROMPT is not None)
        self.lock = Lock(); self.thread = Thread(target=self._run, daemon=True); self.thread.start()
    def _run(self):
        print(f"\n[System] AI Ready. Default Prompt: '{self.instruction}'")
        while True:
            try:
                new_instruction = input()
                if new_instruction:
                    with self.lock: self.instruction = new_instruction; self.new_instruction_event = True
                    print(f"\n[User] Command: '{self.instruction}'")
            except EOFError: break 
    def get_instruction(self):
        with self.lock: self.new_instruction_event = False; return self.instruction
    def has_new_instruction(self):
        with self.lock: return self.new_instruction_event

class RobotController:
    def __init__(self, robotId, timeout_seconds=4.0):
        self.robotId = robotId; self.target_pos = None
        self.target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]); self.at_target = True
        self.timeout_duration = timeout_seconds; self.move_start_time = None
        self.is_pushing = False  
    def set_target(self, pos, orn=None, is_push=False):
        self.target_pos = pos; 
        if orn: self.target_orn = orn
        self.at_target = False; self.move_start_time = time.time()
        self.is_pushing = is_push
    def interrupt(self): self.at_target = True; self.is_pushing = False
    def is_at_target(self): return self.at_target
    def step(self):
        if self.at_target or not p.isConnected(): return
        elapsed = time.time() - self.move_start_time
        curr = p.getLinkState(self.robotId, FRANKA_PANDA_ENDEFFECTOR_INDEX)[0]
        dist = np.linalg.norm(np.array(curr) - np.array(self.target_pos))
        
        tolerance = 0.008 if self.is_pushing else 0.015
        force = 200 if self.is_pushing else 100
        max_vel = 0.8 if self.is_pushing else 1.5  
        
        if dist < tolerance or elapsed > self.timeout_duration: self.at_target = True; self.is_pushing = False; return
        joints = p.calculateInverseKinematics(self.robotId, FRANKA_PANDA_ENDEFFECTOR_INDEX, self.target_pos, self.target_orn)
        for i in range(7): p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=joints[i], force=force, maxVelocity=max_vel)

def add_maze_walls():
    """Create CUHK-themed maze (clean visualization, FIXED STANDARD LAYOUT for evaluation)"""
    radius_inner = 0.50
    radius_outer = 0.90
    rotation_offset = 0.0
    return add_cuhk_maze(radius_inner, radius_outer, rotation_offset, include_invisible_walls=False, randomize_layout=False)

def setup_simulation():
    p.connect(p.GUI); p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0, 0, -9.81); p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], useFixedBase=True)
    robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

    jitter_x = np.random.uniform(-0.03, 0.03)
    jitter_y = np.random.uniform(-0.03, 0.03)
    eval_start_pos =[START_POS_DEFAULT[0] + jitter_x, START_POS_DEFAULT[1] + jitter_y, START_POS_DEFAULT[2]]
    
    objectId = p.loadURDF("cube.urdf", basePosition=eval_start_pos, globalScaling=CUBE_SCALE)
    p.changeVisualShape(objectId, -1, rgbaColor=[1, 0.8, 0, 1])
    p.changeDynamics(objectId, -1, mass=0.5, lateralFriction=0.6)

    p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1]), basePosition=GOAL_POS_VISUAL)
    obs_ids = add_maze_walls()
    p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=HOME_JOINT_ANGLES)
    set_gripper(robotId, 0.0) 
    
    for _ in range(50): p.stepSimulation()
    return robotId, objectId, obs_ids

def get_camera_image(robotId):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    old_pos, old_orn = p.getBasePositionAndOrientation(robotId)
    p.resetBasePositionAndOrientation(robotId, [100, 100, 100], old_orn)
    view_matrix = p.computeViewMatrix(cameraEyePosition=[0.6, 0.0, 1.76], cameraTargetPosition=[0.6, 0.0, 0.0], cameraUpVector=[1, 0, 0])
    proj_matrix = p.computeProjectionMatrixFOV(60.0, 1.0, 0.1, 3.1)
    
    _, _, rgb, _, _ = p.getCameraImage(IMG_WIDTH, IMG_HEIGHT, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER, shadow=0)
    
    p.resetBasePositionAndOrientation(robotId, old_pos, old_orn)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return rgb[:, :, :3]

def set_gripper(robotId, value):
    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, targetPosition=value, force=100)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, targetPosition=value, force=100)

def interpolate_path(path, step_size=0.03):
    if not path or len(path) < 2: return path
    dense =[]
    for i in range(len(path) - 1):
        p1, p2 = np.array(path[i]), np.array(path[i+1]); dist = np.linalg.norm(p2 - p1)
        steps = int(max(dist / step_size, 1))
        for t in range(steps): dense.append(p1 * (1 - t/steps) + p2 * (t/steps))
    dense.append(path[-1])
    return dense

def save_diagnostic_snapshot(rgb_image, heatmap, vector, prompt, output_dir, counter):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    try:
        if heatmap.shape != rgb_image.shape[:2]: heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]))
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        vis = cv2.addWeighted(bgr, 0.6, cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
        cy, cx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        rx = -vector[1]
        ry = vector[0]

        # 1. Normalize the vector first so the arrow is always a consistent, readable length!
        norm = np.linalg.norm([rx, ry])
        if norm > 1e-5:
            rx, ry = rx / norm, ry / norm

        # 2. Multiply by 30 pixels (instead of 200)
        end_x = int(cx + rx * 30) 
        end_y = int(cy - ry * 30) 
        
        # 3. Change thickness from 3 to 1, and tipLength from 0.3 to 0.4 for a cleaner look
        cv2.arrowedLine(vis, (int(cx), int(cy)), (int(end_x), int(end_y)), (255, 255, 255), 1, tipLength=0.4)
        slug = "".join(c for c in prompt if c.isalnum() or c in (" ", "_")).rstrip().replace(" ", "_")[:30]
        cv2.imwrite(os.path.join(output_dir, f"{counter:04d}_{slug}.png"), vis)
    except: pass

AI_DEBUG_ITEMS =[] 

def draw_ai_thought(start_pos, vec_dir, vec_mag):
    global AI_DEBUG_ITEMS
    for item_id in AI_DEBUG_ITEMS:
        p.removeUserDebugItem(item_id)
    AI_DEBUG_ITEMS =[]
    
    end_pos = start_pos + np.array([vec_dir[0], vec_dir[1], 0]) * vec_mag
    line_id = p.addUserDebugLine(start_pos, end_pos, lineColorRGB=[1, 0, 0], lineWidth=4, lifeTime=0)
    text_id = p.addUserDebugText(f"AI: {vec_mag:.2f}m", start_pos + [0,0,0.1], textColorRGB=[1,0,0], textSize=1.5, lifeTime=0)
    
    AI_DEBUG_ITEMS.extend([line_id, text_id])


if __name__ == '__main__':
    robotId, objectId, obs_ids = setup_simulation()
    controller = RobotController(robotId)
    
    print("[System] Closing gripper...")
    set_gripper(robotId, 0.0)
    for _ in range(20): p.stepSimulation()
    
    print("\n[System] Loading Multi-Task Model...")
    model = LanguageConditionedUNet(n_channels=3, n_classes=1, embedding_dim=512).to(DEVICE)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)); model.eval()
    except: print(f"[Error] {MODEL_PATH} not found. Did you run train.py?"); exit()
    
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE); clip.eval()

    planner = RRTPlanner(robotId, FRANKA_PANDA_ENDEFFECTOR_INDEX, FULL_LIMITS_LOWER, ENTRY_LIMITS_UPPER, obs_ids, margin=0.08)
    user_input = UserInputHandler()

    state = RobotState.IDLE; waypoints =[]
    TASK_LOG_DIR = "task_log"; snapshot_counter = 0

    # --- METRIC TRACKING VARIABLES ---
    interventions_count = 0
    start_dist = None
    episode_steps = 0
    # ---------------------------------
    
    # Oscillation detection
    position_history =[]  
    HISTORY_LENGTH = 8  
    OSCILLATION_THRESHOLD = 0.15  
    consecutive_failed_progress = 0  

    try:
        while p.isConnected():
            # =================================================================
            # 1. UNIFIED MANUAL TERMINATION & METRICS EVALUATION
            # =================================================================
            keys = p.getKeyboardEvents()
            gui_reset = ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED
            
            term_reset = False
            with user_input.lock:
                if user_input.new_instruction_event and user_input.instruction is not None:
                    if user_input.instruction.lower().strip() in ['r', 'reset']:
                        term_reset = True
                        user_input.new_instruction_event = False # Consume reset command
                        user_input.instruction = None # Clear it
            
            if gui_reset or term_reset:
                if start_dist is not None and state != RobotState.IDLE:
                    curr_pos_eval, _ = p.getBasePositionAndOrientation(objectId)
                    current_dist = np.linalg.norm(np.array(curr_pos_eval)[:2] - np.array(GOAL_POS_VISUAL)[:2])
                    progress = max(0.0, 1.0 - (current_dist / start_dist))
                    
                    print(f"\n==================================================")
                    print(f"[Metrics] Task MANUALLY TERMINATED / FAILED.")
                    print(f"  -> Total OODA Steps Attempted: {episode_steps}")
                    print(f"  -> Fallback Interventions: {interventions_count}")
                    print(f"  -> Normalized Progress Distance: {progress * 100:.1f}%")
                    print(f"==================================================\n")
                
                print("[System] Resetting environment to Start Position...")
                p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=HOME_JOINT_ANGLES)
                p.resetBasePositionAndOrientation(objectId, START_POS_DEFAULT, [0,0,0,1])
                controller.interrupt()
                state = RobotState.IDLE
                waypoints = []
                
                # Clear metric trackers for next run
                start_dist = None
                episode_steps = 0
                interventions_count = 0
                continue # SKIP the rest of this loop iteration!
            # =================================================================

            # 2. Handle New Instructions (Start a run)
            if user_input.has_new_instruction():
                cmd = user_input.get_instruction()
                if cmd is not None and cmd.lower().strip() not in ['r', 'reset']:
                    controller.interrupt()
                    state = RobotState.PERCEIVING_AND_PLANNING
                    waypoints = []
                    # --- RESET METRICS FOR NEW RUN ---
                    interventions_count = 0
                    episode_steps = 0
                    cube_pos, _ = p.getBasePositionAndOrientation(objectId)
                    start_dist = np.linalg.norm(np.array(cube_pos)[:2] - np.array(GOAL_POS_VISUAL)[:2])
            
            if state == RobotState.EXECUTING and controller.is_at_target() and not waypoints:
                print("  [Loop] Step Done. Re-Perceiving..."); state = RobotState.PERCEIVING_AND_PLANNING

            if state == RobotState.PERCEIVING_AND_PLANNING:
                p.stepSimulation()
                cube_pos, _ = p.getBasePositionAndOrientation(objectId)
                dist_to_goal = np.linalg.norm(np.array(cube_pos) - np.array(GOAL_POS_VISUAL))
                
                position_history.append(np.array(cube_pos[:2]))
                if len(position_history) > HISTORY_LENGTH:
                    position_history.pop(0)
                
                is_oscillating = False
                if len(position_history) >= HISTORY_LENGTH:
                    recent_positions = np.array(position_history)
                    centroid = recent_positions.mean(axis=0)
                    max_deviation = np.max([np.linalg.norm(pos - centroid) for pos in recent_positions])
                    
                    if max_deviation < OSCILLATION_THRESHOLD:
                        is_oscillating = True
                        consecutive_failed_progress += 1
                        print(f"  [Warning] Oscillation detected! (deviation: {max_deviation:.3f}m, count: {consecutive_failed_progress})")
                    else:
                        consecutive_failed_progress = 0
                
                if dist_to_goal < 0.05:
                    print(f"\n[System] SUCCESS! Cube reached goal (Dist: {dist_to_goal:.3f}m)")
                    break 
                
                prompt = user_input.get_instruction()
                if prompt is None:
                    time.sleep(0.1)
                    continue
                
                rgb = get_camera_image(robotId)
                img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)
                
                # --- WORKING INFERENCE LOGIC: Dense Vector Field ---
                with torch.no_grad():
                    txt = tokenizer([prompt], padding=True, return_tensors="pt").to(DEVICE)
                    txt_feat = clip.get_text_features(**txt); txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                    logits, pred_vec = model(img_t, txt_feat)
                    heatmap = torch.sigmoid(logits).squeeze().cpu().numpy()
                    raw_vector_field = pred_vec.squeeze().cpu().numpy() 

                if heatmap.max() > 0.1:
                    obj_pos, _ = p.getBasePositionAndOrientation(objectId)
                    interaction_pos = np.array(obj_pos)
                    
                    if TEST_MODE == "MODEL_A":
                        print("  [Model A] Executing Phase I logic. Blind 15cm push.")
                        # FIX: Make Model A push blindly towards the goal so it makes *some* progress before hitting the wall
                        blind_vec = np.array(GOAL_POS_VISUAL[:2]) - np.array(interaction_pos[:2])
                        unit_vec = blind_vec / np.linalg.norm(blind_vec)
                        safe_dist = 0.15 # Hardcoded 15cm push
                        
                        episode_steps += 1  # <--- FIX: Ensure OODA steps are counted for Model A!
                        
                        push_vec_3d = np.array([unit_vec[0], unit_vec[1], 0.0])
                        start_push = interaction_pos - (push_vec_3d * PRE_PUSH_OFFSET); start_push[2] = PUSH_Z
                        end_push = interaction_pos + (push_vec_3d * safe_dist); end_push[2] = PUSH_Z
                        
                        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
                        waypoints.append(("move", start_push.copy() + [0,0,HOVER_Z], orn, False))
                        waypoints.append(("move", start_push, orn, False))
                        for pt in interpolate_path([start_push, end_push], 0.01): 
                            waypoints.append(("move", pt, orn, True))
                        waypoints.append(("move", end_push.copy() + [0,0,HOVER_Z], orn, False))
                        state = RobotState.EXECUTING

                    else:
                        # Extract vector from the densest point
                        cy, cx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        raw_vector = raw_vector_field[:, cy, cx]
                        
                        pred_dist = np.linalg.norm(raw_vector)
                        if pred_dist < 0.02:
                            print("  [AI] Predicted move too small (<2cm). Holding position."); time.sleep(0.5); continue
                        
                        safe_dist = min(pred_dist, 0.05) 
                        unit_vec = raw_vector / pred_dist
                        
                        # Oscillation recovery
                        if is_oscillating and consecutive_failed_progress >= 3:
                            interventions_count += 1  # <--- METRIC: Triggered Oscillation Evasion
                            goal_vec = np.array(GOAL_POS_VISUAL[:2]) - np.array(interaction_pos[:2])
                            goal_vec = goal_vec / np.linalg.norm(goal_vec)
                            blended_vec = 0.7 * goal_vec + 0.3 * unit_vec
                            unit_vec = blended_vec / np.linalg.norm(blended_vec)
                            
                            print(f"  [Recovery] Oscillation detected! Biasing toward goal: [{unit_vec[0]:.2f}, {unit_vec[1]:.2f}]")
                            safe_dist = min(safe_dist * 1.5, 0.08)
                            
                            if consecutive_failed_progress >= 5:
                                position_history.clear(); consecutive_failed_progress = 0
                        
                        print(f"  [AI] Vector: [{unit_vec[0]:.2f}, {unit_vec[1]:.2f}] | Speed/Dist: {safe_dist:.3f}m")
                        episode_steps += 1  # <--- METRIC: Increment OODA Step
                        draw_ai_thought(interaction_pos, unit_vec, safe_dist)
                        save_diagnostic_snapshot(rgb, heatmap, raw_vector, prompt, TASK_LOG_DIR, snapshot_counter); snapshot_counter += 1

                        push_vec_3d = np.array([unit_vec[0], unit_vec[1], 0.0])
                        start_push = interaction_pos - (push_vec_3d * PRE_PUSH_OFFSET); start_push[2] = PUSH_Z
                        end_push = interaction_pos + (push_vec_3d * safe_dist); end_push[2] = PUSH_Z
                        
                        curr_ee = p.getLinkState(robotId, FRANKA_PANDA_ENDEFFECTOR_INDEX)[0]
                        
                        path_approach = None
                        approach_angles =[0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2] 
                        
                        for retry in range(MAX_PLANNING_RETRIES):
                            if retry == 0:
                                hover_pos = start_push.copy(); hover_pos[2] = HOVER_Z
                                path_approach = planner.plan(curr_ee, hover_pos)
                            else:
                                angle_offset = approach_angles[retry % len(approach_angles)]
                                offset_dist = 0.15 
                                approach_offset = np.array([np.cos(angle_offset) * offset_dist, np.sin(angle_offset) * offset_dist, 0])
                                hover_pos = start_push.copy() + approach_offset
                                hover_pos[2] = HOVER_Z
                                path_approach = planner.plan(curr_ee, hover_pos)
                            if path_approach:
                                print(f"  [Planner] Found path (attempt {retry + 1})")
                                break
                        
                        if path_approach:
                            orn = p.getQuaternionFromEuler([np.pi, 0, 0])
                            for pt in interpolate_path(path_approach): waypoints.append(("move", pt, orn, False))
                            waypoints.append(("move", start_push, orn, False))
                            
                            for pt in interpolate_path([start_push, end_push], 0.01): waypoints.append(("move", pt, orn, True))
                            
                            retract_horizontal = end_push.copy()
                            retract_horizontal -= push_vec_3d * (PRE_PUSH_OFFSET * 1.5)  
                            retract_horizontal[2] = PUSH_Z  
                            waypoints.append(("move", retract_horizontal, orn, False))
                            
                            retract_vertical = retract_horizontal.copy()
                            retract_vertical[2] = HOVER_Z
                            waypoints.append(("move", retract_vertical, orn, False))
                            state = RobotState.EXECUTING
                        else:
                            print("  [Planner] All approach strategies failed. Trying nudge fallback...")
                            for angle_adjust in[0, 0.3, -0.3, 0.6, -0.6]:
                                adjusted_angle = np.arctan2(unit_vec[1], unit_vec[0]) + angle_adjust
                                fallback_vec = np.array([np.cos(adjusted_angle), np.sin(adjusted_angle), 0.0])
                                fallback_start = interaction_pos - (fallback_vec * PRE_PUSH_OFFSET * 0.5)
                                fallback_start[2] = PUSH_Z
                                fallback_hover = fallback_start.copy(); fallback_hover[2] = HOVER_Z
                                
                                fallback_path = planner.plan(curr_ee, fallback_hover)
                                if fallback_path:
                                    interventions_count += 1  # <--- METRIC: Triggered Nudge Fallback
                                    print(f"  [Planner] Fallback succeeded with angle adjustment {angle_adjust:.2f}")
                                    fallback_end = interaction_pos + (fallback_vec * safe_dist * 0.5)
                                    fallback_end[2] = PUSH_Z
                                    orn = p.getQuaternionFromEuler([np.pi, 0, 0])
                                    for pt in interpolate_path(fallback_path): waypoints.append(("move", pt, orn, False))
                                    waypoints.append(("move", fallback_start, orn, False))
                                    for pt in interpolate_path([fallback_start, fallback_end], 0.01): waypoints.append(("move", pt, orn, True))
                                    
                                    retract_horizontal = fallback_end.copy()
                                    retract_horizontal -= fallback_vec * (PRE_PUSH_OFFSET * 1.5) 
                                    retract_horizontal[2] = PUSH_Z  
                                    waypoints.append(("move", retract_horizontal, orn, False))
                                    retract_vertical = retract_horizontal.copy()
                                    retract_vertical[2] = HOVER_Z
                                    waypoints.append(("move", retract_vertical, orn, False))
                                    state = RobotState.EXECUTING
                                    break
                            else:
                                print("  [Planner] Complete failure. Skipping this step."); time.sleep(0.5)
                    
                else:
                    print(f"  [AI] Low confidence ({heatmap.max():.2f}). Object not found."); time.sleep(0.5)

            elif state == RobotState.EXECUTING:
                if controller.is_at_target() and waypoints:
                    act = waypoints.pop(0)
                    if act[0] == "move": 
                        tgt = act[1]; tgt[2] = max(tgt[2], MIN_SAFE_Z)
                        is_push = act[3] if len(act) > 3 else False
                        controller.set_target(tgt, act[2], is_push=is_push)
                    elif act[0] == "gripper": set_gripper(robotId, act[1])
            
            controller.step(); p.stepSimulation(); time.sleep(SIM_SLEEP_TIME)
            
            curr_pos, _ = p.getBasePositionAndOrientation(objectId)
            dist_now = np.linalg.norm(np.array(curr_pos)[:2] - np.array(GOAL_POS_VISUAL)[:2])

            if dist_now < 0.08 and state != RobotState.IDLE:
                print(f"\n==================================================")
                print(f"[Metrics] Task SUCCESS! Cube reached goal.")
                print(f"  -> Total OODA Steps (Efficiency): {episode_steps}")
                print(f"  -> Fallback Interventions: {interventions_count}")
                print(f"  -> Normalized Progress Distance: 100.0%")
                print(f"==================================================\n")
                
                controller.interrupt()
                waypoints = []
                state = RobotState.IDLE  # Pause and wait for next instruction
                start_dist = None
                
    finally:
        if p.isConnected: p.disconnect()


                
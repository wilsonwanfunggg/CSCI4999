import time
from fairino import Robot # Import the Fairino SDK

def main():
    print("--- Starting Hardware Kinematic Validation ---")
    
    # 1. Connect to the FR5 Robot via Ethernet
    # The default IP for FR5 is almost always 192.168.58.2
    print("Connecting to FR5...")
    robot = Robot.RPC('192.168.58.2')
    
    # --- PHYSICAL COORDINATES (IN MILLIMETERS) ---
    # Look at the Teach Pendant screen and type the exact X, Y numbers here:
    CUBE_X = 400.0  # mm (PyBullet was 0.40m)
    CUBE_Y = -150.0 # mm (PyBullet was -0.15m)
    
    # Teach Pendant Rotations (Keep these constant so the gripper points straight down)
    # Check your pendant. Usually pointing down is RX=180, RY=0, RZ=0
    RX = 180.0 
    RY = 0.0
    RZ = 0.0
    
    # Kinematic Heights & Distances (in mm!)
    HOVER_Z = 300.0        # 30cm hover
    PUSH_Z = 20.0          # 2cm height to hit the cube center
    PRE_PUSH_OFFSET = 80.0 # 8cm gap before pushing
    PUSH_DIST = 50.0       # 5cm push
    
    # 2. Define the Waypoints: [X, Y, Z, RX, RY, RZ]
    hover_pos =[CUBE_X - PRE_PUSH_OFFSET, CUBE_Y, HOVER_Z, RX, RY, RZ]
    start_push =[CUBE_X - PRE_PUSH_OFFSET, CUBE_Y, PUSH_Z, RX, RY, RZ]
    end_push =[CUBE_X + PUSH_DIST, CUBE_Y, PUSH_Z, RX, RY, RZ]
    retract_pos =[CUBE_X - (PRE_PUSH_OFFSET * 1.5), CUBE_Y, PUSH_Z, RX, RY, RZ]
    
    # Movement Settings
    tool = 0       # Default tool coordinate system
    user = 0       # Default user coordinate system
    vel = 50.0     # Speed: 50 mm/s (Keep it slow and safe!)
    acc = 50.0     # Acceleration
    
    # 3. Execute the Trajectory using MoveCart (Cartesian Linear Move)
    print("Moving to hover position...")
    robot.MoveCart(hover_pos, tool, user, vel, acc, 100.0, -1.0, -1)
    time.sleep(3) # Wait for move to finish
    
    print("Lowering to table...")
    robot.MoveCart(start_push, tool, user, vel, acc, 100.0, -1.0, -1)
    time.sleep(2)
    
    print("Executing 5cm push...")
    robot.MoveCart(end_push, tool, user, vel, acc, 100.0, -1.0, -1)
    time.sleep(2)
    
    print("Executing horizontal anti-flip retraction...")
    robot.MoveCart(retract_pos, tool, user, vel, acc, 100.0, -1.0, -1)
    time.sleep(2)
    
    print("Returning to hover...")
    robot.MoveCart([retract_pos[0], retract_pos[1], HOVER_Z, RX, RY, RZ], tool, user, vel, acc, 100.0, -1.0, -1)
    
    print("--- Physical Test Complete! ---")

if __name__ == "__main__":
    main()
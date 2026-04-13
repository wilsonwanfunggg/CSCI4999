import torch
import cv2
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from model import LanguageConditionedUNet

# --- CONFIGURATION ---
MODEL_PATH = "best_multitask_model.pth"
IMAGE_PATH = "real_world_test.jpg"  # The photo of your cardboard setup
PROMPT = "solve the maze"           # The instruction you want to test
OUTPUT_PATH = "real_world_result.png"

IMG_WIDTH, IMG_HEIGHT = 224, 224

# Set device
if torch.cuda.is_available(): DEVICE = "cuda"
elif torch.backends.mps.is_available(): DEVICE = "mps"
else: DEVICE = "cpu"

def main():
    print(f"--- Running Offline Real-World Perception Test on {DEVICE} ---")

    # 1. Load the trained model
    print("Loading Multi-Task U-Net...")
    model = LanguageConditionedUNet(n_channels=3, n_classes=1, embedding_dim=512).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load CLIP for text embedding
    print("Loading CLIP Text Encoder...")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip.eval()

    # 3. Process the Real-World Image
    print(f"Processing image: {IMAGE_PATH}")
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print(f"[Error] Could not find {IMAGE_PATH}")
        return
    
    # Resize to match training dimensions (224x224)
    img_resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

    # 4. Run AI Inference
    print(f"Running AI prediction for prompt: '{PROMPT}'")
    with torch.no_grad():
        txt = tokenizer([PROMPT], padding=True, return_tensors="pt").to(DEVICE)
        txt_feat = clip.get_text_features(**txt)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        
        logits, pred_vec = model(img_t, txt_feat)
        heatmap = torch.sigmoid(logits).squeeze().cpu().numpy()
        raw_vector_field = pred_vec.squeeze().cpu().numpy()

    # 5. Extract Data and Visualize (Same logic as dynamic_task.py)
    if heatmap.max() < 0.1:
        print("AI Confidence is too low. Cube not detected.")
    else:
        # Find the pixel with the highest confidence (the yellow cube)
        cy, cx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        raw_vector = raw_vector_field[:, cy, cx]
        
        print(f"AI localized target at pixel (x: {cx}, y: {cy})")
        print(f"Predicted raw vector: {raw_vector}")

        # Visualization: Blend heatmap with the real image
        vis = cv2.addWeighted(img_resized, 0.6, cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
        
        # Draw the predicted vector arrow
        rx = -raw_vector[1]
        ry = raw_vector[0]
        norm = np.linalg.norm([rx, ry])
        if norm > 1e-5:
            rx, ry = rx / norm, ry / norm
            
        # Draw arrow pointing in the AI's predicted direction
        end_x = int(cx + rx * 40) 
        end_y = int(cy - ry * 40) 
        cv2.arrowedLine(vis, (int(cx), int(cy)), (int(end_x), int(end_y)), (255, 255, 255), 2, tipLength=0.4)
        
        # Save the result
        cv2.imwrite(OUTPUT_PATH, vis)
        print(f"\nSUCCESS! Visualization saved to {OUTPUT_PATH}")
        print("Use this image as 'Figure X' in your Chapter 7 Report.")

if __name__ == '__main__':
    main()
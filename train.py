import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm 

# Import language-conditinoned U-net model implemented
from model import LanguageConditionedUNet

# Hyperparameters
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR = "behavioral_cloning_data"
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2 
MODEL_SAVE_PATH = "best_multitask_model.pth"


# PyTorch Dataset 
class BehavioralCloningDataset(Dataset):
    def __init__(self, image_paths, mask_paths, vector_paths, prompt_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.vector_paths = vector_paths
        self.prompt_paths = prompt_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image file
        image = cv2.imread(self.image_paths[idx])
        # BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # convert from Numpy array to Pytorch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        # convert 2D tensor -> 3D
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        vector = np.load(self.vector_paths[idx])
        vector = torch.from_numpy(vector).float()

        with open(self.prompt_paths[idx], "r") as f:
            prompt = f.read().strip()
        
        return image, mask, vector, prompt

# training & validation
if __name__ == '__main__':
    print(f"--- Using Device: {DEVICE} ---")

    # embed text prompt using CLIP
    print("Loading CLIP model...")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_model.eval() 

    # Find all data paths
    all_image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.png")))
    all_mask_paths = sorted(glob.glob(os.path.join(DATA_DIR, "masks", "*.png")))
    all_vector_paths = sorted(glob.glob(os.path.join(DATA_DIR, "vectors", "*.npy")))
    all_prompt_paths = sorted(glob.glob(os.path.join(DATA_DIR, "prompts", "*.txt")))

    assert len(all_image_paths) == len(all_vector_paths), "Mismatch between images and vectors."

    # Create full dataset
    full_dataset = BehavioralCloningDataset(all_image_paths, all_mask_paths, all_vector_paths, all_prompt_paths)

    # Split into training, validation sets
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # initialize model
    model = LanguageConditionedUNet(n_channels=3, n_classes=1, embedding_dim=512).to(DEVICE)
    # use AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # loss functions
    pos_weight = torch.tensor([20.0]).to(DEVICE)
    criterion_mask = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    criterion_vector = nn.MSELoss()

    best_val_loss = float('inf')
    print("--- Starting Training ---")

    # train & validation
    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_train_loss = 0.0
        
        for images, masks, vectors, prompts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, masks, vectors = images.to(DEVICE), masks.to(DEVICE), vectors.to(DEVICE)
            
            # Get text features from CLIP
            with torch.no_grad():
                inputs = clip_tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
                text_features = clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            pred_masks, pred_vec_fields = model(images, text_features)

            loss_mask = criterion_mask(pred_masks, masks)
            
            # --- FIX: Pool the dense vector field using the ground truth cube mask ---
            mask_weight = (masks > 0.5).float()
            sum_mask = mask_weight.sum(dim=(2,3)).view(-1, 1)
            # Averages the predicted vector ONLY where the cube is actually located
            pred_vectors = (pred_vec_fields * mask_weight).sum(dim=(2,3)) / (sum_mask + 1e-8)
            
            loss_vector = criterion_vector(pred_vectors, vectors) * 10.0
            total_loss = loss_mask + loss_vector


            # Forward pass
            optimizer.zero_grad()
            
            # Backward & optimize
            total_loss.backward()
            optimizer.step()
            
            running_train_loss += total_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # Validation
        model.eval() 
        running_val_loss = 0.0
        with torch.no_grad(): 
            for images, masks, vectors, prompts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, masks, vectors = images.to(DEVICE), masks.to(DEVICE), vectors.to(DEVICE)

                # Get text features
                inputs = clip_tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
                text_features = clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                pred_masks, pred_vec_fields = model(images, text_features)

                loss_mask = criterion_mask(pred_masks, masks)
            
            # --- FIX: Pool the dense vector field using the ground truth cube mask ---
                mask_weight = (masks > 0.5).float()
                sum_mask = mask_weight.sum(dim=(2,3)).view(-1, 1)
            # Averages the predicted vector ONLY where the cube is actually located
                pred_vectors = (pred_vec_fields * mask_weight).sum(dim=(2,3)) / (sum_mask + 1e-8)
            
                loss_vector = criterion_vector(pred_vectors, vectors) * 10.0
                total_loss = loss_mask + loss_vector
                
                running_val_loss += total_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # optimal model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

    print("\n--- Training Finished ---")
    print(f"Best model saved at {MODEL_SAVE_PATH} with validation loss: {best_val_loss:.4f}")
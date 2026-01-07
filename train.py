import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

# Import our custom modules
from dataset import OCRDataset, CharacterMap, collate_fn
from model import CRNN

# --- Configuration ---
# Data paths
TRAIN_GT_FILE = 'train.txt'
VALIDATION_GT_FILE = 'validation.txt'
CHAR_MAP_FILE = 'char_map.json'
MODEL_SAVE_DIR = 'models'
BEST_MODEL_NAME = 'best_model.pth'

# Model parameters (MUST MATCH model.py and dataset.py)
IMG_HEIGHT = 64
MAX_IMG_WIDTH = 800 # Set a fixed max width for padding
INPUT_CHANNELS = 1 # Grayscale
RNN_HIDDEN_SIZE = 512

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
# --- MODIFIED: Lowered LR for stability ---
LEARNING_RATE = 0.0001 # 1e-4
# ---------------------

def decode_ctc_output(output, char_map):
    """
    Decodes the raw output from the CTC model into human-readable text.
    Uses a simple best-path decoding.
    """
    # output shape: (seq_len, batch_size, num_classes)
    pred_indices = torch.argmax(output, dim=2)
    # pred_indices shape: (seq_len, batch_size)
    
    # Transpose to (batch_size, seq_len)
    pred_indices = pred_indices.t().cpu().numpy()
    
    decoded_texts = []
    
    for indices in pred_indices:
        decoded_text = []
        last_char = None
        for idx in indices:
            if idx == 0: # 0 is the CTC <BLANK> token
                last_char = None
                continue
            
            char = char_map.int_to_char.get(idx, '?')
            
            if char != last_char:
                decoded_text.append(char)
            last_char = char
            
        decoded_texts.append("".join(decoded_text))
    return decoded_texts

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Prepare Data ---
    print("Building character map...")
    if not os.path.exists(CHAR_MAP_FILE):
        char_map_builder = CharacterMap(TRAIN_GT_FILE, CHAR_MAP_FILE)
    else:
        char_map_builder = CharacterMap() # Create empty map
        char_map_builder.load_map(CHAR_MAP_FILE) # Load from file
        print(f"Character map loaded from {CHAR_MAP_FILE} (vocab size: {char_map_builder.vocab_size})")
        
    vocab_size = char_map_builder.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = OCRDataset(TRAIN_GT_FILE, char_map_builder, IMG_HEIGHT, MAX_IMG_WIDTH)
    val_dataset = OCRDataset(VALIDATION_GT_FILE, char_map_builder, IMG_HEIGHT, MAX_IMG_WIDTH)

    # Use num_workers=2 for Colab
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print("Data loading complete.")

    # --- 2. Initialize Model, Loss, Optimizer ---
    print("Initializing model...")
    model = CRNN(IMG_HEIGHT, INPUT_CHANNELS, vocab_size, RNN_HIDDEN_SIZE).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler to lower LR if training gets stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        patience=5, 
        factor=0.1
    )

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)
    
    best_val_loss = float('inf')

    # --- 3. Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        for i, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            input_lengths = torch.full(
                size=(outputs.size(1),), 
                fill_value=outputs.size(0), 
                dtype=torch.long
            ).to(device)

            loss = criterion(outputs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            
            # Add Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()

            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                outputs = model(images)
                input_lengths = torch.full(
                    size=(outputs.size(1),), 
                    fill_value=outputs.size(0), 
                    dtype=torch.long
                ).to(device)
                
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        print(f"  Average Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Show Example Decoding
        try:
            images_ex, targets_ex, target_lengths_ex = next(iter(val_loader))
            images_ex = images_ex.to(device)
            
            outputs_ex = model(images_ex)
            decoded = decode_ctc_output(outputs_ex, char_map_builder)
            
            print("\n--- Validation Example ---")
            print(f"  Prediction: {decoded[0]}")
            
            # --- FIX for "cannot be converted to Scalar" error ---
            # Get the length of the first target as a Python integer
            first_len = target_lengths_ex[0].item()
            
            # Get the first target's indices
            target_indices = targets_ex[0][:first_len]
            
            # Loop through the 0D tensors and convert to text
            true_text = "".join([char_map_builder.int_to_char.get(t.item(), '?') for t in target_indices])
            # --- END FIX ---

            print(f"  Ground Truth: {true_text}")
            print("--------------------------\n")
        except Exception as e:
            print(f"Error in decoding example: {e}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f}) ***\n")

    print("Training complete.")

if __name__ == "__main__":
    train()


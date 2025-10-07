# improved_model_management.py
import os
import glob
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from .model import XiangqiNet

# -----------------------------
#   Dataset Definition
# -----------------------------

class XiangqiDataset(Dataset):
    """
    Loads all game_*.pkl files from a directory.
    Supports files that are:
        - list of samples
        - tuple: (list_of_samples, metadata)
        - dict: {"game_data": list_of_samples, "metadata": ...}
    """
    def __init__(self, data_dir):
        super().__init__()
        self.samples = []

        pkl_files = sorted(glob.glob(os.path.join(data_dir, "game_*.pkl")))
        if not pkl_files:
            raise FileNotFoundError(f"No game_*.pkl files found in {data_dir}")

        print(f"Found {len(pkl_files)} game files in {data_dir}")

        for file in pkl_files:
            with open(file, "rb") as f:
                game_obj = pickle.load(f)

            # Case 1: list
            if isinstance(game_obj, list):
                game_data = game_obj

            # Case 2: tuple (game_data, metadata)
            elif isinstance(game_obj, tuple) and len(game_obj) >= 1:
                game_data = game_obj[0]
                if not isinstance(game_data, list):
                    raise ValueError(f"First element of tuple in {file} must be a list")

            # Case 3: dict {"game_data": [...]}
            elif isinstance(game_obj, dict) and "game_data" in game_obj:
                game_data = game_obj["game_data"]

            else:
                raise ValueError(f"Unexpected format in {file}: {type(game_obj)}")

            # Extract samples
            for sample in game_data:
                if isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    state, pi, z = sample[:3]
                    self.samples.append((state, pi, z))
                else:
                    raise ValueError(f"Invalid sample in {file}: {sample}")

        print(f"Loaded {len(self.samples)} total samples from {len(pkl_files)} games.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, pi, z = self.samples[idx]
        state = torch.tensor(state, dtype=torch.float32)
        pi = torch.tensor(pi, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)
        return state, pi, z

# -----------------------------
#   Model Manager (same as before)
# -----------------------------
class ModelManager:
    def __init__(self, base_dir="models"):
        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        self.best_models_dir = os.path.join(base_dir, "best_models")
        self.exports_dir = os.path.join(base_dir, "exports")
        self.plots_dir = os.path.join(base_dir, "plots")
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, history, 
                       game_results, training_config, is_best=False):
        """Save a complete training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint data
        checkpoint = {
            # Model and training state
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            
            # Training progress
            'epoch': epoch,
            'metrics': metrics,
            'history': dict(history),
            'game_results': game_results,
            'training_config': training_config,
            
            # Model architecture info
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'config', {}),
            
            # Metadata
            'timestamp': timestamp,
            'torch_version': torch.__version__,
            'device': str(next(model.parameters()).device),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as latest
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save as best if applicable
        if is_best:
            best_path = os.path.join(self.best_models_dir, f"best_model_{timestamp}.pt")
            torch.save(checkpoint, best_path)
            
            # Also save as current best
            current_best_path = os.path.join(self.best_models_dir, "best_model.pt")
            torch.save(checkpoint, current_best_path)
            print(f"Best model saved: {best_path}")
            
            # Save training summary
            self.save_training_summary(checkpoint, best_path.replace('.pt', '_summary.json'))
        
        # Clean up old checkpoints (keep last 5)
        self.cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def save_training_summary(self, checkpoint, summary_path):
        """Save a human-readable training summary"""
        summary = {
            'training_completed': datetime.now().isoformat(),
            'total_epochs': checkpoint['epoch'],
            'final_metrics': checkpoint['metrics'],
            'best_metrics': {
                'best_val_loss': min(checkpoint['history']['val_loss']),
                'best_value_sign_accuracy': max(checkpoint['history']['val_value_sign_accuracy']),
                'best_mcts_agreement': max(checkpoint['history']['val_mcts_agreement'])
            },
            'model_info': {
                'total_parameters': checkpoint['total_parameters'],
                'trainable_parameters': checkpoint['trainable_parameters'],
                'model_class': checkpoint['model_class']
            },
            'training_config': checkpoint['training_config'],
            'game_results': checkpoint['game_results']
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f" Training summary saved: {summary_path}")
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """Load a checkpoint and restore training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f" Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f" Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f" Final validation loss: {checkpoint['metrics']['loss']:.4f}")
        print(f" MCTS agreement: {checkpoint['metrics']['mcts_agreement']:.3f}")
        print(f" Value sign accuracy: {checkpoint['metrics']['value_sign_accuracy']:.3f}")
        
        return checkpoint
    
    def export_model_for_inference(self, model, export_name=None):
        """Export model for inference (production use)"""
        if export_name is None:
            export_name = f"xiangqi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model.eval()
        
        # Export state dict only (smaller file)
        state_dict_path = os.path.join(self.exports_dir, f"{export_name}_state_dict.pt")
        torch.save(model.state_dict(), state_dict_path)
        
        # Export complete model (larger but self-contained)
        full_model_path = os.path.join(self.exports_dir, f"{export_name}_full.pt")
        torch.save(model, full_model_path)
        
        # Export to ONNX (for cross-platform inference)
        try:
            onnx_path = os.path.join(self.exports_dir, f"{export_name}.onnx")
            dummy_input = torch.randn(1, 16, 10, 9)  # Adjust to your input shape
            torch.onnx.export(
                model, dummy_input, onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['state'],
                output_names=['policy', 'value']
            )
            print(f"ONNX model exported: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
        
        # Export metadata
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'model_class': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'input_shape': [16, 10, 9],  # Adjust to your input shape
            'output_shapes': {
                'policy': [8100],
                'value': [1]
            }
        }
        
        metadata_path = os.path.join(self.exports_dir, f"{export_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Model exported for inference:")
        print(f" State dict: {state_dict_path}")
        print(f" Full model: {full_model_path}")
        print(f" Metadata: {metadata_path}")
        
        return {
            'state_dict_path': state_dict_path,
            'full_model_path': full_model_path,
            'onnx_path': onnx_path if 'onnx_path' in locals() else None,
            'metadata_path': metadata_path
        }
    
    def cleanup_old_checkpoints(self, keep_last=5):
        """Keep only the most recent checkpoints"""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        
        if len(checkpoint_files) > keep_last:
            checkpoint_files.sort()
            files_to_delete = checkpoint_files[:-keep_last]
            
            for file in files_to_delete:
                file_path = os.path.join(self.checkpoint_dir, file)
                os.remove(file_path)
                print(f" Removed old checkpoint: {file}")
    
    def resume_training(self, model, optimizer, scheduler=None):
        """Resume training from the latest checkpoint"""
        latest_checkpoint = os.path.join(self.checkpoint_dir, "latest.pt")
        
        if os.path.exists(latest_checkpoint):
            checkpoint = self.load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
            return checkpoint['epoch'], checkpoint['history'], checkpoint['game_results']
        else:
            print("No checkpoint found, starting fresh training")
            return 0, defaultdict(list), {'red': 0, 'black': 0, 'draw': 0, 'adjudicated': 0}
    
    def list_available_models(self):
        """List all available models and checkpoints"""
        print("\n Available Models:")
        
        # Best models
        if os.path.exists(self.best_models_dir):
            best_models = [f for f in os.listdir(self.best_models_dir) if f.endswith('.pt')]
            if best_models:
                print(f"\n Best Models ({len(best_models)}):")
                for model in sorted(best_models):
                    model_path = os.path.join(self.best_models_dir, model)
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f" {model} ({size_mb:.1f} MB)")
        
        # Recent checkpoints
        if os.path.exists(self.checkpoint_dir):
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('checkpoint_') and f.endswith('.pt')]
            if checkpoints:
                print(f"\n Recent Checkpoints ({len(checkpoints)}):")
                for checkpoint in sorted(checkpoints)[-5:]:  # Show last 5
                    checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
                    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                    print(f" {checkpoint} ({size_mb:.1f} MB)")
        
        # Exported models
        if os.path.exists(self.exports_dir):
            exports = [f for f in os.listdir(self.exports_dir) if f.endswith('.pt')]
            if exports:
                print(f"\n Exported Models ({len(exports)}):")
                for export in sorted(exports):
                    export_path = os.path.join(self.exports_dir, export)
                    size_mb = os.path.getsize(export_path) / (1024 * 1024)
                    print(f" {export} ({size_mb:.1f} MB)")

    def plot_training_curves(self, history, save_path=None):
        plt.figure(figsize=(14, 8))

        # Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label="Train Loss")
        plt.plot(history['val_loss'], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training & Validation Loss")

        # Accuracy / agreement
        plt.subplot(2, 2, 2)
        plt.plot(history['val_value_sign_accuracy'], label="Val Value Sign Acc")
        plt.plot(history['val_mcts_agreement'], label="Val MCTS Agreement")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Validation Metrics")

        # Policy vs value loss breakdown
        if 'train_policy_loss' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['train_policy_loss'], label="Train Policy Loss")
            plt.plot(history['val_policy_loss'], label="Val Policy Loss")
            plt.plot(history['train_value_loss'], label="Train Value Loss")
            plt.plot(history['val_value_loss'], label="Val Value Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Policy vs Value Loss")

        # Learning rate
        plt.subplot(2, 2, 4)
        plt.plot(history['learning_rate'], label="LR")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Training curves saved: {save_path}")
        else:
            plt.show()

# -----------------------------
#   Integrated Training Example
# -----------------------------
def integrated_training():
    """Example integration of ModelManager with dataset + training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Model manager
    model_manager = ModelManager()

    # Training config
    training_config = {
        'batch_size': 16,
        'learning_rate': 2e-4,
        'epochs': 30,
        'optimizer': 'Adam',
        'scheduler': 'StepLR'
    }

    # -----------------------------
    # Load dataset
    # -----------------------------
    full_dataset = XiangqiDataset("self_play_data/")  # path to folder with game_*.pkl

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)

    # -----------------------------
    # Initialize model + optimizer
    # -----------------------------
    model = XiangqiNet().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="runs/xiangqi")

    # Try resume
    start_epoch, history, game_results = model_manager.resume_training(model, optimizer, scheduler)

    patience = 5
    no_improve_epochs = 0
    best_val_loss = float('inf')
    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(start_epoch, training_config['epochs']):
        model.train()
        total_loss = 0.0

        for states, policies, values in train_loader:
            states, policies, values = states.to(device), policies.to(device), values.to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = model(states)

            # Make sure pred_value and values have same shape [batch_size]
            pred_value = pred_value.squeeze(-1)
            values = values.squeeze(-1)

            eps = 0.1  # smoothing factor
            smoothed_policies = (1 - eps) * policies + eps / policies.size(1)
            log_probs = torch.log_softmax(pred_policy, dim=1)
            loss_policy = -(smoothed_policies * log_probs).sum(dim=1).mean()

            loss_value = value_loss_fn(pred_value, values)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # -----------------------------
        # Validation
        # -----------------------------
    
        model.eval()
        val_loss, correct_signs, total_samples = 0.0, 0, 0
        val_policy_loss_total, val_value_loss_total = 0.0, 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for states, policies, values in val_loader:
                states, policies, values = states.to(device), policies.to(device), values.to(device)
                pred_policy, pred_value = model(states)

                pred_value = pred_value.squeeze(-1)
                values = values.squeeze(-1)

                loss_policy = policy_loss_fn(pred_policy, policies.argmax(dim=1))
                loss_value = value_loss_fn(pred_value, values)
                batch_loss = loss_policy + loss_value

                val_loss += batch_loss.item()
                val_policy_loss_total += loss_policy.item()
                val_value_loss_total += loss_value.item()

                # Sign accuracy
                correct_signs += ((pred_value.sign() == values.sign()).sum().item())
                total_samples += values.size(0)

                # Collect for MAE & correlation
                all_preds.extend(pred_value.cpu().numpy())
                all_targets.extend(values.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_policy_loss = val_policy_loss_total / len(val_loader)
        avg_val_value_loss = val_value_loss_total / len(val_loader)

        val_sign_acc = correct_signs / total_samples if total_samples > 0 else 0.0
        val_mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))
        val_corr = float(np.corrcoef(all_preds, all_targets)[0, 1]) if len(all_preds) > 1 else 0.0

        # -----------------------------
        # Early stopping logic
        # -----------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏹ Early stopping at epoch {epoch} (no val improvement).")
                break

        # Real MCTS agreement
        correct_moves = 0
        total_moves = 0
        with torch.no_grad():
            for states, pi, _ in val_loader:  # pi = MCTS visit probabilities from self-play
                states, pi = states.to(device), pi.to(device)
                pred_policy, _ = model(states)
                pred_moves = pred_policy.argmax(dim=1)
                true_moves = pi.argmax(dim=1)
                correct_moves += (pred_moves == true_moves).sum().item()
                total_moves += states.size(0)

        val_mcts_agreement = correct_moves / total_moves if total_moves > 0 else 0.0

        # Metrics
        val_metrics = {
            'loss': avg_val_loss,
            'mcts_agreement': val_mcts_agreement,
            'value_sign_accuracy': val_sign_acc
        }

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_value_sign_accuracy'].append(val_sign_acc)
        history['val_mcts_agreement'].append(val_mcts_agreement)
        history['learning_rate'].append(scheduler.get_last_lr()[0])

        # Now safe because metrics are computed above
        history['train_policy_loss'].append(loss_policy.item())
        history['train_value_loss'].append(loss_value.item())
        history['val_policy_loss'].append(avg_val_policy_loss)
        history['val_value_loss'].append(avg_val_value_loss)
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr)


        # -----------------------------
        # ADD HERE: Overfitting Detection
        # -----------------------------
        if 'overfitting_flags' not in history:
            history['overfitting_flags'] = []

        if len(history['train_loss']) > 5:
            recent_train = np.mean(history['train_loss'][-5:])
            recent_val = np.mean(history['val_loss'][-5:])
            if recent_val > recent_train * 1.2:  # 20% higher than train loss
                flag = f"Epoch {epoch}: Val loss {recent_val:.4f} vs Train loss {recent_train:.4f}"
                print(f"Possible overfitting detected: {flag}")
                history['overfitting_flags'].append(flag)

        # -----------------------------
        # ADD HERE: TensorBoard Logging
        # -----------------------------
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/ValSign", val_sign_acc, epoch)
        writer.add_scalar("Agreement/MCTS", val_mcts_agreement, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        if epoch % 5 == 0 or is_best:
            model_manager.save_checkpoint(
                model, optimizer, scheduler, epoch,
                metrics=val_metrics, history=history,
                game_results=game_results,
                training_config=training_config,
                is_best=is_best
            )

            model_manager.plot_training_curves(history, os.path.join(model_manager.plots_dir, f"curves_epoch_{epoch}.png"))

        scheduler.step(avg_val_loss)


    # -----------------------------
    # Export final model
    # -----------------------------
    model_manager.export_model_for_inference(model, "final_xiangqi_model")
    model_manager.list_available_models()
    writer.close()

if __name__ == "__main__":
    manager = ModelManager()
    manager.list_available_models()
    integrated_training()  # Uncomment to run training

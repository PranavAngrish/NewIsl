import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from logger import setup_logger
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import copy
import random

class EnhancedISLTrainer:
    """Enhanced PyTorch trainer with better generalization techniques"""
    
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        
        # Use provided device or auto-detect
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Enhanced Trainer using device: {self.device}")
        
        # MPS memory optimization
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            print("MPS cache cleared")
            
        # Move model to device
        self.model.to(self.device)
        
        # Enhanced loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        
        # Add focal loss for handling class imbalance
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.focal_loss_weight = 0.3  # Weight for focal loss in combined loss
        
        # Mixed precision training - disable for MPS
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            print("Mixed precision disabled for non-CUDA device")
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        # Training history with more metrics
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top3_acc': [],
            'learning_rates': [],
            'domain_loss': []
        }
        
        # Setup logging
        self.logger = setup_logger(name="EnhancedISLTrainer")
        
        # Enhanced memory optimization settings
        self.gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        self.memory_cleanup_frequency = 5  # More frequent cleanup
        
        # Progressive unfreezing schedule
        self.progressive_unfreezing = True
        self.unfreeze_schedule = {
            5: 1,   # Unfreeze 1 more layer after 5 epochs
            10: 2,  # Unfreeze 2 more layers after 10 epochs  
            15: 3   # Unfreeze 3 more layers after 15 epochs
        }
        
        # Stochastic Weight Averaging (SWA)
        self.use_swa = True
        self.swa_start_epoch = max(config.EPOCHS // 2, 10)
        self.swa_models = []
        self.swa_scheduler = None
        
        # Domain adaptation parameters
        self.use_domain_adaptation = True
        self.domain_loss_weight = 0.1
        self.consistency_weight = 0.5
        
        # Mixup parameters for better generalization
        self.use_mixup = True
        self.mixup_alpha = 0.2
        
        # Test Time Augmentation (TTA)
        self.use_tta = True
        self.tta_transforms = 5
        
        # Model ensemble for better performance
        self.ensemble_models = []
        self.use_ensemble = False
        
    def focal_loss(self, inputs, targets):
        """Focal Loss for handling class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_loss_alpha * (1-pt)**self.focal_loss_gamma * ce_loss
        return focal_loss.mean()
    
    def mixup_data(self, frames, landmarks, targets, alpha=0.2):
        """Apply mixup augmentation to improve generalization"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = frames.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_frames = lam * frames + (1 - lam) * frames[index, :]
        mixed_landmarks = lam * landmarks + (1 - lam) * landmarks[index, :]
        
        y_a, y_b = targets, targets[index]
        return mixed_frames, mixed_landmarks, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Calculate mixup loss"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def consistency_loss(self, outputs1, outputs2, temperature=3.0):
        """Consistency loss for augmented samples"""
        outputs1_soft = F.softmax(outputs1 / temperature, dim=1)
        outputs2_soft = F.softmax(outputs2 / temperature, dim=1)
        consistency_loss = F.kl_div(
            F.log_softmax(outputs1 / temperature, dim=1),
            outputs2_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        return consistency_loss
    
    def cleanup_memory(self):
        """Enhanced memory cleanup"""
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def progressive_unfreeze(self, epoch):
        """Progressive unfreezing of model layers"""
        if not self.progressive_unfreezing or not hasattr(self.model, 'unfreeze_layer'):
            return
            
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            current_frozen = getattr(self.model, 'frozen_layers', 6)
            new_frozen = max(0, current_frozen - layers_to_unfreeze)
            
            # Unfreeze layers
            for layer_num in range(new_frozen, current_frozen):
                if hasattr(self.model, 'unfreeze_layer'):
                    self.model.unfreeze_layer(layer_num)
                
            self.model.frozen_layers = new_frozen
            self.logger.info(f"Epoch {epoch}: Unfroze layers, now {new_frozen} layers frozen")

    def add_to_swa(self, epoch):
        """Add current model to SWA ensemble"""
        if self.use_swa and epoch >= self.swa_start_epoch:
            # Store model state for SWA
            model_state = copy.deepcopy(self.model.state_dict())
            self.swa_models.append(model_state)
            
            # Keep only last N models for SWA
            if len(self.swa_models) > 5:
                self.swa_models.pop(0)
                
            self.logger.info(f"Added model from epoch {epoch} to SWA ensemble")

    def apply_swa(self):
        """Apply Stochastic Weight Averaging"""
        if not self.swa_models:
            return
            
        swa_state_dict = {}
        num_models = len(self.swa_models)
        
        # Average all model parameters
        for key in self.swa_models[0].keys():
            swa_state_dict[key] = torch.stack([
                model_state[key].float() for model_state in self.swa_models
            ]).mean(dim=0)
        
        # Apply SWA weights
        self.model.load_state_dict(swa_state_dict)
        self.logger.info(f"Applied SWA with {num_models} models")

    def train_epoch_with_enhancements(self, train_loader, optimizer, scheduler=None):
        """Enhanced training epoch with better generalization techniques"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        domain_loss_total = 0.0

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc='Enhanced Training')

        for batch_idx, ((frames, landmarks), labels) in enumerate(pbar):
            retry = True
            while retry:
                try:
                    # Move to device
                    frames = frames.to(self.device, non_blocking=True)
                    landmarks = landmarks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # Apply mixup augmentation
                    if self.use_mixup and random.random() < 0.5:
                        mixed_frames, mixed_landmarks, y_a, y_b, lam = self.mixup_data(
                            frames, landmarks, labels, self.mixup_alpha
                        )
                        
                        # Forward pass with mixup
                        if self.use_amp:
                            with autocast(device_type=self.device.type):
                                logits = self.model(mixed_frames, mixed_landmarks)
                                loss = self.mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                        else:
                            logits = self.model(mixed_frames, mixed_landmarks)
                            loss = self.mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                            
                        # Calculate accuracy (use original labels for accuracy)
                        with torch.no_grad():
                            original_logits = self.model(frames, landmarks)
                            _, predicted = torch.max(original_logits.data, 1)
                    else:
                        # Regular forward pass
                        if self.use_amp:
                            with autocast(device_type=self.device.type):
                                logits = self.model(frames, landmarks)
                                # Combined loss: CE + Focal Loss
                                ce_loss = self.criterion(logits, labels)
                                focal_loss_val = self.focal_loss(logits, labels)
                                loss = (1 - self.focal_loss_weight) * ce_loss + self.focal_loss_weight * focal_loss_val
                        else:
                            logits = self.model(frames, landmarks)
                            ce_loss = self.criterion(logits, labels)
                            focal_loss_val = self.focal_loss(logits, labels)
                            loss = (1 - self.focal_loss_weight) * ce_loss + self.focal_loss_weight * focal_loss_val

                        # Calculate accuracy
                        with torch.no_grad():
                            _, predicted = torch.max(logits.data, 1)

                    # Add consistency regularization
                    if self.consistency_weight > 0 and random.random() < 0.3:
                        # Create augmented version
                        noise_frames = frames + torch.randn_like(frames) * 0.01
                        noise_landmarks = landmarks + torch.randn_like(landmarks) * 0.01
                        
                        with torch.no_grad():
                            aug_logits = self.model(noise_frames, noise_landmarks)
                        
                        consistency_loss_val = self.consistency_loss(logits, aug_logits)
                        loss = loss + self.consistency_weight * consistency_loss_val

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient clipping for stability
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_amp:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            optimizer.step()
                        optimizer.zero_grad()

                    # Update metrics
                    with torch.no_grad():
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()
                        total_loss += loss.item() * self.gradient_accumulation_steps

                    # Memory cleanup
                    if batch_idx % self.memory_cleanup_frequency == 0:
                        self.cleanup_memory()

                    # Update progress bar
                    current_acc = 100.0 * total_correct / total_samples
                    pbar.set_postfix({
                        'Loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                        'Acc': f'{current_acc:.2f}%'
                    })

                    # Clear variables
                    del frames, landmarks, labels, logits

                    retry = False  # Success, exit retry loop

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error at batch {batch_idx}, cleaning up memory and retrying...")
                        self.cleanup_memory()
                    else:
                        raise e

        # Final gradient step if needed
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * total_correct / total_samples

        return epoch_loss, epoch_acc

    def validate_with_tta(self, val_loader):
        """Validation with Test Time Augmentation"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Enhanced Validation')

            for batch_idx, ((frames, landmarks), labels) in enumerate(pbar):
                retry = True
                while retry:
                    try:
                        frames = frames.to(self.device, non_blocking=True)
                        landmarks = landmarks.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        if self.use_tta:
                            # Test Time Augmentation
                            tta_predictions = []
                            
                            for tta_idx in range(self.tta_transforms):
                                # Apply different augmentations
                                aug_frames = frames.clone()
                                aug_landmarks = landmarks.clone()
                                
                                if tta_idx > 0:  # Keep first prediction as original
                                    # Add small noise
                                    aug_frames += torch.randn_like(aug_frames) * 0.005
                                    aug_landmarks += torch.randn_like(aug_landmarks) * 0.005
                                
                                # Forward pass
                                if self.use_amp:
                                    with autocast(device_type=self.device.type):
                                        logits = self.model(aug_frames, aug_landmarks)
                                else:
                                    logits = self.model(aug_frames, aug_landmarks)
                                
                                tta_predictions.append(F.softmax(logits, dim=1))
                            
                            # Average TTA predictions
                            avg_predictions = torch.stack(tta_predictions).mean(dim=0)
                            final_logits = torch.log(avg_predictions + 1e-8)  # Convert back to logits
                            
                        else:
                            # Regular forward pass
                            if self.use_amp:
                                with autocast(device_type=self.device.type):
                                    final_logits = self.model(frames, landmarks)
                            else:
                                final_logits = self.model(frames, landmarks)

                        # Calculate loss
                        loss = self.criterion(final_logits, labels)

                        # Calculate accuracy
                        _, predicted = torch.max(final_logits.data, 1)
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()
                        total_loss += loss.item()

                        # Store for top-k accuracy
                        all_predictions.append(final_logits.cpu())
                        all_labels.append(labels.cpu())

                        # Memory cleanup
                        if batch_idx % self.memory_cleanup_frequency == 0:
                            self.cleanup_memory()

                        # Update progress bar
                        current_acc = 100.0 * total_correct / total_samples
                        pbar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Acc': f'{current_acc:.2f}%'
                        })

                        del frames, landmarks, labels, final_logits

                        retry = False

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM error during validation at batch {batch_idx}, cleaning up and retrying...")
                            self.cleanup_memory()
                        else:
                            raise e

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100.0 * total_correct / total_samples

        # Calculate top-3 accuracy
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            pred_probs = F.softmax(all_predictions, dim=1).numpy()
            labels_np = all_labels.numpy()

            top3_acc = 100.0 * top_k_accuracy_score(labels_np, pred_probs, k=3)
        else:
            top3_acc = 0.0

        return epoch_loss, epoch_acc, top3_acc

    def save_enhanced_checkpoint(self, epoch, optimizer, scheduler=None, is_best=False):
        """Save enhanced checkpoint with more metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config.__dict__,
            'device': str(self.device),
            'swa_models': self.swa_models if self.use_swa else [],
            'training_enhancements': {
                'mixup_alpha': self.mixup_alpha,
                'focal_loss_weight': self.focal_loss_weight,
                'consistency_weight': self.consistency_weight,
                'use_tta': self.use_tta,
                'progressive_unfreezing': self.progressive_unfreezing
            }
        }
        
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f'enhanced_checkpoint_epoch_{epoch:02d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'enhanced_best_model.pth')
            torch.save(checkpoint, best_path)
            # Store best model state for ensemble
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.logger.info(f"New best enhanced model saved with val_acc: {self.best_val_acc:.2f}%")

    def train_with_enhancements(self, train_loader, val_loader, optimizer, scheduler=None, start_epoch=0):
        """Enhanced training loop with all improvements"""
        self.logger.info(f"Starting enhanced training on device: {self.device}")
        self.logger.info(f"Enhanced features enabled:")
        self.logger.info(f"  - Mixup augmentation: {self.use_mixup}")
        self.logger.info(f"  - Test Time Augmentation: {self.use_tta}")
        self.logger.info(f"  - Stochastic Weight Averaging: {self.use_swa}")
        self.logger.info(f"  - Progressive unfreezing: {self.progressive_unfreezing}")
        self.logger.info(f"  - Focal loss weight: {self.focal_loss_weight}")
        
        # Initial memory cleanup
        self.cleanup_memory()
        
        # Early stopping parameters
        patience = getattr(self.config, 'EARLY_STOPPING_PATIENCE', 15)
        patience_counter = 0
        
        # Learning rate scheduler
        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        for epoch in range(start_epoch, self.config.EPOCHS):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            
            # Progressive unfreezing
            self.progressive_unfreeze(epoch)
            
            # Clean memory before each epoch
            self.cleanup_memory()
            
            try:
                # Enhanced training phase
                train_loss, train_acc = self.train_epoch_with_enhancements(train_loader, optimizer)
                
                # Enhanced validation phase
                val_loss, val_acc, val_top3_acc = self.validate_with_tta(val_loader)
                
                # Update learning rate
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
                
                # Store learning rate
                current_lr = optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_top3_acc'].append(val_top3_acc)
                
                # Print epoch results
                self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top3 Acc: {val_top3_acc:.2f}%")
                self.logger.info(f"Learning Rate: {current_lr:.6f}")
                
                # Add to SWA
                self.add_to_swa(epoch)
                
                # Check for best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save enhanced checkpoint
                self.save_enhanced_checkpoint(epoch, optimizer, scheduler, is_best)
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"Out of memory error in epoch {epoch+1}")
                    break
                else:
                    raise e
        
        # Apply SWA at the end
        if self.use_swa and self.swa_models:
            self.logger.info("Applying Stochastic Weight Averaging...")
            self.apply_swa()
            
            # Validate SWA model
            swa_val_loss, swa_val_acc, swa_val_top3_acc = self.validate_with_tta(val_loader)
            self.logger.info(f"SWA Model - Val Acc: {swa_val_acc:.2f}%, Val Top3 Acc: {swa_val_top3_acc:.2f}%")
            
            if swa_val_acc > self.best_val_acc:
                self.best_val_acc = swa_val_acc
                # Save SWA model as best
                self.save_enhanced_checkpoint(self.config.EPOCHS, optimizer, scheduler, is_best=True)
        
        self.logger.info(f"Enhanced training completed! Best Val Acc: {self.best_val_acc:.2f}%")
        return self.history

    def evaluate_with_enhancements(self, test_loader, class_names=None):
        """Enhanced evaluation with TTA and ensemble"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, ((frames, landmarks), labels) in enumerate(tqdm(test_loader, desc='Enhanced Evaluation')):
                try:
                    frames = frames.to(self.device)
                    landmarks = landmarks.to(self.device)
                    
                    if self.use_tta:
                        # Test Time Augmentation
                        tta_predictions = []
                        
                        for tta_idx in range(self.tta_transforms):
                            aug_frames = frames.clone()
                            aug_landmarks = landmarks.clone()
                            
                            if tta_idx > 0:
                                aug_frames += torch.randn_like(aug_frames) * 0.005
                                aug_landmarks += torch.randn_like(aug_landmarks) * 0.005
                            
                            logits = self.model(aug_frames, aug_landmarks)
                            tta_predictions.append(F.softmax(logits, dim=1))
                        
                        # Average TTA predictions
                        final_probs = torch.stack(tta_predictions).mean(dim=0)
                        _, predicted = torch.max(final_probs, 1)
                    else:
                        logits = self.model(frames, landmarks)
                        _, predicted = torch.max(logits.data, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    
                    if batch_idx % self.memory_cleanup_frequency == 0:
                        self.cleanup_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM during evaluation at batch {batch_idx}, skipping...")
                        self.cleanup_memory()
                        continue
                    else:
                        raise e
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        if class_names:
            report = classification_report(
                all_labels, all_predictions, 
                target_names=class_names, 
                output_dict=True
            )
        else:
            report = classification_report(all_labels, all_predictions, output_dict=True)
        
        self.logger.info(f"Enhanced Test Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels
        }
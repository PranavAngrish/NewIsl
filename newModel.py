import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import numpy as np

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class EnhancedTemporalFusionLayer(nn.Module):
    """Enhanced temporal fusion with better regularization"""
    
    def __init__(self, embed_dim, sequence_length, dropout_rate=0.1, num_heads=8, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        
        # Temporal positional embeddings with learnable scaling
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, sequence_length, embed_dim) * 0.02
        )
        self.pos_embed_scale = nn.Parameter(torch.ones(1))
        
        # Multi-head attention with better regularization
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Enhanced feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Drop path for regularization
        self.drop_path = DropPath(drop_path_rate)
        
        # Attention temperature for sharper/softer attention
        self.attention_temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x shape: (B, T, D)
        x = x + self.temporal_pos_embed * self.pos_embed_scale
        
        # Self-attention with residual connection and drop path
        attn_output, attn_weights = self.temporal_attention(x, x, x)
        x = x + self.drop_path(attn_output)
        x = self.layer_norm1(x)
        
        # Feed-forward with residual connection and drop path
        ffn_output = self.ffn(x)
        x = x + self.drop_path(ffn_output)
        x = self.layer_norm2(x)
        
        return x

class EnhancedLandmarkProcessor(nn.Module):
    """Enhanced landmark processor with better feature extraction"""
    
    def __init__(self, embed_dim, sequence_length, dropout_rate=0.1, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-scale landmark embedding
        self.landmark_embedding = nn.Sequential(
            nn.Linear(170, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Separate processing for different landmark types
        self.hand_processor = nn.Linear(126, embed_dim // 2)  # 2 hands * 21 points * 3 coords
        self.pose_processor = nn.Linear(44, embed_dim // 2)   # 11 pose points * 4 coords
        
        # Positional embeddings
        self.landmark_pos_embed = nn.Parameter(
            torch.randn(1, sequence_length, embed_dim) * 0.02
        )
        
        # Enhanced attention mechanism
        self.landmark_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feature enhancement network
        self.feature_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, landmarks):
        # landmarks shape: (B, T, 170)
        batch_size, seq_len = landmarks.shape[:2]
        
        # Split landmarks into hand and pose components
        hand_landmarks = landmarks[:, :, :126]  # Hand landmarks
        pose_landmarks = landmarks[:, :, 126:]  # Pose landmarks
        
        # Process separately
        hand_features = self.hand_processor(hand_landmarks)
        pose_features = self.pose_processor(pose_landmarks)
        
        # Combine features
        combined_features = torch.cat([hand_features, pose_features], dim=-1)
        
        # Main embedding
        x = self.landmark_embedding(landmarks)
        
        # Add combined features as residual
        x = x + combined_features
        
        x = self.dropout(x)
        x = x + self.landmark_pos_embed
        
        # Self-attention
        attn_output, _ = self.landmark_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feature enhancement
        enhanced = self.feature_enhancer(x)
        x = self.layer_norm2(x + enhanced)
        
        return x

class AdaptiveCrossModalFusion(nn.Module):
    """Adaptive cross-modal fusion with learned attention weights"""
    
    def __init__(self, embed_dim, dropout_rate=0.1, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross-attention layers with different heads for different modalities
        self.video_to_landmark_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.landmark_to_video_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        
        # Adaptive fusion weights
        self.video_weight = nn.Parameter(torch.ones(1))
        self.landmark_weight = nn.Parameter(torch.ones(1))
        
        # Enhanced fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Gate mechanism for adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, video_features, landmark_features):
        # Cross-attention: video -> landmark
        video_attended, _ = self.video_to_landmark_attention(
            query=video_features,
            key=landmark_features,
            value=landmark_features
        )
        video_attended = self.layer_norm1(video_features + video_attended)
        
        # Cross-attention: landmark -> video
        landmark_attended, _ = self.landmark_to_video_attention(
            query=landmark_features,
            key=video_features,
            value=video_features
        )
        landmark_attended = self.layer_norm2(landmark_features + landmark_attended)
        
        # Weighted combination
        weighted_video = video_attended * self.video_weight
        weighted_landmark = landmark_attended * self.landmark_weight
        
        # Concatenate for fusion
        concatenated = torch.cat([weighted_video, weighted_landmark], dim=-1)
        
        # Gated fusion
        gate = self.fusion_gate(concatenated)
        fused_basic = weighted_video + weighted_landmark
        
        # Enhanced fusion through network
        fused_enhanced = self.fusion_network(concatenated)
        
        # Apply gate
        fused = gate * fused_enhanced + (1 - gate) * fused_basic
        fused = self.layer_norm3(fused)
        
        return fused

class EnhancedISLViTModel(nn.Module):
    """Enhanced ISL detection model with improved regularization"""
    
    def __init__(self, config, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Load pre-trained ViT model with frozen early layers
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Progressive unfreezing: freeze first 6 layers initially
        self.frozen_layers = 6
        self.freeze_early_layers(self.frozen_layers)
        
        # Get ViT embedding dimension
        self.vit_embed_dim = self.vit_model.config.hidden_size
        
        # Enhanced projection with batch normalization
        self.vit_projection = nn.Sequential(
            nn.Linear(self.vit_embed_dim, config.EMBED_DIM),
            nn.BatchNorm1d(config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        # Enhanced temporal fusion
        self.temporal_fusion = EnhancedTemporalFusionLayer(
            embed_dim=config.EMBED_DIM,
            sequence_length=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE,
            num_heads=config.NUM_HEADS,
            drop_path_rate=0.1
        )
        
        # Enhanced landmark processor
        self.landmark_processor = EnhancedLandmarkProcessor(
            embed_dim=config.EMBED_DIM,
            sequence_length=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE,
            num_heads=config.NUM_HEADS // 2
        )
        
        # Adaptive cross-modal fusion
        self.cross_modal_fusion = AdaptiveCrossModalFusion(
            embed_dim=config.EMBED_DIM,
            dropout_rate=config.DROPOUT_RATE,
            num_heads=config.NUM_HEADS
        )
        
        # Enhanced classification head with multiple pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM),  # *2 for avg+max pooling
            nn.BatchNorm1d(config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 1.5),  # Higher dropout before final layer
            
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM // 2),
            nn.BatchNorm1d(config.EMBED_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(config.EMBED_DIM // 2, num_classes)
        )
        
        # Label smoothing loss
        self.label_smoothing = 0.1
        
    def freeze_early_layers(self, num_layers):
        """Freeze early transformer layers"""
        for name, param in self.vit_model.named_parameters():
            if 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < num_layers:
                    param.requires_grad = False
                    
    def unfreeze_layer(self, layer_num):
        """Unfreeze a specific layer for progressive training"""
        for name, param in self.vit_model.named_parameters():
            if f'encoder.layer.{layer_num}' in name:
                param.requires_grad = True
    
    def forward(self, frames, landmarks):
        """Enhanced forward pass with better feature processing"""
        batch_size, sequence_length = frames.shape[:2]
        
        # Reshape frames for ViT: (B*T, C, H, W)
        frames_reshaped = frames.view(batch_size * sequence_length, *frames.shape[2:])
        
        # Process through ViT with mixed precision
        with torch.cuda.amp.autocast():
            vit_outputs = self.vit_model(pixel_values=frames_reshaped)
            # Get CLS token features: (B*T, hidden_size)
            frame_features = vit_outputs.last_hidden_state[:, 0, :]
        
        # Project to our embedding dimension
        frame_features = self.vit_projection(frame_features)
        
        # Reshape back to sequence: (B, T, embed_dim)
        frame_features = frame_features.view(batch_size, sequence_length, self.config.EMBED_DIM)
        
        # Enhanced temporal fusion
        video_features = self.temporal_fusion(frame_features)
        
        # Enhanced landmark processing
        landmark_features = self.landmark_processor(landmarks)
        
        # Adaptive cross-modal fusion
        fused_features = self.cross_modal_fusion(video_features, landmark_features)
        
        # Enhanced pooling: combine average and max pooling
        fused_features_transposed = fused_features.transpose(1, 2)  # (B, D, T)
        avg_pooled = self.global_avg_pool(fused_features_transposed).squeeze(-1)
        max_pooled = self.global_max_pool(fused_features_transposed).squeeze(-1)
        
        # Concatenate pooled features
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits
    
    def get_attention_weights(self, frames, landmarks):
        """Get attention weights for visualization"""
        with torch.no_grad():
            # Similar forward pass but return attention weights
            # Implementation for debugging and visualization
            pass
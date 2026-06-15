import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMClassifier(nn.Module):
    """
    Combined CNN-LSTM model for processing temporal sequences of spatial-spectral EEG representations.
    """
    def __init__(self, cnn_feature_extractor, num_classes=5, lstm_hidden_size=256, num_lstm_layers=2, dropout_rate=0.5):
        super(CNNLSTMClassifier, self).__init__()
        self.cnn = cnn_feature_extractor
        
        # Determine the size of the feature vector extracted by CNN
        # Assuming the CNN has a sequential feature_extractor before the classifier
        self.cnn_out_dim = self._get_cnn_output_dim()
        
        # Sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def _get_cnn_output_dim(self):
        # Helper to extract the feature representation dimensions from the CNN feature extractor
        dummy_input = torch.zeros(1, self.cnn.feature_extractor[0].in_channels, 1, 9, 9)
        with torch.no_grad():
            feat = self.cnn.feature_extractor(dummy_input)
            feat = self.cnn.adaptive_pool(feat)
            return feat.view(1, -1).size(1)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, bands, grid_h, grid_w)
        batch_size, seq_len, bands, h, w = x.size()
        
        # Reshape for CNN feature extraction
        # We process each frame in the sequence through the CNN (treating it as temporal dimension depth = 1)
        x_reshaped = x.view(batch_size * seq_len, bands, 1, h, w)
        cnn_feats = self.cnn.feature_extractor(x_reshaped)
        cnn_feats = self.cnn.adaptive_pool(cnn_feats)
        cnn_feats = cnn_feats.view(batch_size, seq_len, -1)
        
        # Pass features through LSTM
        lstm_out, _ = self.lstm(cnn_feats)
        
        # Use final state for classification
        out = self.fc(lstm_out[:, -1, :])
        return out


class AdaptiveBottleneckTransformer(nn.Module):
    """
    Adaptive Bottleneck Transformer for Multimodal EEG, Audio, and Vision Fusion.
    Introduces shared bottleneck tokens to exchange cross-modal information efficiently.
    Reference: S. Bralina et al., Expert Systems with Applications, 2026.
    """
    def __init__(self, embed_dim=128, num_bottlenecks=4, num_heads=4, num_layers=2):
        super(AdaptiveBottleneckTransformer, self).__init__()
        
        self.num_bottlenecks = num_bottlenecks
        self.embed_dim = embed_dim
        
        # Learnable bottleneck tokens
        self.bottleneck_tokens = nn.Parameter(torch.randn(1, num_bottlenecks, embed_dim))
        
        # Linear projections for input modalities
        self.eeg_proj = nn.Linear(256, embed_dim)
        self.audio_proj = nn.Linear(128, embed_dim)
        self.video_proj = nn.Linear(512, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Modality specific pools
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ELU(),
            nn.Linear(64, 5) # 5 emotion classes
        )

    def forward(self, eeg_feat, audio_feat, video_feat):
        batch_size = eeg_feat.size(0)
        
        # Project inputs to common embedding dimension
        eeg_proj = self.eeg_proj(eeg_feat).unsqueeze(1)    # (B, 1, D)
        audio_proj = self.audio_proj(audio_feat).unsqueeze(1) # (B, 1, D)
        video_proj = self.video_proj(video_feat).unsqueeze(1) # (B, 1, D)
        
        # Tile bottleneck tokens for batch
        bottlenecks = self.bottleneck_tokens.expand(batch_size, -1, -1) # (B, N_b, D)
        
        # Multimodal sequence: [Bottlenecks, EEG, Audio, Video]
        multimodal_seq = torch.cat([bottlenecks, eeg_proj, audio_proj, video_proj], dim=1)
        
        # Run through transformer
        fused_seq = self.transformer(multimodal_seq)
        
        # Gather information from the updated bottleneck tokens
        fused_representation = fused_seq[:, :self.num_bottlenecks, :].mean(dim=1)
        
        return self.classifier(fused_representation)


class ReinforcementLearningAdaptation(nn.Module):
    """
    Policy Network for adaptive learning and fusion weight optimization.
    Dynamically generates weighting factors for multimodal features based on current noise/uncertainty.
    Part of Phase 5 active development (June 2026).
    """
    def __init__(self, state_dim=3, action_dim=3):
        super(ReinforcementLearningAdaptation, self).__init__()
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # Output weights sum to 1
        )
        
    def forward(self, modality_states):
        # modality_states: metrics like SNR, confidence, or noise levels for each modality
        weights = self.policy_network(modality_states)
        return weights

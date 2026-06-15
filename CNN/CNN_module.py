import torch
import torch.nn as nn
import numpy as np

class EEG3DCNN(nn.Module):
    """
    3D Convolutional Neural Network for spatial-spectral EEG feature classification.
    Designed for classification of 3D-mapped EEG signals (bands x grid_h x grid_w).
    """
    def __init__(self, in_channels=4, num_classes=5, conv_channels=[32, 64, 128], fc_features=512, dropout_rate=0.5):
        super(EEG3DCNN, self).__init__()
        
        # input shape: (batch_size, in_channels, depth, height, width)
        # where in_channels could be bands, depth is temporal window, height/width is spatial grid
        
        layers = []
        current_channels = in_channels
        
        for next_channels in conv_channels:
            layers.append(nn.Conv3d(current_channels, next_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
            layers.append(nn.BatchNorm3d(next_channels))
            layers.append(nn.ELU())
            layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            current_channels = next_channels
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Placeholder for adaptive pooling to handle variable input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 2, 2))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1] * 1 * 2 * 2, fc_features),
            nn.BatchNorm1d(fc_features),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_features, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features)
        out = self.classifier(features)
        return out


def generate_spatial_grid(eeg_data, channel_coords, grid_size=(9, 9)):
    """
    Converts 1D channel EEG data into a 2D spatial grid topology.
    eeg_data: numpy array of shape (num_samples, num_channels, num_bands)
    channel_coords: dict mapping channel name/index to (x, y) coordinates on a grid
    """
    num_samples, num_channels, num_bands = eeg_data.shape
    grid_data = np.zeros((num_samples, num_bands, grid_size[0], grid_size[1]))
    
    for ch_idx in range(num_channels):
        if ch_idx in channel_coords:
            x, y = channel_coords[ch_idx]
            grid_data[:, :, x, y] = eeg_data[:, ch_idx, :]
            
    return torch.tensor(grid_data, dtype=torch.float32)


# 30-channel standard coordinate mapping for the EAV dataset grid
EAV_30_GRID_COORDS = {
    0: (0, 3), 1: (0, 5),          # FP1, FP2
    2: (1, 2), 3: (1, 4), 4: (1, 6), # F7, F3, Fz
    5: (2, 2), 6: (2, 4), 7: (2, 6), # FT7, FC3, FCz
    8: (3, 1), 9: (3, 3), 10: (3, 5), 11: (3, 7), # T7, C3, Cz, T8
    12: (4, 2), 13: (4, 4), 14: (4, 6), # TP7, CP3, CPz
    15: (5, 1), 16: (5, 3), 17: (5, 5), 18: (5, 7), # P7, P3, Pz, P8
    19: (6, 2), 20: (6, 4), 21: (6, 6), # PO7, PO3, POz
    22: (7, 3), 23: (7, 5), # O1, O2
}

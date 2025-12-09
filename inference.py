
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

class RobustBiLSTM(nn.Module):
    def __init__(self, feat_dim=512, num_classes=42, hidden=384, num_layers=4, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(feat_dim)
        self.projection = nn.Linear(feat_dim, hidden)
        self.lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden, num_layers=num_layers,
            dropout=dropout, bidirectional=True, batch_first=False
        )
        self.output_proj = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, input_lengths):
        x = self.ln(x)
        x = self.projection(x)
        x = F.relu(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), enforce_sorted=False)
        out_packed, _ = self.lstm(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed)
        return self.output_proj(out)

def load_model(model_path, device):
    model = RobustBiLSTM()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).float() 
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model_lite.pth')
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(args.model) and os.path.exists(args.input):
        model = load_model(args.model, device)
        # Load data (Assuming input is .npy)
        # Add your data loading logic here for your group
        print("Model loaded successfully. Ready for inference.")
    else:
        print("Check your file paths.")

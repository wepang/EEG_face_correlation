import torch.nn as nn

class EEG_Transformer(nn.Module):
    def __init__(self, input_dim=24, num_classes=3, d_model=16, nhead=2, num_layers=1):
        super(EEG_Transformer, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim * d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
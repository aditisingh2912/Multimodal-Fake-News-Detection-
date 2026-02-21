import torch
import torch.nn as nn

# Keep this: It's the specific 'head' the model was trained with
class TokenClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.1):
        super(TokenClassifier, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(input_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class RED_DOT(nn.Module):
    def __init__(self, tf_layers=4, tf_head=8, tf_dim=128, emb_dim=768):
        super().__init__()
        # Use the parameters from your assignment
        self.emb_dim = emb_dim

        # This acts as the global summary of the news claim
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        # The core Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=tf_head,
                dim_feedforward=tf_dim,
                batch_first=True,
                norm_first=True  # Matches 'pre_norm=True' in original code
            ),
            num_layers=tf_layers,
        )

        # The specific head for True/Fake classification
        self.binary_classifier = TokenClassifier(self.emb_dim, self.emb_dim)

    def forward(self, x):
        # x shape: [Batch, 2, 768]
        b_size = x.shape[0]

        # Prepend the learnable CLS token
        cls_token = self.cls_token.expand(b_size, 1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Process through the Transformer
        x = self.transformer(x)

        # Take only the first token (CLS) for the final answer
        x_truth = x[:, 0, :]
        return self.binary_classifier(x_truth)
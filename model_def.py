import torch
import torch.nn as nn


class TokenClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.1):
        super(TokenClassifier, self).__init__()
        self.layer_norm   = nn.LayerNorm(input_size)
        self.gelu         = nn.GELU()
        self.dropout      = nn.Dropout(dropout_prob)
        self.fc           = nn.Linear(input_size, hidden_size // 2)
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
    """
    Guided multimodal fake-news detector.

    Sequence fed to the Transformer (matches training):
        [cls_token,  interaction,  image,  text]

    x_out[:, 0, :] is the cls_token — the learned global summary
    that the classifier head was trained to read from.

    Input to forward(): shape [B, 2, 768]
        index 0 → image embedding (CLIP ViT-L/14)
        index 1 → text  embedding (CLIP ViT-L/14)
    """

    def __init__(self, tf_layers=4, tf_head=8, tf_dim=128, emb_dim=768):
        super().__init__()
        self.emb_dim       = emb_dim
        self.model_version = "guided"

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=tf_head,
                dim_feedforward=tf_dim,
                batch_first=True,
                norm_first=True
            ),
            num_layers=tf_layers,
        )

        self.binary_classifier = TokenClassifier(self.emb_dim, self.emb_dim)

    def forward(self, x):
        # 1. Squeeze any accidental extra dim: [1,1,2,768] → [1,2,768]
        if x.dim() == 4:
            x = x.squeeze(1)

        # 2. Guarantee exactly 2 tokens arrive (image, text)
        x = x[:, :2, :]
        b_size = x.shape[0]

        # 3. Slice and L2-normalise
        img_feat_raw = x[:, 0, :]                                      # [B, 768]
        txt_feat_raw = x[:, 1, :]                                      # [B, 768]
        img_feat = img_feat_raw / img_feat_raw.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat_raw / txt_feat_raw.norm(dim=-1, keepdim=True)

        # 4. Cosine relevance scalar
        rel_img   = img_feat.unsqueeze(1)                              # [B, 1, 768]
        rel_txt   = txt_feat.unsqueeze(1)                              # [B, 1, 768]
        relevance = torch.bmm(rel_img, rel_txt.transpose(1, 2))        # [B, 1, 1]

        # 5. Guided interaction — near-zero when image/text are mismatched
        interaction = (img_feat * txt_feat) * relevance.view(b_size, 1)  # [B, 768]

        # 6. ── THE FIX ──────────────────────────────────────────────────
        # Prepend cls_token so the sequence matches what the model saw
        # during training:  [cls_token, interaction, image, text]
        #
        # x_out[:, 0, :] is then the cls_token output — the only token
        # the classifier head was trained to read from.
        #
        # Without this, x_out[:, 0, :] was reading the interaction vector
        # instead of cls_token, so the classifier received a vector it had
        # never been trained on → output insensitive to any input.
        cls_tokens = self.cls_token.expand(b_size, -1, -1)            # [B, 1, 768]

        x_seq = torch.cat(
            (cls_tokens, interaction.unsqueeze(1), rel_img, rel_txt),
            dim=1
        )  # [B, 4, 768]

        # 7. Transformer encoding
        x_out   = self.transformer(x_seq)
        x_truth = x_out[:, 0, :]    # cls_token output — trained classification anchor

        return self.binary_classifier(x_truth)
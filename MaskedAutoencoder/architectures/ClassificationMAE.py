import torch
import torch.nn as nn

class ClassificationMAE(nn.Module):
    """A linear classifier is trained on self-supervised representations learned by MAE.
    Args:
        n_classes: number of classes
        mae: mae model
        embedding_dim: embedding dimension of mae output
        detach: if True, only the classification head is updated.
    """
    def __init__(self, n_classes, mae, embedding_dim=256, detach=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mae = mae
        self.output_head = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, n_classes)
        )
        self.detach = detach

    def forward(self, images):
        """
        Args:
            Images: batch of images
        Returns:
            logits: batch of logits from the ouput_head
        Remember to detach the representations if self.detach=True, and
        Remember that we do not use masking here.
        """
        if self.detach:
          with torch.no_grad():
            embedding = self.mae.forward_encoder_representation(images)
        else:
            embedding = self.mae.forward_encoder_representation(images)

        return self.output_head(embedding)
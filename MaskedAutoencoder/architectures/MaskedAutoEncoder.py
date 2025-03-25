import torch
import torch.nn as nn
from data.utils import patchify
from architectures.utils import random_masking
class MaskedAutoEncoder(nn.Module):
    """MAE Encoder
    Args:
        encoder: vit encoder
        decoder: vit decoder
        encoder_embedding_dim: embedding size of encoder
        decoder_embedding_dim: embedding size of decoder
        patch_size: image patch size
        num_patches: number of patches
        mask_ratio: percentage of masked patches
    """
    def __init__(self, encoder, decoder, encoder_embedding_dim=256,
                 decoder_embedding_dim=128, patch_size=4, num_patches=8,
                 mask_ratio=0.75):
        super().__init__()
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

        self.masked_length = int(num_patches * num_patches * mask_ratio)
        self.keep_length = num_patches * num_patches - self.masked_length

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_input_projection = nn.Linear(patch_size * patch_size * 3, encoder_embedding_dim)
        self.decoder_input_projection = nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.decoder_output_projection = nn.Linear(decoder_embedding_dim, patch_size * patch_size * 3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embedding_dim) * 0.02)
        self.encoder_position_encoding = nn.Parameter(torch.randn(1, num_patches * num_patches, encoder_embedding_dim) * 0.02)
        self.decoder_position_encoding = nn.Parameter(torch.randn(1, num_patches * num_patches, decoder_embedding_dim) * 0.02)
        self.masked_tokens = nn.Parameter(torch.randn(1, 1, decoder_embedding_dim) * 0.02)

    def forward_encoder(self, images, ids_shuffle=None):
        """
        Encode input images using the following steps:

        1. Divide the images into smaller patches using the `patchify` function.
        2. Apply a linear projection to each image patch.
        3. Add position encoding to the projected patches.
        4. Mask out a subset of patches using the `random_masking` function.
           - Note that `ids_shuffle` is optional. If it is omitted, you need to
             generate a random permutation of patch indices and pass it to the
             `random_masking` function
        5. Concatenate the CLS token embedding with the masked patch embeddings.
           - The embedding of the CLS token is defined as `self.cls_token`
        6. Pass the combined tensor to the ViT encoder and return its output,
           along with the mask and the ids_restore tensor obtained in step 4.
        """
        patches = patchify(images, patch_size=self.patch_size)
        linear_embed = self.encoder_input_projection(patches)
        position_embed = linear_embed + self.encoder_position_encoding
        b, l, f = position_embed.size()
        keep_length = int((1 - self.mask_ratio) * l)

        if not ids_shuffle:

          ids_shuffle = torch.zeros((b, l), device=linear_embed.device, dtype = torch.long)
          for i in range(b):
            ids_shuffle[i] = torch.randperm(l, device=linear_embed.device, dtype = torch.long)

        kept, mask, ids_restore = random_masking(position_embed, keep_length=keep_length,ids_shuffle=ids_shuffle)
        cls_token = torch.broadcast_to(self.cls_token, (b, 1, f))
        embedding = torch.cat((kept, cls_token), 1)
        encoded_embedding = self.encoder(embedding)
        return encoded_embedding, mask, ids_restore


    def forward_decoder(self, encoder_embeddings, ids_restore):
        """
        Decode encoder embeddings using the following steps:

        1. Apply a linear projection to the encoder output.
        2. Extract the CLS token from the projected decoder embeddings and set
           it aside.
        3. Restore the sequence by inserting MASK tokens into the decoder
           embeddings, while also removing the CLS token from the sequence.
           - The embedding of the MASK token is defined as `self.masked_tokens`
        4. Add position encoding to the restored decoder embeddings.
        5. Re-concatenate the CLS token with the decoder embeddings.
        6. Pass the combined tensor to the ViT decoder, and retrieve the decoder
           output by excluding the CLS token.
        7. Apply the decoder output projection to the decoder output to predict
           image patches, and return the result.
        """
        ids_restore = ids_restore.to("cuda")
        linear_project = self.decoder_input_projection(encoder_embeddings)

        cls_token = linear_project[:, 0, :]
        linear_project_no_cls = linear_project[:, 1:]

        mask_length = ids_restore.size(1) - encoder_embeddings.size(1)

        mask_tensor = torch.broadcast_to(self.masked_tokens, (linear_project.size(0), mask_length, linear_project.size(2)))
        masked_embed = torch.zeros((linear_project.size(0), ids_restore.size(1), linear_project.size(2)), device = linear_project.device)
        full_x = torch.cat((mask_tensor, linear_project_no_cls), 1)

        for i in range(linear_project.size(0)):
          masked_embed[i] = full_x[i][ids_restore[i]]

        cls_token = torch.broadcast_to(cls_token, (masked_embed.size(0), 1, masked_embed.size(2)))

        decoder_input = torch.cat((cls_token, masked_embed), 1)

        decoder_output = self.decoder(decoder_input)[:, 1:]

        return self.decoder_output_projection(decoder_output)

    def forward(self, images):
        encoder_output, mask, ids_restore = self.forward_encoder(images)
        decoder_output = self.forward_decoder(encoder_output, ids_restore)
        return decoder_output, mask

    def forward_encoder_representation(self, images):
        """
        Encode input images **without** applying random masking, following step
        1, 2, 3, 5, 6 of `forward_encoder`
        """
        patches = patchify(images, patch_size=self.patch_size)
        linear_embed = self.encoder_input_projection(patches)
        position_embed = linear_embed + self.encoder_position_encoding
        b, l, f = position_embed.size()
        cls_token = torch.broadcast_to(self.cls_token, (b, 1, f))
        embedding = torch.cat((position_embed, cls_token), 1)
        encoded_embedding = self.encoder(embedding)
        return encoded_embedding
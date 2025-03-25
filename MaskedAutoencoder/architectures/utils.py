import torch

def index_sequence(x, ids):
    """Index tensor (x) with indices given by ids
    Args:
        x: input sequence tensor, can be 2D (batch x length) or 3D (batch x length x feature)
        ids: 2D indices (batch x length) for re-indexing the sequence tensor
    """
    if len(x.shape) == 3:
        ids = ids.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.take_along_dim(x, ids, dim=1)


def random_masking(x, keep_length, ids_shuffle):
    """Apply random masking on input tensor
    Args:
        x: input patches (batch x length x feature)
        keep_length: length of unmasked patches
        ids_shuffle: random indices for shuffling the input sequence. This is an
            array of size (batch x length) where each row is a permutation of
            [0, 1, ..., length-1]. We will pass this array to index_sequence function
            to chooose the unmasked patches.

    Returns:
        kept: unmasked part of x: (batch x keep_length x feature)
        mask: a 2D (batch x length) mask tensor of 0s and 1s indicated which
            part of x is masked out. The value 0 indicates not masked and 1
            indicates masked.
        ids_restore: indices to restore x. This is an array of size (batch x length).
    """
    kept_ids = ids_shuffle[:, :keep_length]
    kept = index_sequence(x, kept_ids)
    mask = torch.ones(x.size(0), x.size(1), device=x.device)
    mask.scatter_(1, kept_ids, 0)

    ids_restore = torch.argsort(ids_shuffle)

    return kept, mask, ids_restore


def restore_masked(kept_x, masked_x, ids_restore):
    """Restore masked patches
    Args:
        kept_x: unmasked patches: (batch x keep_length x feature)
        masked_x: masked patches: (batch x (length - keep_length) x feature)
        ids_restore: indices to restore x: (batch x length)
    Returns:
        restored patches
    """
    shuflled_x = torch.cat((kept_x, masked_x), 1)
    return index_sequence(shuflled_x, ids_restore)
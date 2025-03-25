import torch.cuda
import torch.optim as optim
from architectures.MaskedAutoEncoder import MaskedAutoEncoder
from architectures.Transformer import Transformer
from data.load_data import load_data
from data.utils import patchify
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os

if __name__ == '__main__':

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, _ = load_data()
    # Initialize MAE model
    model = MaskedAutoEncoder(
        Transformer(embedding_dim=256, n_layers=4),
        Transformer(embedding_dim=128, n_layers=2),
    )
    # Move the model to GPU
    model.to(torch_device)
    # Create optimizer

    # You may want to tune these hyperparameters to get better performance
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)

    total_steps = 0
    num_epochs = 20
    train_logfreq = 100

    losses = []

    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        # Train
        data_iterator = tqdm(trainloader)
        for x, y in data_iterator:
            total_steps += 1
            x = x.to(torch_device)
            image_patches = patchify(x)
            predicted_patches, mask = model(x)
            loss = torch.sum(torch.mean(torch.square(image_patches - predicted_patches), dim=-1) * mask) / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_postfix(loss=loss.item())
            if total_steps % train_logfreq == 0:
                losses.append(loss.item())

        # Periodically save model
        torch.save(model.state_dict(), os.path.join('../../models', "mae_pretrained.pt"))

    plt.plot(losses)
    plt.title('MAE Train Loss')
    plt.show()
    plt.savefig("../../reports/figures/MAE_train_loss.png", dpi=300, bbox_inches='tight')

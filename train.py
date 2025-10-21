import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from load_model import load_model_from_checkpoint


from models.diffusion import Diffusion
from models.encoder import Encoder
from models.CLIP import CLIP


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
model_file = "./data/v1-5-pruned-emaonly.ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = torch.Generator(device=device)
generator.seed()


max_steps = 1000
batch_size = 8
num_epochs = 200

models = load_model_from_checkpoint(model_file, device)
encoder = Encoder().to(device=device)
encoder.load_state_dict(models['encoder'], strict=True)

diffusion = Diffusion().to(device=device)

clip = CLIP(vocab_size=49408, n_dims=768, n_tokens=77, n_heads=12).to(device=device)
clip.load_state_dict(models['clip'], strict=True)

encoder.requires_grad_(False)
clip.requires_grad_(False)

tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

def cosine_beta_schedule(timesteps, s=0.008, device=device):
    t = torch.linspace(0, timesteps, timesteps + 1, device=device, dtype=torch.float64)
    f = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-8, 0.999).float()

betas = cosine_beta_schedule(max_steps, device=device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1: 
        emb = F.pad(emb, (0,1))
    return emb


transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH), antialias=True),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),
])


class Flickr30kDataset(Dataset):
    def __init__(self, split="train", transform=None):

        print(f"Loading Flickr30k dataset for split: {split}")
        self.dataset = load_dataset("ceyda/flickr30k_turkish_english", split=split)
        self.transform = transform
        print("Dataset loaded successfully.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
  
        image = item['image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image_tensor = self.transform(image)

        caption = item['caption_en'][0]

        return image_tensor, caption

# from torch.utils.data import Subset
# train_dataset = Subset(train_dataset, range(1000)) # Use only first 1000 samples

train_dataset = Flickr30kDataset(split="train", transform=transform)

test_dataset = Flickr30kDataset(split="validation", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4 
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)



optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)



for epoch in range(num_epochs):
    loss = 0.0

    pbar = tqdm(train_loader)
    for inputs, prompt in pbar:
        B = inputs.size(0)
        inputs = inputs.to(device)

        with torch.no_grad():
            tokens = tokenizer(list(prompt), padding="max_length", max_length=77,
                                 truncation=True, return_tensors="pt").input_ids
            context = tokens.input_ids.to(device)
            context = clip(context)

            x_0 = encoder(inputs, noise = 0) # B, 4, W, H

        t = torch.randint(0, max_steps, (B,), device=device) # B
        time_embedded = timestep_embedding(t, 320).to(device)

        noise = torch.randn_like(x_0, device=device) # B, 4, W, H
        
        alpha_bar_t = alpha_bars[t].view(B, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1-alpha_bar_t) * noise

        predicted_noise = diffusion(x_t, context, time_embedded)

        loss = F.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)

        optimizer.step()
    scheduler.step()

        
    print(loss.item())

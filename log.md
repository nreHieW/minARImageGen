### Compile 
```python
vqvae.module.encoder = torch.compile(vqvae.module.encoder, fullgraph=False, mode="max-autotune")
vqvae.module.decoder = torch.compile(vqvae.module.decoder, fullgraph=False, mode="max-autotune")
```
This gets `RuntimeError: cutlassF: no kernel found to launch!`

## Baselines
```python
import torch
from diffusers import FluxPipeline
import torch.nn.functional as F
from lpips import LPIPS
import os 
import torchvision 
from utils.evaluate.fid import calculate_fid_given_paths
from tqdm import tqdm
from utils.imagenet_dataset import ImageNetDataset, DataLoader

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
vae = pipe.vae.cuda()


val_ds = ImageNetDataset(root_dir="data/", split="validation")
val_loader = DataLoader(val_ds, batch_size=128, num_workers=4, pin_memory=True)
os.makedirs("ref_val_images/original", exist_ok=True)
os.makedirs("ref_val_images/generated", exist_ok=True)
recon_losses = []
perceptual_losses = []

lpips_model = LPIPS(net="vgg").cuda().eval()

with torch.no_grad():
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        images, _ = batch
        images = images.cuda().to(torch.bfloat16)
        xhat = vae(images).sample
        for b in range(images.size(0)):
            torchvision.utils.save_image(xhat[b], f"ref_val_images/generated/{i}_{b}.png")
            torchvision.utils.save_image(images[b], f"ref_val_images/original/{i}_{b}.png")
        
        recon_loss = F.mse_loss(xhat, images)
        perceptual_loss = lpips_model(xhat, images).mean()
        recon_losses.append(recon_loss.item())
        perceptual_losses.append(perceptual_loss.item())
calculate_fid_given_paths(["ref_val_images/original", "ref_val_images/generated"], batch_size=128, device="cuda")
```
This gives the following scores:
- FID: `3.3812174188498716`
- MSE: `0.2568434303069054`
- LPIPS: `0.22775398606382063`

# TODOs
loss weights

Replace `BatchNorm` with `SyncBatchNorm`. See [here](https://github.com/pytorch/pytorch/issues/66504)
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


```python
import torch
import torch.nn.functional as F
from lpips import LPIPS
import os 
import torchvision 
from utils.evaluate.fid import calculate_fid_given_paths
from tqdm import tqdm
from utils.imagenet_dataset import ImageNetDataset, DataLoader
from models.ref_vae import VQVAE

# Initialize the VQVAE model
path = "/home/ubuntu/minVAR/vae_ch160v4096z32.pth"
model = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True, 
              share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16))
model.load_state_dict(torch.load(path))
model = model.cuda().eval()

# Setup data and evaluation
val_ds = ImageNetDataset(root_dir="data/", split="validation")
val_loader = DataLoader(val_ds, batch_size=128, num_workers=4, pin_memory=True)
os.makedirs("ref_val_images_vqvae/original", exist_ok=True)
os.makedirs("ref_val_images_vqvae/generated", exist_ok=True)
recon_losses = []
perceptual_losses = []

lpips_model = LPIPS(net="vgg").cuda().eval()

with torch.no_grad():
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        images, _ = batch
        images = images.cuda()  # VQVAE might not need bfloat16
        
        # Get reconstruction (adjust based on your model's output format)
        xhat = model(images)[0]  # Modify if your model returns a different structure
        
        for b in range(images.size(0)):
            torchvision.utils.save_image(xhat[b], f"ref_val_images_vqvae/generated/{i}_{b}.png")
            torchvision.utils.save_image(images[b], f"ref_val_images_vqvae/original/{i}_{b}.png")
        
        recon_loss = F.mse_loss(xhat, images)
        perceptual_loss = lpips_model(xhat, images).mean()
        recon_losses.append(recon_loss.item())
        perceptual_losses.append(perceptual_loss.item())

# Calculate final metrics
avg_mse = sum(recon_losses) / len(recon_losses)
avg_lpips = sum(perceptual_losses) / len(perceptual_losses)
fid = calculate_fid_given_paths(["ref_val_images_vqvae/original", "ref_val_images_vqvae/generated"], 
                               batch_size=128, device="cuda")

print(f"FID: {fid}")
print(f"MSE: {avg_mse}")
print(f"LPIPS: {avg_lpips}")
```
This gives the following scores:
- FID: `3.4039871851974226`
- MSE: `0.3275837734760836`
- LPIPS: `0.26950350727723993`

MUON doesnt really work

At 16384 codebook size, the initial uniform entropy is log(16384) = 14
removing the 32th resolution screws with the codebook

## TODO

implement llamagen

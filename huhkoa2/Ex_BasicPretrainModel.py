from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
from diffusers import DDPMScheduler
import torch
from diffusers.utils import make_image_grid
from diffusers import UNet2DModel
import tqdm
import numpy as np

def main():
    repo = "google/ddpm-cat-256"
    model = UNet2DModel.from_pretrained(
        repo,
        use_safetensors=True
    )

    scheduler = DDPMScheduler.from_pretrained(repo)

    print(model)
    print(model.config)
    print(scheduler.config)
    
    
    torch.manual_seed(0)
    noisy_sample = torch.randn(1, model.config.in_channels, 
                               model.config.sample_size,
                               model.config.sample_size)
    
    model.to("cuda")
    noisy_sample = noisy_sample.to("cuda")
    
    def save_image_sample(sample, i):
        image = sample.cpu().permute(0,2,3,1)
        image = (image + 1.0)*127.5
        image = image.numpy().astype("uint8")
        image_pil = PIL.Image.fromarray(image[0])
        image_pil.save("sample_%03d.png" % i)
        
    
    sample = noisy_sample
    
    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        with torch.no_grad():
            residual = model(sample, t).sample
        sample = scheduler.step(residual,t,sample).prev_sample
        
        
    

if __name__ == "__main__":
    main()
import torch
from dataclasses import dataclass
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid
from diffusers import UNet2DModel
import os

@dataclass
class GenConfig:
    image_size = 128
    model_dir = "Assign03/gen_model"
    output_dir = "Assign03/gen_images"
    seed = 0
    eval_batch_size = 1
   

def main():
    config = GenConfig()
    
    modelLoc = os.path.join(config.model_dir, "finalModel.pt")
    if (not os.path.exists(modelLoc)):
        modelLoc = os.path.join(config.model_dir, "checkpoint.pt") #use latest checkpoint of final model does not exist
    if (not os.path.exists(modelLoc)):
        print("CANNOT FIND FINAL OR CHECKPOINT MODELS! EXITING!!!")
        exit(1)
        
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
       
    model = UNet2DModel(
        in_channels = 3,
        out_channels = 3,
        sample_size = config.image_size,
        layers_per_block = 2,
        block_out_channels = (128,128,256,256,512,512),
        down_block_types = [
            "DownBlock2D","DownBlock2D",
            "DownBlock2D","DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"
        ],
        up_block_types = [
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"
        ]        
    ).to(device)
   
    noise_scheduler = DDPMScheduler()
   
    os.makedirs(config.output_dir, exist_ok=True)
    finalModelDict = torch.load(modelLoc)
   
    model.load_state_dict(finalModelDict["network"])
   
    pipeline = DDPMPipeline(
        unet=model,
        scheduler=noise_scheduler)
   
    for i in range(1000):
        evaluateEpoch(config, i, pipeline)

def evaluateEpoch(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(epoch)
    ).images
   
    image_grid = make_image_grid(images, 1, 1)
   
    test_dir = os.path.join(config.output_dir)
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(os.path.join(test_dir,
                                    "Image_%04d.png" % epoch))

if __name__ == "__main__":
    main()
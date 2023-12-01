import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.utils import make_image_grid, pt_to_pil
from dataclasses import dataclass
from transformers import get_cosine_schedule_with_warmup
import tqdm
from accelerate import Accelerator, notebook_launcher

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    eval_batch_size = 16
    mixed_precision = "fp16"
    output_dir = "gen_model"
    gradient_accumulation_steps = 1
    start_epoch = 0
    total_epochs = 100
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 20
    overwrite_output_dir = True
    seed = 0


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        if image is not None:
            images.append(image)
    return images


def main():
    config = TrainingConfig()
   
    images = load_images_from_folder('Assign03/train_images')
    
    dataloader = DataLoader(images,
                            batch_size=config.train_batch_size,
                            shuffle=True)
 
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
    )
    print(model)
    
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader)*config.total_epochs))
    
    
    checkpoint_filename = os.path.join(config.output_dir,
                                       "checkpoint.pt")
    config.start_epoch = 0
    if os.path.exists(checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        config.start_epoch = checkpoint["epoch"]+1
        model.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("Loading previous checkpoint...")
    
    def evaluate(config, epoch, model, latents_shape):
        upleft = torch.randn(latents_shape,
                             generator=torch.manual_seed(
                                 config.seed))
        upright = torch.randn(latents_shape,
                             generator=torch.manual_seed(
                                 config.seed+1))
        downleft = torch.randn(latents_shape,
                             generator=torch.manual_seed(
                                 config.seed+2))
        downright = torch.randn(latents_shape,
                             generator=torch.manual_seed(
                                 config.seed+3))
        
        def sample(u,v,upleft,upright,downleft,downright):
            up = upleft + (upright - upleft)*u
            down = downleft + (downright - downleft)*u
            
            s = up + (down - up)*v
            return s
        
        cnt = int(np.sqrt(config.eval_batch_size))
        all_noise = []
        
        for v in np.linspace(0, 1, cnt):
            for u in np.linspace(0,1,cnt):
                all_noise.append(sample(u,v,
                                        upleft,upright,
                                        downleft,downright))
                
        all_noise = torch.stack(all_noise)
        all_noise = all_noise.to(model.device)
        images = model.decode(all_noise).sample
        images = images.cpu().detach()
        images = pt_to_pil(images)        
        
        image_grid = make_image_grid(images, cnt, cnt)
        
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(os.path.join(test_dir,
                                     "Image_%04d.png" % epoch))
    
    def train_loop(config, model, optimizer,
                   noise_scheduler, dataloader,
                   lr_scheduler):
        
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs")
        )
        
        (model,
         optimizer,
         dataloader,
         noise_scheduler,
         lr_scheduler) = accelerator.prepare(
                                        model,
                                        optimizer,
                                        dataloader,
                                        noise_scheduler,
                                        lr_scheduler
                                        )
         
        if accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers("training")  
        
        global_step = 0
        
        for epoch in range(config.start_epoch,
                           config.total_epochs):        
            progress_bar = tqdm.tqdm(total=len(dataloader),
                                disable=not accelerator.is_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for batch in dataloader:
                clean_images = batch["images"]                                
                
                with accelerator.accumulate(model):
                    latent_dist = model.encode(clean_images).latent_dist
                    latents = latent_dist.sample()
                    latents_shape = latents.shape[1:]
                    kl = latent_dist.kl()
                    gen_images = model.decode(latents).sample
                    
                    loss = F.mse_loss(gen_images,
                                      clean_images)
                    
                    loss += torch.mean(kl)*0.1
                    
                    
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "step": global_step,
                    "lr": lr_scheduler.get_last_lr()[0]                     
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler)
                
                if(epoch+1)%config.save_image_epochs == 0:
                    evaluate(config, epoch,
                             accelerator.unwrap_model(model),
                             latents_shape)
                                                 
                if(epoch+1)%config.save_model_epochs == 0:
                    pipeline.save_pretrained(config.output_dir)

                    unwrapped = accelerator.unwrap_model(model)
                    save_info = {
                        "epoch": epoch,
                        "network": unwrapped.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()                
                    }
                    torch.save(save_info, checkpoint_filename)
                        
                        
    args = (config, model, optimizer, noise_scheduler,
                dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)  

if __name__ == "__main__":
    main()

from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
import torch
from diffusers.utils import make_image_grid

def main():
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        #to make it faster
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    #can change scheduler to make it faster
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    
    #prints out configuration
    print(pipeline)
    
    print(pipeline.scheduler.compatibles)
    pipeline.to("cuda")
    pipeline.enable_attention_slicing()
    
    generator = torch.Generator(device ="cuda").manual_seed(7)
    
    prompt = "a cat on a sailing ship"
    
    def get_inputs(batch_size, prompt):
        generator = [torch.Generator("cuda").manual_seed(i)
                     for i in range(batch_size)]
        prompts = batch_size*[prompt]
        num_inference_steps=80
        
        #return a dictionary
        return{
            "generator": generator,
            "prompt": prompts,
            "num_inference_steps": num_inference_steps
        }
    
    #image = pipeline(prompt, generator=generator,
    #                 num_inference_steps=75).images[0]
    #image.save("test.png")    
    
    images = pipeline(**get_inputs(32, prompt)).images
    grid_image = make_image_grid(images, 8, 4)
    grid_image.save("grid.png")
    
if __name__=="__main__":
    main()
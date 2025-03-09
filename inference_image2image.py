import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

torch.device('mps')  # MPS is the Metal Performance Shaders backend

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
    , torch_dtype=torch.float16
    # , variant="fp16"
    , use_safetensors=True
).to("mps")
# pipeline.enable_model_cpu_offload()
# # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# prepare image
# url = "/Users/alejandrostawsky/Downloads/compressed_image.jpg" #
url = "/Users/alejandrostawsky/Downloads/professional headshot.jpeg" # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url).resize((512, 512))
init_image.show()

prompt = "Show this man and Lionel Messi happily posing for a picture in Punta del Este, both with Uruguayan soccer jerseys on. Make sure it is in 1024 by 1024 resolution."
# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.8, num_inference_steps=50, guidance_scale=18).images[0]  # Move the pipeline to MPS (Metal backend)
# image.show()
make_image_grid([init_image, image], rows=1, cols=2).show()

print()
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

torch.device('mps')  # MPS is the Metal Performance Shaders backend

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32
    # , variant="fp16"
    , use_safetensors=True
)
# pipeline.enable_model_cpu_offload()
# # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "/Users/alejandrostawsky/Downloads/professional headshot.jpeg" # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Show this man and Lionel Messi happily posing for a picture in Punta del Este, both with Uruguayan soccer jerseys on. Make sure it is 8k resolution."

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)

print()
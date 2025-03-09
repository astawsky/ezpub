from huggingface_hub import InferenceClient
from envs import hf_token, inference_client_model

client = InferenceClient(inference_client_model, token=hf_token)

text_for_image = "Lionel Messi with the Uruguayan jersey in Punta del Este, promoting the new addidas shoes. Resolution in 8K."

# output is a PIL.Image object
image = client.text_to_image(text_for_image)

image.show(title=text_for_image)

print()
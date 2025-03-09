from diffusers.utils import load_image
from PIL import Image

# Load the image from URL or local path
url = "/Users/alejandrostawsky/Downloads/professional headshot.jpeg"
init_image = load_image(url)

# Compress the image by saving it with lower quality (e.g., 50 out of 100)
compressed_image_path = "/Users/alejandrostawsky/Downloads/compressed_image.jpg"
init_image.save(compressed_image_path, "JPEG", quality=10)  # Lower quality for compression

# Load the compressed image again to display it
compressed_image = Image.open(compressed_image_path)

# Print the compressed image
compressed_image.show()

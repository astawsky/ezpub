import kagglehub

# Download latest version
path = kagglehub.dataset_download("groffo/ads16-dataset")

print("Path to dataset files:", path)
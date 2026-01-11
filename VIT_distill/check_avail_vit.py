import timm

# List all ViT models (wildcard search)
all_vits = timm.list_models('vit_*')

# List only those with pre-trained weights available
pretrained_vits = timm.list_models('vit_*', pretrained=True)

# Print the first 10 to check
print(pretrained_vits[:10])
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from fastai.vision.all import *
from torchsummary import summary

# %%
# Using device 2 to avoid other users on the server
torch.cuda.set_device(2)
default_device()

# %% [markdown]
# # Data Processing
#
# 1. Copy files from local box folder to the server (and convert HEIC)
#
# ```bash
# # From local machine
# mogrify -monitor -format jpg FrankOrFrary/**/*.HEIC
# find FrankOrFrary -name "*.HEIC" -print0 | xargs -0 rm -rf
# rsync -aivP FrankOrFrary dgx01:/data/cs152/
# ```
#
# 2. Convert file types (e.g., HEIC)
#
# ```bash
# #!/usr/bin/env bash
#
# # Set options for recurisve glob
# shopt -s globstar nullglob
#
# RAW_DATASET_PATH=/data/cs152/FrankOrFrary/raw
# PROCESSED_DATASET_PATH=/data/cs152/FrankOrFrary/processed
#
# # Loop over all images in the raw dataset
# for image_to_convert in "$RAW_DATASET_PATH"/**/*; do
#     # Skip directories
#     if [[ -d "$image_to_convert" ]]; then continue; fi
#
#     # Get the file name and replace extension with jpg
#     image_name=$(basename "$image_to_convert")
#     image_name="${image_name%.*}.jpg"
#
#     # Ensure the output directory exists
#     parent_name=$(basename "$(dirname "$image_to_convert")")
#     mkdir -p "$PROCESSED_DATASET_PATH"/"$parent_name"
#
#     # Create the new image name
#     image_name="$PROCESSED_DATASET_PATH"/"$parent_name"/"$image_name"
#
#     # Create the new image if it doesn't exist
#     if [[ -f "$image_name" ]]; then
#         echo "$image_name" already exists
#     else
#         echo "Creating $image_name"
#         convert "$image_to_convert" -strip -thumbnail '1000>' -format jpg "$image_name"
#     fi
# done
#
# ```
#
#
# Some issues:
#
# ```text
# convert: Invalid SOS parameters for sequential JPEG `/data/cs152/FrankOrFrary/raw/Frary/Alan-Frary-15.jpg' @ warning/jpeg.c/JPEGWarningHandler/403.
#
# convert: no decode delegate for this image format `HEIC' @ error/constitute.c/ReadImage/746.
# convert: no images defined `/data/cs152/FrankOrFrary/processed/Frary/Aldo-frary-03.jpg' @ error/convert.c/ConvertImageCommand/3342.
# ```

# %%
path = Path("/data/cs152/FrankOrFrary/processed")
path.ls()

# %%
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224), num_workers=16)
dls.show_batch()
# file <filename> on invalid files

# %%
print("Validation dataset size:", len(dls.valid_ds))
print("Training dataset size:", len(dls.train_ds))

# %%
learn = vision_learner(dls, resnet34, metrics=accuracy)
summary(learn.model);

# %%
learn.lr_find()

# %%
learn.fine_tune(4, 5e-3)

# %%
learn.show_results()

# %%
# interp = Interpretation.from_learner(learn)
interp = ClassificationInterpretation.from_learner(learn)

# %%
interp.plot_top_losses(9, figsize=(10, 10))

# %%
interp.plot_confusion_matrix(figsize=(10, 10))

# %%
learn.export("./FrankOrFraryResnet.pkl")

# %%
# !mv /data/cs152/FrankOrFrary/processed/FrankOrFraryResnet.pkl .

# %%
# !ls

# %%

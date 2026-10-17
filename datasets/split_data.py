from pathlib import Path
import random
import shutil

import yaml

def split_data(source_path, target_path,  already_split=False, split_ratio=[0.75, 0.15, 0.10]):

    # Get path to source and target dirs
    source = Path(source_path)
    target = Path(target_path)

    if target.exists():
        raise FileExistsError("Target directory already exists.")

    if already_split:
        folders = [source / "train", source / "valid"]

    else:
        folders = [source / "export"]

    # Collect all image and lables from the source directories
    pairs = []
    for folder in folders:
        for image in (folder / "images").glob("*.jpg"):
            label = folder / "labels" / f"{image.stem}.txt"
            pairs.append((image, label))

    # Randomize the order
    random.Random(69).shuffle(pairs)


    # Split the images into train, validation, and test sets
    n_train = int(len(pairs) * split_ratio[0])
    n_val = int(len(pairs) * split_ratio[1])
    #n_test = int(len(pairs) * split_ratio[2])

    splits = {
    "train": pairs[:n_train],
    "valid": pairs[n_train:n_train + n_val],
    "test": pairs[n_train + n_val:],
    }

    for splits_name, selected_pairs in splits.items():
        image_target = target / splits_name / "images"
        label_target = target / splits_name / "labels"
        image_target.mkdir(parents=True)
        label_target.mkdir(parents=True)
    
        for index, (image, label) in enumerate(selected_pairs):
            shutil.copy2(image, image_target / f"{index}.jpg")
            shutil.copy2(label, label_target / f"{index}.txt")
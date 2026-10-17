import argparse

import torch
import yaml
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init", 
        choices = ["coco", "scratch"], 
        required = True
    )
    
    parser.add_argument(
    "--n_epochs",
    type=int,
    default=None,
    )

    parser.add_argument(
    "--data",
    choices=["potholes", "traffic"],
    default="potholes",
    )

    #Parse arguments
    args = parser.parse_args()

    if args.init == "coco":
        model = YOLO("yolov8n.pt")
    elif args.init == "scratch":
        model = YOLO("yolov8n.yaml")


    dataset_paths = {
    "potholes": "datasets/potholes/new_splits/data.yaml",
    "traffic": "datasets/traffic/data.yaml",
    }        


    # Load training config
    with open("configs/train_config.yaml") as file:
        config = yaml.safe_load(file)

    # If not specified, use the default number of epochs from the config file.
    if args.n_epochs is not None:
        config["epochs"] = args.n_epochs

    config["device"] = 0 if torch.cuda.is_available() else "cpu"
    epochs = config["epochs"]

    model.train(
        data=dataset_paths[args.data],
        project="runs",
        name=f"{args.data}_{args.init}_{epochs}epochs",
        **config,
    )

if __name__ == "__main__":

    main()
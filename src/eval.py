import argparse

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path", 
        type=str,
        required=True,
    )
    
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, such as 'cpu' or '0'. Defaults to CUDA when available.",
    )

    #Parse arguments
    args = parser.parse_args()
    data = args.test_data_path
    model = YOLO(args.weights_path)

    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    metrics = model.val(
        data=data,
        split="test",
        imgsz=640,
        device=device,
        plots=True,
    )

    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    main()
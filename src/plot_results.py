import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_results(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows or "epoch" not in reader.fieldnames:
        raise ValueError(f"{path} does not contain valid Ultralytics results")

    columns = {
        name: [float(row[name]) for row in rows]
        for name in reader.fieldnames
        if name != "epoch"
    }
    columns["epoch"] = [float(row["epoch"]) for row in rows]
    return columns


def plot_results(results: dict[str, list[float]], output: Path) -> None:
    epochs = results["epoch"]
    groups = [
        ("Losses", ["train/box_loss", "train/cls_loss", "train/dfl_loss",
                    "val/box_loss", "val/cls_loss", "val/dfl_loss"]),
        ("Metrics", ["metrics/precision(B)", "metrics/recall(B)",
                     "metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, (title, columns) in zip(axes, groups):
        for column in columns:
            if column in results:
                axis.plot(epochs, results[column], label=column)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ultralytics results.csv")
    parser.add_argument("results_csv", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, help="Output image path (default: next to CSV)"
    )
    args = parser.parse_args()

    output = args.output or args.results_csv.with_name("results.png")
    plot_results(read_results(args.results_csv), output)


if __name__ == "__main__":
    main()

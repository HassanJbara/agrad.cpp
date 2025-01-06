import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def generate_moon_dataset(output_dir, n_samples=100, noise=0.1):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the dataset
    X, y = make_moons(n_samples=n_samples, noise=noise)
    y = y * 2 - 1  # make y be -1 or 1

    # Save to CSV
    csv_path = os.path.join(output_dir, "moon_dataset.csv")
    np.savetxt(
        csv_path,
        np.hstack((X, y.reshape(-1, 1))),
        delimiter=",",
        header="x1,x2,label",
        comments="",
    )

    # Create visualization
    plt.figure(figsize=(6, 5))

    # Plot classes with different colors
    mask_neg = y == -1
    mask_pos = y == 1
    plt.scatter(X[mask_neg, 0], X[mask_neg, 1], c="blue", label="Class 0", s=50)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c="red", label="Class 1", s=50)
    plt.title("Moon Dataset")
    plt.xlabel("X₁")
    plt.ylabel("X₂")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "moon_dataset_visualization.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate moon-shaped dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for dataset and visualization",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples in the dataset"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1, help="Noise level in the dataset"
    )

    args = parser.parse_args()
    generate_moon_dataset(args.output_dir, args.n_samples, args.noise)

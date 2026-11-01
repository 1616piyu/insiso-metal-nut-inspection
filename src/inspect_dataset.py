
from pathlib import Path

DATASET = Path("data/raw/metal_nut")

def main():
    folders = [
        "train/good",
        "test/good",
        "test/bent",
        "test/color",
        "test/flip",
        "test/scratch",
        "ground_truth/bent",
        "ground_truth/color",
        "ground_truth/flip",
        "ground_truth/scratch",
    ]

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET}"
        )

    for folder in folders:
        path = DATASET / folder

        if not path.exists():
            print(f"MISSING: {folder}")
            continue

        images = [
            p for p in path.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        print(f"{folder:25s}: {len(images)} files")

if __name__ == "__main__":
    main()


from pathlib import Path
import random
import shutil

SOURCE = Path("data/raw/metal_nut")
PROCESSED = Path("data/processed")
BASE_OUTPUT = Path("data/yolo_rebuilt")

SEED = 42

# Number of defective images assigned to train, validation, test.
DEFECT_SPLITS = {
    "bent": (18, 4, 3),
    "color": (15, 3, 4),
    "flip": (16, 3, 4),
    "scratch": (16, 4, 3),
}

SPLITS = ["train", "val", "test"]

EXPECTED_COUNTS = {
    "train": 263,
    "val": 36,
    "test": 36,
}


def choose_output_directory():
    """Choose a new directory without deleting previous outputs."""

    output = BASE_OUTPUT
    suffix = 2

    while output.exists():
        output = Path(f"{BASE_OUTPUT}_{suffix}")
        suffix += 1

    return output


def copy_image_and_label(image_path, split, output, label_source=None):
    """Copy an image and its label into the selected split."""

    image_dest = output / "images" / split / image_path.name
    label_dest = output / "labels" / split / f"{image_path.stem}.txt"

    image_dest.parent.mkdir(parents=True, exist_ok=True)
    label_dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_dest)

    if label_source is None:
        # Normal images have no defect boxes.
        label_dest.write_text("", encoding="utf-8")
    else:
        if not label_source.exists():
            raise FileNotFoundError(
                f"Annotation not found for {image_path.name}: "
                f"{label_source}"
            )

        shutil.copy2(label_source, label_dest)


def main():
    rng = random.Random(SEED)

    # Validate the normal-image directories.
    good_train_dir = SOURCE / "train" / "good"
    good_test_dir = SOURCE / "test" / "good"

    if not good_train_dir.exists():
        raise FileNotFoundError(f"Missing directory: {good_train_dir}")

    if not good_test_dir.exists():
        raise FileNotFoundError(f"Missing directory: {good_test_dir}")

    good_train = sorted(good_train_dir.glob("*.png"))
    good_test = sorted(good_test_dir.glob("*.png"))

    if len(good_train) != 220:
        raise ValueError(
            f"Expected 220 normal training images, found {len(good_train)}"
        )

    if len(good_test) != 22:
        raise ValueError(
            f"Expected 22 normal test images, found {len(good_test)}"
        )

    rng.shuffle(good_train)
    normal_train = good_train[:198]
    normal_val = good_train[198:]

    # Validate processed defect images and their labels before copying.
    defect_data = {}

    for category, counts in DEFECT_SPLITS.items():
        image_dir = PROCESSED / "images" / "defects"
        label_dir = PROCESSED / "labels" / "defects"

        images = sorted(image_dir.glob(f"{category}_*.png"))
        expected = sum(counts)

        if len(images) != expected:
            raise ValueError(
                f"{category}: expected {expected} images, found {len(images)}"
            )

        for image in images:
            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                raise FileNotFoundError(f"Missing label: {label}")

            # Validate YOLO label fields and normalized coordinates.
            for line_number, line in enumerate(
                label.read_text(encoding="utf-8").splitlines(), start=1
            ):
                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid YOLO label: {label}, line {line_number}"
                    )

                cls, xc, yc, width, height = map(float, parts)

                if cls != 0:
                    raise ValueError(f"Unexpected class ID in {label}")

                if not (
                    0 <= xc <= 1
                    and 0 <= yc <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                ):
                    raise ValueError(
                        f"Coordinates outside valid range: "
                        f"{label}, line {line_number}"
                    )

        defect_data[category] = images

    # Use a fresh output directory. Nothing existing is deleted.
    output = choose_output_directory()

    print(f"Creating dataset at: {output.resolve()}")

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    # 1. Normal train and validation images.
    for image in normal_train:
        copy_image_and_label(image, "train", output)

    for image in normal_val:
        copy_image_and_label(image, "val", output)

    # 2. Keep the original normal test images held out.
    for image in good_test:
        copy_image_and_label(image, "test", output)

    # 3. Add defective images according to the fixed split counts.
    for category, counts in DEFECT_SPLITS.items():
        images = defect_data[category].copy()
        rng.shuffle(images)

        start = 0

        for split, count in zip(SPLITS, counts):
            selected = images[start:start + count]
            start += count

            for image in selected:
                label = (
                    PROCESSED / "labels" / "defects"
                    / f"{image.stem}.txt"
                )

                copy_image_and_label(
                    image, split, output, label_source=label
                )

    # 4. Verify final dataset counts.
    print("\nDataset preparation complete.")

    for split in SPLITS:
        images = list((output / "images" / split).glob("*.png"))
        labels = list((output / "labels" / split).glob("*.txt"))

        if len(images) != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f"{split}: expected {EXPECTED_COUNTS[split]} images, "
                f"found {len(images)}"
            )

        if len(labels) != len(images):
            raise RuntimeError(
                f"{split}: {len(images)} images but {len(labels)} labels"
            )

        print(f"{split:5s}: {len(images)} images, {len(labels)} labels")

    print("\nSUCCESS: dataset created without deleting old files.")
    print(f"Dataset path: {output.resolve()}")
    print("Update configs/metal_nut.yaml to use this directory.")


if __name__ == "__main__":
    main()

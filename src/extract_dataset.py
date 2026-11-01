
from pathlib import Path
import tarfile

ARCHIVE = Path("data/raw/metal_nut.tar.xz")
OUTPUT_DIR = Path("data/raw")

def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {ARCHIVE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(ARCHIVE, mode="r:xz") as archive:
        # Check paths before extraction.
        for member in archive.getmembers():
            target = (OUTPUT_DIR / member.name).resolve()

            if not target.is_relative_to(OUTPUT_DIR.resolve()):
                raise ValueError(
                    f"Unsafe archive path: {member.name}"
                )

        archive.extractall(OUTPUT_DIR)

    print("Dataset extracted successfully.")
    print("Dataset location:", OUTPUT_DIR / "metal_nut")

if __name__ == "__main__":
    main()

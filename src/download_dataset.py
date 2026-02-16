import zipfile
from pathlib import Path
from PIL import Image
import io
import random

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data"
ZIP_PATH = Path.home() / "tensorflow_datasets/downloads/cats_vs_dogs" / (
    "down.micr.com_down_3_E_1_3E1C-ECDB-4869-83t5dL0AqEqZkh827kQD8ImFN3e1ro0VHHaobmSQAzSvk.zip"
)

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# test gets the remaining 10%

SEED = 42

def make_dirs():
    for split in ("train", "val", "test"):
        for cls in ("cats", "dogs"):
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def extract_dataset():
    make_dirs()

    with zipfile.ZipFile(ZIP_PATH) as zf:
        # Collect valid image paths for each class
        cat_files = sorted(
            n for n in zf.namelist()
            if n.startswith("PetImages/Cat/") and n.lower().endswith(".jpg")
        )
        dog_files = sorted(
            n for n in zf.namelist()
            if n.startswith("PetImages/Dog/") and n.lower().endswith(".jpg")
        )

        print(f"Found {len(cat_files)} cat images, {len(dog_files)} dog images")

        for cls_name, file_list in [("cats", cat_files), ("dogs", dog_files)]:
            random.seed(SEED)
            random.shuffle(file_list)

            n = len(file_list)
            n_train = int(n * TRAIN_RATIO)
            n_val   = int(n * VAL_RATIO)

            splits = {
                "train": file_list[:n_train],
                "val":   file_list[n_train:n_train + n_val],
                "test":  file_list[n_train + n_val:],
            }

            for split, files in splits.items():
                saved = 0
                for i, zpath in enumerate(files):
                    try:
                        data = zf.read(zpath)
                        # Validate it's a readable image
                        img = Image.open(io.BytesIO(data))
                        img.verify()
                        out_path = OUT_DIR / split / cls_name / f"{cls_name}_{i}.jpg"
                        out_path.write_bytes(data)
                        saved += 1
                    except Exception:
                        pass  # skip corrupt files (a few exist in the dataset)
                print(f"  {split}/{cls_name}: {saved} images saved")

    print("\nDone! Dataset is ready in data/train, data/val, data/test")

if __name__ == "__main__":
    extract_dataset()

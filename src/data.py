from . import library as lib

IMG_SIZE = 128

PROJECT_ROOT = lib.os.path.dirname(lib.os.path.dirname(lib.os.path.abspath(__file__)))
DATASET_DIR = lib.os.path.join(PROJECT_ROOT, "dataset")


def find_folder(root, name):
    for current, dirnames, _ in lib.os.walk(root):
        for d in sorted(dirnames):
            if d.lower() == name.lower():
                return lib.os.path.join(current, d)
    return None


def download_dataset():
    if not lib.os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    train_path = find_folder(DATASET_DIR, "Training")
    test_path = find_folder(DATASET_DIR, "Testing")

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Could not find 'Training' and 'Testing' folders inside {DATASET_DIR}"
        )

    return train_path, test_path


def get_classes(train_path):
    classes = sorted(lib.os.listdir(train_path))
    return classes


def load_dataset(dataset_path, classes):
    x_data = []
    y_data = []

    for label, class_name in enumerate(classes):

        class_folder = lib.os.path.join(dataset_path, class_name)

        if not lib.os.path.isdir(class_folder):
            continue

        for file in lib.os.listdir(class_folder):
            img_path = lib.os.path.join(class_folder, file)

            image = lib.cv2.imread(img_path)
            if image is None:
                continue

            image = lib.cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image.astype(lib.np.float32)
            image = image / 255.0

            x_data.append(image)
            y_data.append(label)

    x_data = lib.np.array(x_data, dtype=lib.np.float32)
    y_data = lib.np.array(y_data, dtype=lib.np.int32)

    x_data = lib.np.transpose(x_data,(0, 3, 1, 2))
    return x_data, y_data
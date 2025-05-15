import os

def get_class_labels(data_dir: str) -> dict:
    """
    Generates a mapping {index: class_name} from a dataset directory.
    The order matches torchvision.datasets.ImageFolder behavior (alphabetically sorted subfolder names).

    Args:
        data_dir (str): Path to the 'train' directory.

    Returns:
        dict: Mapping from class index to class name.
    """
    classes = []
    for root, dirs, files in os.walk(data_dir):
        if root == data_dir:
            continue
        rel_path = os.path.relpath(root, data_dir)
        if files:  # Include only folders that contain image files
            classes.append(rel_path.replace("\\", "/"))  # Normalize path for Windows

    # Sort classes alphabetically (same behavior as ImageFolder)
    classes = sorted(classes)

    # Build index-to-class mapping
    class_labels = {i: name for i, name in enumerate(classes)}
    return class_labels

# Example usage
if __name__ == "__main__":
    train_dir = "plant_disease_dataset/train"
    labels = get_class_labels(train_dir)
    print("# CLASS_LABELS dictionary:\nCLASS_LABELS = {")
    for idx, name in labels.items():
        print(f'    {idx}: "{name}",')
    print("}")

import os

def remove_suffixes(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            old_path = os.path.join(root, file)
            new_filename = file.replace('_A', '').replace('_B', '')

            new_path = os.path.join(root, new_filename)
            
            os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_paths = ["path/to/A/train", "path/to//B/train"]

    for folder_path in folder_paths:
        remove_suffixes(folder_path)
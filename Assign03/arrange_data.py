import os
import shutil

def reorganize_directory_structure(original_path, new_structure_path):
    if not os.path.exists(new_structure_path):
        os.makedirs(new_structure_path)

    for label in ['A', 'B']:
        for phase in ['train', 'test']:
            source_folder = os.path.join(original_path, f'{phase}{label}')
            destination_folder = os.path.join(new_structure_path, label, phase)

            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            for filename in os.listdir(source_folder):
                source_file_path = os.path.join(source_folder, filename)
                destination_file_path = os.path.join(destination_folder, filename)
                shutil.copy(source_file_path, destination_file_path)

if __name__ == "__main__":
    original_path = "/path/to/archive" #this is the facades dataset directory
    new_structure_path = "facade_dataset_pix2pix"
    reorganize_directory_structure(original_path, new_structure_path)

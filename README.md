# CS 548-12 Image Generation and Translation
## Getting the dataset from the repo
The facades dataset that is curated for the Pix2Pix model is located in the Assign02 directory under Facades. This dataset is used to run both Assign02 code and Assign03 scripts. The `A03_data.py` script has a path to the dataset in the `Assign02/Facades` directory and uses the the `A/train` building facade images.


## Getting the dataset from Kaggle
The original dataset that was used for CycleGAN was taken from: https://www.kaggle.com/datasets/balraj98/facades-dataset but it has to be modified for Pix2Pix.

**To obtain and modify the dataset:**
1. Download the dataset from https://www.kaggle.com/datasets/balraj98/facades-dataset
2. It will download as archive.zip
3. Unzip the file to a folder in your repository.
4. Copy the relative path of the unzipped folder.
5. Paste the path for the `original_path` variable under main in the `arrange_data.py` script in Assign03.
6. Run the `arrange_data.py` script.
7. Copy the relative filepaths to the `folder_paths` array under main in the `rename_filenames.py` script in Assign03.
8. Run the `rename_filenames.py` script .
9. Use the command line to run the `combine_A_and_B.py` script that was taken from the CG repo. `python Assign03/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data`
10. Change the `source_folder_path` variable in the `A03_data.py` script to properly load the data.



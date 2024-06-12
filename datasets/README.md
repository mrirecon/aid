# Preparing datasets
This README file provides instructions for using the Python script to preprocess the [fastMRI](https://fastmri.med.nyu.edu/) dataset. The script processes HDF5 files, extracts k-space data, performs the Fourier transform to reconstruct images, and saves the images in cfl format.


## Usage

### Command-Line Arguments

- `--data_folder`: Path to the folder containing the input `.h5` files. Default is `/scratch/gluo/compressed/fastMRI/multicoil_train`.
- `--save_folder`: Path to the folder where the processed images will be saved. Default is `/scratch/gluo/fastMRI`.
- `--start_id`: Starting ID for naming the output files. Default is `1000000`.

### Running the Script

1. Open a terminal or command prompt.
2. Navigate to the directory containing the script.
3. Download the BART binary file from the provided link, make it executable and set the `BART_PATH` environment variable to it.
3. Run the script with the desired arguments.

For example:
```bash
wget -nv https://huggingface.co/Guanxiong/MRI-Image-Priors/resolve/main/Data/bart
chmod +x bart
export BART_PATH=$(pwd)/bart
python fastmri.py --data_folder /path/to/input --save_folder /path/to/output --start_id 1000
```

## Utility Functions

The script uses the following utility functions from the `utils` module:

- `utils.bart`: Executes BART (Berkeley Advanced Reconstruction Toolbox) commands.
- `utils.check_out`: Executes shell commands.
- `utils.getname`: Generates a standardized file name for the output.
- `utils.writecfl`: Writes complex images to files in CFL format.

Ensure these functions are correctly implemented in the `utils` module.

## Error Handling

The script handles corrupted HDF5 files by printing an error message and skipping the file. If no files are found in the specified data folder, it prints a "No files found" message and exits.

We provided the binary file for BART reconstruction toolbox on this [link](https://huggingface.co/Guanxiong/MRI-Image-Priors/tree/main/Data). If it doesn't work on your local system, please clone the [BART repository](https://github.com/mrirecon/bart) and compile it on your local system and set the `BART_PATH` environment variable to the path of the compiled binary.

## Contact

For any questions or issues, please contact the script author or maintainer.

# Preparing datasets
This README file provides instructions for using the Python script to preprocess the the [fastMRI](https://fastmri.med.nyu.edu/) dataset. The script processes HDF5 files, extracts k-space data, performs the Fourier transform to reconstruct images, and saves the images in cfl format.


## Usage

### Command-Line Arguments

- `--data_folder`: Path to the folder containing the input `.h5` files. Default is `/scratch/gluo/compressed/fastMRI/multicoil_train`.
- `--save_folder`: Path to the folder where the processed images will be saved. Default is `/scratch/gluo/fastMRI`.
- `--start_id`: Starting ID for naming the output files. Default is `1000000`.

### Running the Script

1. Open a terminal or command prompt.
2. Navigate to the directory containing the script.
3. Run the script with the desired arguments. For example:

```bash
python preprocess_fastmri.py --data_folder /path/to/input --save_folder /path/to/output --start_id 1000
```

### Example

```bash
python preprocess_fastmri.py --data_folder /scratch/gluo/compressed/fastMRI/multicoil_train --save_folder /scratch/gluo/fastMRI --start_id 1000000
```

This command processes the fastMRI dataset located in `/scratch/gluo/compressed/fastMRI/multicoil_train` and saves the processed images to `/scratch/gluo/fastMRI`, starting with a file ID of `1000000`.

## Utility Functions

The script uses the following utility functions from the `utils` module:

- `utils.bart`: Executes BART (Berkeley Advanced Reconstruction Toolbox) commands.
- `utils.check_out`: Executes shell commands.
- `utils.getname`: Generates a standardized file name for the output.
- `utils.writecfl`: Writes complex images to files in CFL format.

Ensure these functions are correctly implemented in the `utils` module.

## Error Handling

The script handles corrupted HDF5 files by printing an error message and skipping the file. If no files are found in the specified data folder, it prints a "No files found" message and exits.

## Contact

For any questions or issues, please contact the script author or maintainer.

---

This README should help you understand and use the fastMRI preprocessing script effectively. Make sure to check and adapt the paths and parameters according to your environment and data.
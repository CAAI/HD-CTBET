# HD-CTBET

HD-CTBET is a fork of HD-BET repository, but targeted for CT BET rather than MRI-based. For the original repo, see https://github.com/MIC-DKFZ/HD-BET and the original publication:

Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W,
Bendszus M, Maier-Hein KH, Kickingereder P. Automated brain extraction of multi-sequence MRI using artificial neural
networks. Hum Brain Mapp. 2019; 1–13. https://doi.org/10.1002/hbm.24750


## Installation Instructions

1) Clone this repository:
    ```bash
    git clone https://github.com/CAAI/HD-CTBET
    ```
2) Go into the repository (the folder with the setup.py file) and install:
    ```
    cd HD-CTBET
    pip install -e .
    ```
3) Per default, model parameters will be downloaded from [zenodo](https://zenodo.org/record/7970359) to ~/hd-bet_params. If you wish to use a different folder, open
HD_CTBET/paths.py in a text editor and modify ```folder_with_parameter_files```


## How to use it

```bash
hd-ctbet -i INPUT_FILENAME
```

INPUT_FILENAME must be a nifti (.nii.gz) file containing 3D MRI image data.

For batch processing it is faster to process an entire folder at once as this will mitigate the overhead of loading
and initializing the model for each case:

```bash
hd-ctbet -i INPUT_FOLDER -o OUTPUT_FOLDER
```

The above command will look for all nifti files (*.nii.gz) in the INPUT_FOLDER and save the brain masks under the same name
in OUTPUT_FOLDER.

### GPU is nice, but I don't have one of those... What now?

HD-CTBET has CPU support. Running on CPU takes a lot longer though and you will need quite a bit of RAM. To run on CPU,
we recommend you use the following command:

```bash
hd-ctbet -i INPUT_FOLDER -o OUTPUT_FOLDER -device cpu -mode fast -tta 0
```
This works of course also with just an input file:

```bash
hd-ctbet -i INPUT_FILENAME -device cpu -mode fast -tta 0
```

The options *-mode fast* and *-tta 0* will disable test time data augmentation (speedup of 8x) and use only one model instead of an ensemble of five models
for the prediction.

### More options:
For more information, please refer to the help functionality:

```bash
hd-ctbet --help
```

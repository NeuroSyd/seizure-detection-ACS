# Supervised learning in automatic channel selection for epileptic seizure detection

##### Code to reproduce results reported in our paper published as:
Truong, N. D., L. Kuhlmann, M. R. Bonyadi, J. Yang, A. Faulks, and O. Kavehei (2017). "Supervised learning in automatic channel selection for epileptic seizure detection." _Expert Systems with Applications_ 86, 199-207. DOI:10.1016/j.eswa.2017.05.055.

This project aims to improve computational efficiency of the winning solution by Michael Hills (https://github.com/MichaelHills/seizure-detection) for UPenn and Mayo Clinic's Seizure Detection Challenge on Kaggle (https://www.kaggle.com/c/seizure-detection). By using our automatic channel selection engine, the run-time seizure detection is 2x faster than Hills approach.

#### How to run the code
1. Set the paths in the \*.json file.

2. Set number of channels to be selected for each target in file nchannels.csv. To select all channels, set the number of channels larger than or equal to the number of channels of each target.

3. Run cross-validation.
```console
python cross_validation.py
```

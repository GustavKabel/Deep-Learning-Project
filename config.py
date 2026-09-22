# config.py
# Computed global statistics for Z-score standardization of 13 Sentinel-2 bands
GLOBAL_MEAN = [0.03231, 0.02861, 0.02610, 0.02481, 0.02845, 0.04019, 0.04543, 0.04415, 0.04859, 0.01888, 0.00245, 0.02901, 0.01991]
GLOBAL_STD = [0.02487, 0.02588, 0.02521, 0.02787, 0.02776, 0.02845, 0.03004, 0.02922, 0.03087, 0.01691, 0.00463, 0.02132, 0.01745]

# Inverse frequency class weights for E1 Loss Optimization (Missing and Undefined masked to 0.0)
LOSS_WEIGHTS = [0.0000, 0.7127, 3.2432, 1.4117, 0.7891, 0.0000]

# Full Dataset Pixel Counts
CLASS_COUNTS = {
    0: 15846976,   # MISSING (1.37%)
    1: 404840132,  # CLEAR (35.07%)
    2: 88971956,   # CLOUD SHADOW (7.71%)
    3: 204398495,  # SEMI TRANSPARENT CLOUD (17.71%)
    4: 365647459,  # CLOUD (31.68%)
    5: 74515014    # UNDEFINED (6.46%)
}


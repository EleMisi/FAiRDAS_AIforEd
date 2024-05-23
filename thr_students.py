from collections import OrderedDict

# Define thresholds for IQR normalization
thresholds = {

    'exp0': OrderedDict(
        DIDI=0,
        score_absolute_error=2,
    ),

    'exp1': OrderedDict(
        DIDI=0.7,
        score_absolute_error=0.7,
    ),

    'exp2': OrderedDict(
        DIDI=0.5,
        score_absolute_error=0.5,
    ),

    'exp3': OrderedDict(
        DIDI=0.2,
        score_absolute_error=0.2,
    ),

}

from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import MeanEncoder
from feature_engine.encoding import CountFrequencyEncoder

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

def encoder():
    return {
        "OneHotEncoder": OneHotEncoder(),
        "MeanEncoder": MeanEncoder(),
        "CountFrequencyEncoder": CountFrequencyEncoder(
            encoding_method='frequency'
        ),
    }

def resampler(seed=0):
    return {
        "SMOTE": SMOTE(random_state=seed),
        "RandomUnderSampler": RandomUnderSampler(random_state=seed),
        "SMOTEENN": SMOTEENN(random_state=seed, n_jobs=-1)
    }










































# end of code

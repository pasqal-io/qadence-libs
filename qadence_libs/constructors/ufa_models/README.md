## A submodule to quickly make different kinds of QNNs

### Features:

Independently specify the feature map, ansatz, and observable configurations. The `build_qnn` function will automatically build the QNN based on the configurations provided.

### Feature Map:

- `num_features`: Number of features to be encoded.
- `basis_set`: Basis set to be used for encoding the features. Fourier or Chebyshev.
- `reupload_scaling`: Scaling strategy for reuploading the features. Constant Tower or Exponential.
- `feature_range`: Range of data that the input data is assumed to come from.
- `target_range`: Range of data that the encoder assumes as natural range.
- `multivariate_strategy`: Strategy to be used for encoding multiple features. Series or Parallel.
- `feature_map_strategy`: Strategy to be used for encoding the features. Digital, Analog, or Rydberg.
- `param_prefix`: Prefix to be used for the parameters of the feature map.
- `num_repeats`: Number of times each feature is reuploaded.
- `operation`: Operation to be used for encoding the features.
- `inputs`: Inputs to be used for encoding the features.

### Ansatz:

- `num_layers`: Number of layers in the ansatz.
- `ansatz_type`: Type of ansatz to be used. HEA or IIA.
- `ansatz_strategy`: Strategy to be used for encoding the features. Digital, SDAQC or Rydberg.
- `strategy_args`: Arguments to be passed to the strategy.
- `param_prefix`: Prefix to be used for the parameters of the ansatz.

### Observable:

- `detuning`: The detuning term in the observable.
- `detuning_strength`: Strength of the detuning term in the observable.

### Usage:

```python

from qadence.types import BasisSet, ReuploadScaling

from config import AnsatzConfig, FeatureMapConfig, ObservableConfig
from quantum_models import build_qnn

fm_config = FeatureMapConfig(
    num_features=1,
    basis_set=BasisSet.CHEBYSHEV,
    reupload_scaling=ReuploadScaling.TOWER,
    feature_range=(-1.2, 1.2),
    feature_map_strategy="digital",
    multivariate_strategy="series",
)

ansatz_config = AnsatzConfig(
    num_layers=2,
    ansatz_type="hea",
    ansatz_strategy="rydberg",
)

obs_config = ObservableConfig(detuning_strength="z")

f = build_qnn_model(
    register=3,
    fm_config=fm_config,
    ansatz_config=ansatz_config,
    observable_config=obs_config,
)

```

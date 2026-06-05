import torch.nn as nn
from typing import Dict
import ProtoAI.Networks.Modules as mod
import logging
import json
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

MODULES = {
    "MOE":                  mod.MixtureOfExperts.factory,
    "RESIDUAL":             mod.FC.Residual.factory,
    "TRANSFORMER":          mod.FC.Transformer.factory,
    "CLASSIFIER":           mod.FC.Classifier.factory,
    "MLP":                  mod.FC.MLP.factory,
    "CONVOLUTION":          mod.Conv.Plain.factory,
    "TOTRANSFORMER":        mod.Conv.ToTransformer.factory,
    "TOLINEAR":             mod.Conv.ToLinear.factory,
    "RESIDUALCONVOLUTION":  mod.Conv.Residual.factory,
    "CONVOLUTIONALENCODER": mod.Conv.Encoder.factory,
    "CONVOLUTIONALDECODER": mod.Conv.Decoder.factory
}

def BuildFromConfig(Architecture: Dict[str, Dict[str, int]] | str) -> Callable:
        # Create an empty list to store modules
        network = []

        # Arguments must be a dictionary or a path to the JSON file acting as the config
        if  isinstance(Architecture, dict) or isinstance(Architecture, str) or isinstance(Architecture, Path):
            # Load in JSON config if file path is passed
            if isinstance(Architecture, str) or isinstance(Architecture, Path):
                with open(Architecture, 'r') as f:
                    config = json.load(f)
            else:
                config = Architecture

            log.info(f"Building the Neural Network.")
            for key in config.keys():
                log.info(f"Adding {key} layer.")
                network.append(MODULES[key.upper()](config[key]))
        
        return nn.Sequential(*network)



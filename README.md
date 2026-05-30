<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/a4b1cfaf-424e-4bf9-84e4-afd8622f42f4" />

ProtoAI is a wrapper library built on top of PyTorch that is meant to enable developers with little to no AI experience create and train state-of-the-art prototypes on a variety of tasks. This library is meant to serve as a rapid prototyping API for neural networks and supervised learning tasks. Neural Networks can be manually trained for Autoencoding, Reinforcement Learning, and Unsupervised learning tasks, however the current module focuses on Computer Vision.

Example Configuration File:

```
{
    "Convolution": {
        "in_channels": 3,
        "layer_channels": [64, 128, 256],
        "kernel": [7, 5, 3],
        "stride": [1, 1, 1],
        "padding": [3, 2, 1],
        "pooling": {
            "kernel": 2,
            "stride": 2
        }
    },
    "totransformer": {
        "n_channels": 256
    },
    "transformer": {
        "i_dim": 256,
        "hidden_dim": 256,
        "num_heads": 4,
        "num_layers": 4
    },
    "MoE": {
        "num_experts": 5,
        "top_k": 2,
        "NestedMoE": {
            "d_model": 256,
            "d_hidden": 256
        }
    },
    "classifier": {
        "i_dim": 256,
        "hidden_dim": 256,
        "num_layers": 2,
        "o_dim": 4
    }
}
```

In this file, the script will create a Convolutional layer, whose features are then sent through a transformer architecture. The outputs are then sent through a mixture of nested experts algorithm before finally making it's way through the classifier.
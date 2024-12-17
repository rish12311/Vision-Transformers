# Vision Transformer from Scratch

This is a simplified PyTorch implementation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

The above project has been developed by:

- Granth Bagadia - 2022A7PS0217H
- Harshit Juneja - 2021A7PS2946H
- Rishabh Goyal - 2021B1A72320H

## Usage

Dependencies:

- PyTorch 2.5.1
- torchvision 0.20.1

Run the below script to install the dependencies

```bash
pip install -r requirements.txt
```

You can find the implementation in the `vit.py` file. The main class is `ViTForImageClassification`, which contains the embedding layer, the transformer encoder, and the classification head. To train the model for 10 epochs with a batch size of 32, you can run:

```bash
python train.py --exp-name vit-with-10-epochs --epochs 10 --batch-size 32
```

Please have a look at the `train.py` file for more details.

## Results

The model was trained on the CIFAR-10 dataset. The model config was used to train the model:

```python
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}
```

The model is much smaller than the original ViT models from the paper as we just want to illustrate how the model works rather than achieving state-of-the-art performance.

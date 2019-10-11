# SynAV

SynAV uses SocialGAN as a backbone to synthesize realistic edge-case driving scenarios for training self-driving cars. The realism is evaluated by the critic network of SocialGAN. The generator and discriminator models are trained with Wasserstein GAN loss with gradient penalty different from original SocialGAN. The original SocialGAN paper can be found here:  **<a href="https://arxiv.org/abs/1803.10892">Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks</a>**

<!--<div align='center'>
  <img src='images/model.png' width='1000px'>
</div>

## Setup
All code was developed and tested on Ubuntu 16.04 with Python 3.5 and PyTorch 0.4.

You can setup a virtual environment to run the code like this:

```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies
echo $PWD > env/lib/python3.5/site-packages/sgan.pth  # Add current directory to python path
# Work for a while ...
deactivate  # Exit virtual environment
```

## Pretrained Models
You can download pretrained models by running the script `bash scripts/download_models.sh`. This will download the following models:

- `sgan-models/<dataset_name>_<pred_len>.pt`: Contains 10 pretrained models for all five datasets. These models correspond to SGAN-20V-20 in Table 1.
- `sgan-p-models/<dataset_name>_<pred_len>.pt`: Contains 10 pretrained models for all five datasets. These models correspond to SGAN-20VP-20 in Table 1.

Please refer to [Model Zoo](MODEL_ZOO.md) for results.

## Running Models
You can use the script `scripts/evaluate_model.py` to easily run any of the pretrained models on any of the datsets. For example you can replicate the Table 1 results for all datasets for SGAN-20V-20 like this:

```bash
python scripts/evaluate_model.py \
  --model_path models/sgan-models
```

## Training new models
Instructions for training new models can be [found here](TRAINING.md).-->

# SynAV

One of the biggest challenges of self-driving cars is their inability to predict human behavior reliably, be it of other drivers or pedestrians or bikes. In order to make sure that a car can perform in dangerous situations, it is important to include such situations within their training environment or test environment. An easy way would be to simulate difficult and potentially dangerous human maneuvers in a multi-agent RL system. But that can be computationally expensive and yet a simple approximation of the real-world challenge. Another alternative is to collect a lot of mileage on real roads so that we encounter enough dangerous scenarios. But this can be risky even with a supervising driver. SynAV takes a data-driven approach for synthesizing realistic adversarial traffic from normal traffic data. 

## Model

SynAV uses SocialGAN as a backbone to synthesize realistic edge-case driving scenarios for training self-driving cars. SocialGAN uses a pooling mechanism to capture pairwise agent interactions. The realism is evaluated by the critic network of SocialGAN. The generator and discriminator models are trained with Wasserstein GAN loss with gradient penalty different from original SocialGAN. The original SocialGAN paper can be found here:  **<a href="https://arxiv.org/abs/1803.10892">Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks</a>**

<!--<div align='center'>
  <img src='images/model.png' width='1000px'>
</div>-->


## Setup
All code was developed and tested on Ubuntu 24.2 with Python 3.5 and PyTorch 0.4.

You can setup a virtual environment to run the code like this:

```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies
echo $PWD > env/lib/python3.5/site-packages/sgan.pth  # Add current directory to python path
# Work for a while ...
deactivate  # Exit virtual environment
```
## Data
The model is trained with car trajectories generated using CARLA simulator by Nick Rhinehart. The link to the original dataset is given here  **<a href="https://sites.google.com/view/precog">PRECOG CARLA dataset</a>**
The data is hosted by the author in a publicly shared google drive folder. Create a folder called `datasets/town01` in CarGAN directory and download the data in this location. To access the data, add it to your Google Drive and then access it directly from google drive on mac or use the GDrive API on a linux machine. Once downloaded the `town01` folder should have three subfolders `train`, `val` and `test`.

## Pretrained Models
You can download pretrained models by running the script `bash scripts/download_models.sh`. This will download the model `gan_test_with_model.pt`. You can use the script `scripts/evaluate_model.py` to easily run the pretrained model on the CARLA test dataset. The expected results are ade:0.51 and fde:1.28. The values are in meters.

```bash
python scripts/evaluate_model.py \
  --model_path models
```

## Training new models
Instructions for training new models can be [found here](TRAINING.md).

## Training with Wasserstein GAN Loss with Gradient penalty
The code for training SynAV with WGAN Loss with GP and creating adversarial trajectories are located in the directory `scripts/wgan`. In order to run the training for the discriminator alone use `train_only_d.py`. Run the following code for finding the adversarial trajectory.

```bash
streamlit run scripts/wgan/create_adversarial.py
```

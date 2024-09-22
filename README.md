# Generative Adversarial Network on MNIST dataset

In this project, I used the MNIST dataset, which is a dataset of handwritten digits
which is mostly use for learning about classification algorithms in both Machine
Learning and Deep Learning applications. My framework on choice for this project is
PyTorch, an open source Deep Learning library. 

Here, I design a very simple GAN network forf generating synthetic data of
handwritten digits. I designed the discriminator and the generator in different
classed. And trained the architecture with varying hyperparameters for varying
number of epochs. We know that GANs are very sensitive to hyperparameter tuning
and hence we must be cognizant of what param values we use.

## Architectures of the models
### Generator Network Architecture
![Generator Neural Network](/Images/Generator_Architecture.jpg)

### Discriminator Network Architecture
![Discriminator Neural Network](/Images/Discriminator_Architecture.jpg)

## Results Comparision on Tensorboard Visualization
### GAN Results
Trained for 50 epochs using mean of 0.1307 and std of 0.3081 of the MNIST dataset
for noise generation.
![Results_GAN1](/Images/GAN_1.jpg)

After this is changed the mean and std to 0.5 and 0.5 respectively
and still trained the network for 50 epochs

![Results_GAN_2](/Images/GAN_2.jpg)

After this, I thought of training the network for longer, so I did it for 100 epochs

![Results_GAN_3](/Images/GAN_3.jpg)


As we can see with both of these models, the CNN architecture performs better for
the same data.

## Observations
We can see that with longer training the network starts to generate fake images
that are very closely resembling actual digits. Some numbers we can clearly make
out is '3', '1', '0', '9', '5', etc. We can use a deeper network to get even better
results. For future works, we can implement DC-GAN (a Deep CNN based GAN) to
generate even improved fake data.

## Usage

* Open the folder in and code editor and then navigate to the src directory
```commandline
cd Generative-Adversarial-Network-on-MNIST-dataset/src/
```
* Run the code / train the model
```commandline
python Generative_Adversarial_Networks.py
```

* Visualize results on tensorboard
```commandline
pip install tb-nightly # if pkg not already installed
```
```
tensorboard --logdir runs
``` 

## Dependencies
1. Python version - Python 3.11.3
2. PyTorch Version - torch Version: 2.0.1
3. NVIDIA-SMI GPU - CUDA Version: 12.0 
4. Tensorboard

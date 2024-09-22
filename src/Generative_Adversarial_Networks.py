# ======== A simple Generative Adversarial neural network using PyTorch ========= #

# ======= Imports ========= #
import torch  # entire PyTorch library
import torch.nn as nn  # Taking all neural network models
import torch.optim as optim  # optimization algos, SGD, Adam etc
import torchvision
# import torch.nn.functional as F  # functions without params, ReLU, Tanh etc
from torch.utils.data import DataLoader  # Easier dataset mgmt, create mini batches etc
import torchvision.datasets as datasets  # Standard datasets from PyTorch
import torchvision.transforms as transforms  # transformation to perform on datasets
import torchsummary  # try pip install torchsummary if you don't have this package
from torch.utils.tensorboard import SummaryWriter  # To print to Tensorboard to visualize it


# ============ Building the Discriminator =============== #
class Discriminator(nn.Module):  # inherit from the nn Module
    def __init__(self, img_dim):  # Input dimensions of image from the data
        super().__init__()  # to give the access to methods and properties of a parent and/or sibling class
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),  # Leaky ReLU is a good choice for GANS, slope is 0.1
            nn.Linear(128, 1),  # Outputs a single value, either real or fake
            nn.Sigmoid(),  # to ensure that our result is between 0 and 1
        )

    # define a forward pass for the discriminator
    def forward(self, X):
        return self.disc(X)


# ============ Building the Generator =============== #
class Generator(nn.Module):  # inherit from the nn Module
    def __init__(self, z_dim, img_dim):  # dimension of the latent noise for input and img dimensions
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # To ensure that output is b/w -1 & 1
        )

    # define a forward pass for the discriminator
    def forward(self, X):
        return self.generator(X)


# ============ Setting the Hyperparameters =============== #
#  (very sensitive to GANs)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4  # Best learning rate as per trial and error for Adam Optimizer
z_dim = 64  # can try other dims like 128, 256
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 100

# ============ Initializing our discriminator and generator =============== #

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# ============ Preprocessing and loading of data =============== #
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # mean and STD of MNIST dataset
)

dataset = datasets.MNIST(root="../dataset", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)  # optimizer for discriminator
opt_gen = optim.Adam(gen.parameters(), lr=lr)  # optimizer for generator
criterion = nn.BCELoss()  # the loss function
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")  # Create instance of Summary Writer one fake and one real
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0  # For Tensorboard visualization

# ============ Training of GAN =============== #
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)  # reshape the real images. Keep the number of features, flatten the rest
        batch_size = real.shape[0]

        # ============ Training of Discriminator =============== # max (log(D(real))) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)  # Gaussian with mean 0 and STD 1
        fake = gen(noise)  # Generate some fake images, nothing but G(z)
        disc_real = disc(real).view(-1)  # discriminator on real and then reshaping i.e., flattening

        # ends up returning the min (-log(D(real))) ==> which is nothing but max(log(D(real))
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)  # discriminator on fake, i.e., D(G(z)) and then reshaping i.e., flattening
        """
        disc_fake = disc(fake.detach()).view(-1)  # Using fake.detach() will ensure that when we run the back pass,
        we won't end up clearing the gradient computations in the Loss_D.backward() step. This is only so that we can
        use the fake data to train our Generator without needing to calculate it again, saving computation
        """

        # ends up returning the min (-log(1 - D(G(z)))) ==> Which is nothing but max(log(1 - D(G(z))))
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Personal Note: Look up torch.nn.BCELoss to understand this further

        Loss_D = (lossD_real + lossD_fake) / 2  # Total loss. Why are we taking an average??

        disc.zero_grad()
        # Loss_D.backward()  # this step will clear all the gradients calculated in the forward pass form cache
        Loss_D.backward(retain_graph=True)  # This argument will ensure that the gradients are retained, this can be
        # done instead of fake.detach() as shown above. Basically do either this or that
        opt_disc.step()

        # ============ Training of Generator =============== # min log(1 - D(G(z)))
        # But the above loss leads to saturating gradients problem
        # instead we do max(log(D(G(z))) We use the fake, i.e G(z) calculated above instead of calculating it again
        output = disc(fake).view(-1)  # discriminator on fake, i.e., D(G(z)) and then reshaping i.e., flattening
        Loss_G = criterion(output, torch.ones_like(output))
        # ends up returning the min (-log(D(G(z)))) ==> Which is nothing but max(log(D(G(z))))

        gen.zero_grad()
        Loss_G.backward()
        opt_gen.step()

        # ============ Code for visualization on Tensorboard =============== #
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}]\n"
                f"Loss D: {Loss_D:.4f}, Loss G: {Loss_G:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )

                step += 1

# print(gen)
# torchsummary.summary(gen, (64,))
# print("\n")
# print(disc)
# torchsummary.summary(disc, (784,))


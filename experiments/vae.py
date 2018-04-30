import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F

# VAE model
class VAE(nn.Module):
    def __init__(self, z_dim=32, use_cuda=True):
        super().__init__()

        self.z_dim = z_dim
        self.use_cuda = use_cuda

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 6, stride=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.encoder_mean = nn.Sequential(
            nn.Linear(32 * 8 * 11, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.z_dim)
        )

        self.encoder_logvar = nn.Sequential(
            nn.Linear(32 * 8 * 11, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.z_dim),
            nn.Softplus()
        )

        self.z_to_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32 * 8 * 11),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 32, 5, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 3, 6, stride=3),
            nn.LeakyReLU()
        )

    def reparameterize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""

        eps = torch.randn(mu.size(0), mu.size(1))
        if not self.use_cuda:
            eps = Variable(eps)
        else:
            eps = Variable(eps.cuda())
        z = mu + eps * torch.exp(0.5 * log_var)  # 0.5 to convert var to std
        return z

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)

        mu = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)

        return mu, logvar

    def decode(self, z):
        #h = self.relu1(self.deconv1(z))
        h = self.z_to_decoder(z)
        h = h.view(z.size(0), -1, 8, 11)

        x = self.decoder(h)
        x = x[:, :, 3:123, 1:161]
        #x = F.sigmoid(x)

        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def sample(self, z):
        return self.decode(z)

def vae_loss_fn(recon_x, x, mu, logvar):
    batch_size = x.size(0)

    MSE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    print('  mu min = %g, max=%g' % (mu.min(), mu.max()))
    print('  logvar min = %g' % logvar.min())
    print('  logvar max = %g' % logvar.max())

    beta = 1
    KLD = torch.sum(0.5 * beta * (mu ** 2 + torch.exp(logvar) - logvar - 1))

    # NOTE: possible to multiply KL by a factor

    return (MSE + KLD) / batch_size

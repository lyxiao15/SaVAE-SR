import torch 
import torch.nn as nn 
import torch.distributions as dist


class Encoder(nn.Module):
    def __init__(self, x_dim=120, z_dim=3, eps=1e-4):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.eps = eps
        self.shared_layer = nn.Sequential(
            nn.Linear(self.x_dim, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True)
        )
        self.z_mean_layer = nn.Sequential(
            nn.Linear(100, self.z_dim)
        )
        self.z_std_layer = nn.Sequential(
            nn.Linear(100, self.z_dim),
            nn.Softplus()
        )

    def forward(self, x):
        out = self.shared_layer(x)
        z_mean = self.z_mean_layer(out)
        z_std = self.z_std_layer(out) + self.eps
        return z_mean, z_std 
    
    def penalty(self):
        penalty = 0.0 
        for p in self.shared_layer.parameters():
            penalty += torch.sum(p ** 2)
        return penalty

class Generator(nn.Module):
    def __init__(self, z_dim=3, x_dim=120, eps=1e-4):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.eps = eps

        self.shared_layer = nn.Sequential(
            nn.Linear(self.z_dim, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(inplace=True)
        )
        self.x_mean_layer = nn.Sequential(
            nn.Linear(100, self.x_dim)
        )
        self.x_std_layer = nn.Sequential(
            nn.Linear(100, self.x_dim),
            nn.Softplus()
        )

    def forward(self, z):
        out = self.shared_layer(z)
        x_mean = self.x_mean_layer(out)
        x_std = self.x_std_layer(out)
        return x_mean, x_std
    
    def penalty(self):
        penalty = 0.0 
        for p in self.shared_layer.parameters():
            penalty += torch.sum(p ** 2)
        return penalty


class VAE(nn.Module):
    def __init__(self, x_dim=120, z_dim=3, eps=1e-4, L=1):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.eps = eps
        self.n_samples = L
        self.encoder = Encoder(x_dim=self.x_dim, z_dim=self.z_dim, eps=self.eps)
        self.generator = Generator(z_dim=self.z_dim, x_dim=self.x_dim, eps=self.eps)

    def forward(self, x):
        z_mean, z_std = self.encoder(x)
        z = self.reparameterization(z_mean, z_std)
        p_z_x = dist.Normal(z_mean, z_std)
        x_mean, x_std = self.generator(z)
        p_x_z = dist.Normal(x_mean, x_std)
        return p_x_z, p_z_x, z          

    def reparameterization(self, mu, std):
        if self.train:
            ones = torch.zeros(mu.size()).type_as(mu)
            zeros = torch.ones(mu.size()).type_as(mu)
            noise = dist.Normal(ones, zeros).sample((1,))
            z = noise * std.unsqueeze(0) + mu.unsqueeze(0)
        else:
            z = dist.Normal(mu, std).sample((self.n_samples, ))
        return z



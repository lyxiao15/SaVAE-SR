from typing import *
from scipy.special import erf
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from .utils import Loop, TestLoop
from .network import VAE
from .data import KPISeries, KpiFrameDataset, KpiFrameDataLoader
import sklearn
from .loss import AE, KL0, KL
from .missing_value_imputation import mcmc_missing_imputation

class IntroVAE:
    """
    """
    def __init__(self, max_epoch=100, batch_size=256, latent_dims=3, window_size=120, cuda: bool=False, margin=15.0, print_fn=print):

        self.print_fn = print_fn
        self.window_size = window_size
        self.latent_dims = latent_dims
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.cuda = cuda
        self.model = VAE(x_dim=self.window_size, z_dim=self.latent_dims)

        if self.cuda:
            self.model = self.model.cuda()
            self.alpha = torch.tensor(1.0).cuda()
            self.beta = torch.tensor(1.0).cuda()
            self.margin = torch.tensor(margin).cuda()
            self.z_prior_dist = dist.Normal(
                torch.from_numpy(np.zeros((self.latent_dims,), np.float32)).cuda(),
                torch.from_numpy(np.ones((self.latent_dims,), np.float32)).cuda())
            self.x_prior_dist = dist.Normal(
                torch.from_numpy(np.zeros((self.window_size,), np.float32)).cuda(),
                torch.from_numpy(np.ones((self.window_size,), np.float32)).cuda())
        else:
            self.alpha = torch.tensor(1.0)
            self.beta = torch.tensor(1.0)
            self.margin = torch.tensor(margin)
            self.z_prior_dist = dist.Normal(
                torch.from_numpy(np.zeros((self.latent_dims,), np.float32)),
                torch.from_numpy(np.ones((self.latent_dims,), np.float32)))
            self.x_prior_dist = dist.Normal(
                torch.from_numpy(np.zeros((self.window_size,), np.float32)).cuda(),
                torch.from_numpy(np.ones((self.window_size,), np.float32)).cuda())

    def reparameterization(self, mu, sd, latent=False):
        if latent:
            noise = self.z_prior_dist.sample((self.batch_size,))
        else:
            noise = self.x_prior_dist.sample((self.batch_size,))
        return noise * sd + mu
        


    def fit(self, kpi: KPISeries, valid_kpi: KPISeries=None):
        self.model.train()
        with Loop(max_epochs=self.max_epoch, use_cuda=self.cuda, disp_epoch_freq=5, print_fn=self.print_fn).with_context() as loop:
            # optimizer
            optimizerD = SGD(self.model.encoder.parameters(), lr=0.0002)
            optimizerG = SGD(self.model.generator.parameters(), lr=0.0005)
            lr_scheduler_D = StepLR(optimizerD, step_size=10, gamma=0.75)
            lr_scheduler_G = StepLR(optimizerG, step_size=10, gamma=0.75)

            train_kpiframe_dataset = KpiFrameDataset(kpi, frame_size=self.window_size)
            train_dataloader = KpiFrameDataLoader(train_kpiframe_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            if valid_kpi is not None:
                valid_kpiframe_dataset = KpiFrameDataset(valid_kpi,frame_size=self.window_size, missing_injection_rate=0.)
                valid_dataloader = KpiFrameDataLoader(valid_kpiframe_dataset, batch_size=256, shuffle=True)
            else:
                valid_dataloader = None

            # 1.has zp and zr modules
            for epoch in loop.iter_epochs():
                for _, batch_data in loop.iter_steps(train_dataloader):
                    observe_x, observe_normal = batch_data

                    optimizerD.zero_grad()
                    optimizerG.zero_grad()
                    #|--------------------Update Encoder--------------------|
                    p_x_z, p_z_x, observe_z = self.model(observe_x)

                    # Fake sample generated by the noise zp
                    zp = self.z_prior_dist.sample((self.batch_size,))
                    xp_mean, xp_std = self.model.generator(zp)
                    xp = self.reparameterization(xp_mean, xp_std)

                    # Fake sample generated by the noise z
                    xr_mean, xr_std = self.model.generator(observe_z.detach())
                    xr = self.reparameterization(xr_mean, xr_std)


                    zp_mean, zp_std = self.model.encoder(xp.detach())
                    p_zp_xp = dist.Normal(zp_mean, zp_std)
                    observe_zp = p_zp_xp.sample()
            
                    zr_mean, zr_std = self.model.encoder(xr.detach())
                    p_zr_xr = dist.Normal(zr_mean, zr_std)
                    observe_zr = p_zr_xr.sample()

                    # Compute the loss of encoder 
                    kl0 = KL(observe_z, observe_normal, p_z_x, self.z_prior_dist)
                    kl1 = KL0(observe_zr, observe_normal, p_zr_xr, self.z_prior_dist)
                    kl2 = KL0(observe_zp, observe_normal, p_zp_xp, self.z_prior_dist)
                    
                    L1 = F.relu(self.margin - kl1)
                    L2 = F.relu(self.margin - kl2)
                    rec_loss = AE(observe_x, observe_normal, p_x_z)
                
                    DLoss = self.alpha * (L1 + L2) + self.beta * rec_loss + kl0 + self.model.encoder.penalty()*0.001
                    DLoss.backward(retain_graph=True)
                    clip_grad_norm_(self.model.encoder.parameters(), max_norm=10.)
                    optimizerD.step()


                    #|--------------------Update Generator--------------------|
                    zr_mean, zr_std = self.model.encoder(xr)
                    zp_mean, zp_std = self.model.encoder(xp)
                    p_zr_xr = dist.Normal(zr_mean, zr_std)
                    p_zp_xp = dist.Normal(zp_mean, zp_std)

                    observe_zr = p_zr_xr.sample((self.batch_size,))
                    observe_zp = p_zp_xp.sample((self.batch_size,))
                                        
                    kl1 = KL0(observe_zr, observe_normal, p_zr_xr, self.z_prior_dist)
                    kl2 = KL0(observe_zp, observe_normal, p_zp_xp, self.z_prior_dist)
                    
                    GLoss = self.alpha * (kl1 + kl2) + self.model.generator.penalty() * 0.001

                    GLoss.backward()
                    clip_grad_norm_(self.model.generator.parameters(), max_norm=10.)
                    optimizerG.step()

                    loss = (GLoss + DLoss).item()
                    loop.submit_metric("train_loss", loss)
                lr_scheduler_D.step()
                lr_scheduler_G.step()

                if valid_kpi is not None:
                    with torch.no_grad():
                        for _, batch_data in loop.iter_steps(valid_dataloader):
                            observe_x, observe_normal = batch_data  

                            p_x_z, p_z_x, observe_z = self.model(observe_x)
                            # Fake sample generated by the noise zp
                            zp = self.z_prior_dist.sample((self.batch_size,))
                            xp_mean, xp_std = self.model.generator(zp)
                            xp = self.reparameterization(xp_mean, xp_std)
                            # Fake sample generated by the noise z
                            xr = p_x_z.sample()

                            zp_mean, zp_std = self.model.encoder(xp)
                            p_zp_xp = dist.Normal(zp_mean, zp_std)
                            observe_zp = p_zp_xp.sample()
                    
                            zr_mean, zr_std = self.model.encoder(xr)
                            p_zr_xr = dist.Normal(zr_mean, zr_std)
                            observe_zr = p_zr_xr.sample()

                            # Compute the loss of encoder 
                            kl0 = KL0(observe_z, observe_normal, p_z_x, self.z_prior_dist)
                            kl1 = KL0(observe_zr, observe_normal, p_zr_xr, self.z_prior_dist)
                            kl2 = KL0(observe_zp, observe_normal, p_zp_xp, self.z_prior_dist)
                            
                            L1 = F.relu(self.margin - kl1)
                            L2 = F.relu(self.margin - kl2)
                            rec_loss = AE(observe_x, observe_normal, p_x_z)
                        
                            DLoss = self.alpha * (L1 + L2) + self.beta * rec_loss + kl0 + self.model.encoder.penalty()*0.001
                            GLoss = self.alpha * (kl1 + kl2) + self.beta * rec_loss + self.model.generator.penalty() * 0.001

                            loss = (GLoss + DLoss).item()
                            loop.submit_metric("valid_loss", loss)

    def predict(self, kpi: KPISeries, return_statistics=False, indicator_name="indicator"):
        """
        :param kpi:
        :param return_statistics:
        :param indicator_name:
            default "indicator": Reconstructed probability
            "indicator_prior": E_q(z|x)[log p(x|z) * p(z) / q(z|x)]
            "indicator_erf": erf(abs(x - x_mean) / x_std * scale_factor)
        :return:
        """
        with torch.no_grad():
            with TestLoop(use_cuda=self.cuda, print_fn=self.print_fn).with_context() as loop:
                test_kpiframe_dataset = KpiFrameDataset(kpi, frame_size=self.window_size, missing_injection_rate=0.0)
                test_dataloader = KpiFrameDataLoader(test_kpiframe_dataset, batch_size=32, shuffle=False,
                                                     drop_last=False)
                self.model.eval()
                for _, batch_data in loop.iter_steps(test_dataloader):
                    observe_x, observe_normal = batch_data  # type: Variable, Variable
                    observe_x = mcmc_missing_imputation(observe_normal=observe_normal,
                                                        vae=self.model,
                                                        n_iteration=10,
                                                        x=observe_x)
                    p_x_z, p_z_x, observe_z = self.model(observe_x)
                    # Fake sample generated by the noise zp
                    zp = self.z_prior_dist.sample((self.batch_size,))
                    xp_mean, xp_std = self.model.generator(zp)
                    xp = self.reparameterization(xp_mean, xp_std)
                    # Fake sample generated by the noise z
                    xr = p_x_z.sample()

                    zp_mean, zp_std = self.model.encoder(xp)
                    p_zp_xp = dist.Normal(zp_mean, zp_std)
                    observe_zp = p_zp_xp.sample()
            
                    zr_mean, zr_std = self.model.encoder(xr)
                    p_zr_xr = dist.Normal(zr_mean, zr_std)
                    observe_zr = p_zr_xr.sample()

                    # Compute the loss of encoder 
                    kl0 = KL0(observe_z, observe_normal, p_z_x, self.z_prior_dist)
                    kl1 = KL0(observe_zr, observe_normal, p_zr_xr, self.z_prior_dist)
                    kl2 = KL0(observe_zp, observe_normal, p_zp_xp, self.z_prior_dist)
                    
                    L1 = F.relu(self.margin - kl1)
                    L2 = F.relu(self.margin - kl2)
                    rec_loss = AE(observe_x, observe_normal, p_x_z)
                
                    DLoss = self.alpha * (L1 + L2) + self.beta * rec_loss + kl0 + self.model.encoder.penalty()*0.001
                    GLoss = self.alpha * (kl1 + kl2) + self.beta * rec_loss + self.model.generator.penalty() * 0.001

                    loss = (GLoss + DLoss).item()
                    
                    loop.submit_metric("test_loss", loss)
                    log_p_xz = p_x_z.log_prob(observe_x).data.cpu().numpy()

                    log_p_x = log_p_xz * np.sum(
                        torch.exp(self.z_prior_dist.log_prob(observe_z) - p_z_x.log_prob(observe_z)).cpu().numpy(),
                        axis=-1, keepdims=True)

                    indicator_erf = erf((torch.abs(observe_x - p_x_z.mean) / p_x_z.stddev).cpu().numpy() * 0.1589967)

                    loop.submit_data("indicator", -np.mean(log_p_xz[:, :, -1], axis=0))
                    loop.submit_data("indicator_prior", -np.mean(log_p_x[:, :, -1], axis=0))
                    loop.submit_data("indicator_erf", np.mean(indicator_erf[:, :, -1], axis=0))

                    loop.submit_data("x_mean", np.mean(
                        p_x_z.mean.data.cpu().numpy()[:, :, -1], axis=0))
                    loop.submit_data("x_std", np.mean(
                        p_x_z.stddev.data.cpu().numpy()[:, :, -1], axis=0))

                    indicator = np.concatenate(loop.get_data_by_name(indicator_name))
                    x_mean = np.concatenate(loop.get_data_by_name("x_mean"))
                    x_std = np.concatenate(loop.get_data_by_name("x_std"))

            indicator = np.concatenate([np.ones(shape=self.window_size - 1) * np.min(indicator), indicator])
            if return_statistics:
                return indicator, x_mean, x_std
            else:
                return indicator

    def save(self, path, description):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), path + '/' + description + '_' + 'model.pth')

    def load(self, path):
        self.model.load_state_dict(torch.load(path + 'model.pth'))
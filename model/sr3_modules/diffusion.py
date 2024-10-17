import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from gudhi.wasserstein import wasserstein_distance
# from torch_topological.nn import CubicalComplex
# from torch_topological.nn import WassersteinDistance
import gudhi as gd
import cv2
import time
import threading
from queue import Queue
from skimage.filters import frangi


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

# class TopologicalWDLoss(nn.Module):
#     def __init__(self):
#         super(TopologicalWDLoss, self).__init__()
#         self.cubical = CubicalComplex()
#         self.wd_loss_func = WassersteinDistance(q=2)

#     def forward(self, hr, sr):
#         loss = 0.0

#         for x, y in zip(hr, sr):
#             x = x.squeeze()
#             y = y.squeeze()

#             per_hr = self.cubical(x)[0]
#             per_sr = self.cubical(y)[0]
#             wd_loss = self.wd_loss_func(per_hr, per_sr)
#             loss += wd_loss
#         return loss

# class TopologicalWDLoss(nn.Module):
#     def __init__(self):
#         super(TopologicalWDLoss, self).__init__()
#         self.cubical = CubicalComplex()
#         self.wd_loss_func = WassersteinDistance(q=2)

#     def process_batch_item(self, x, y, result_queue):
#         x = x.squeeze()
#         y = y.squeeze()

#         per_hr = self.cubical(x)[0]
#         per_sr = self.cubical(y)[0]
#         wd_loss = self.wd_loss_func(per_hr, per_sr)
#         result_queue.put(wd_loss)

#     def forward(self, hr, sr):
#         threads = []
#         result_queue = Queue()

#         for x, y in zip(hr, sr):
#             thread = threading.Thread(target=self.process_batch_item, args=(x, y, result_queue))
#             thread.start()
#             threads.append(thread)

#         for thread in threads:
#             thread.join()

#         loss = 0.0
#         while not result_queue.empty():
#             loss += result_queue.get()

#         return loss

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def match_cofaces_with_gudhi(image_data, cofaces):
    height, width = image_data.shape
    result = []

    for dim, pairs in enumerate(cofaces[0]):
        for birth, death in pairs:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            death_y, death_x = np.unravel_index(death, (height, width))
            pers = (1.00-image_data.ravel()[birth], 1.00-image_data.ravel()[death])
            result.append((dim, pers,((birth_y, birth_x), (death_y, death_x))))

    for dim, births in enumerate(cofaces[1]):
        for birth in births:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            pers = (1.00-image_data.ravel()[birth], 1.0)
            result.append((dim, pers, ((birth_y, birth_x), None)))

    return result

def persistent_homology(image_data, output_file_name="output", device=torch.device('cuda:0')):
    """Computes and visualizes the persistent homology for the given image data."""
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=1 - image_data.flatten()
    )
    time1 = time.time()
    cc.persistence()
    time2 = time.time()
    cofaces = cc.cofaces_of_persistence_pairs()
    time3 = time.time()
    result = match_cofaces_with_gudhi(image_data=image_data, cofaces=cofaces)
    time4 = time.time()

    frangi_img = frangi(1-image_data)
    time5 = time.time()
    new_result = []

    for dim, (birth, death) , coordinates in result:
        if dim == 1:
            continue
        distance = np.abs(birth - death) / np.sqrt(2)
        weight = distance * frangi_img[coordinates[0][0], coordinates[0][1]]

        weight_threshold = 0.01
        if weight > weight_threshold:
            new_result.append([birth, death])

    time6 = time.time()

    gudhi_time = time2 - time1
    cofaces_time = time3 - time2
    match_time = time4 - time3
    frangi_time = time5 - time4
    filter_time = time6 - time5

    return np.array(new_result), [gudhi_time, cofaces_time, match_time, frangi_time, filter_time]

class WassersteinDistanceLoss(nn.Module):
    def __init__(self):
        super(WassersteinDistanceLoss, self).__init__()

    def process_image(self, hr_img, sr_img, index, losses, per_time, wd_time, gudhi_time, cofaces_time, match_time, frangi_time, filter_time):
        time1 = time.time()
        hr_connect, time_res_hr = persistent_homology(hr_img[0])
        sr_connect, time_res_sr = persistent_homology(sr_img[0])
        time2 = time.time()
        losses[index] = wasserstein_distance(hr_connect, sr_connect)
        time3 = time.time()

        gudhi_time[index] = time_res_hr[0]
        cofaces_time[index] = time_res_hr[1]
        match_time[index] = time_res_hr[2]
        frangi_time[index] = time_res_hr[3]
        filter_time[index] = time_res_hr[4]

        per_time[index] = time2 - time1
        wd_time[index] = time3 - time2

    def forward(self, hr, sr):
        num_images = hr.shape[0]
        losses = [None] * num_images
        per_time = [None] * num_images
        wd_time = [None] * num_images
        gudhi_time = [None] * num_images
        cofaces_time = [None] * num_images
        match_time = [None] * num_images
        frangi_time = [None] * num_images
        filter_time = [None] * num_images


        threads = []

        for i in range(num_images):
            hr_img = hr[i].detach().float().cpu().numpy()
            sr_img = sr[i].detach().float().cpu().numpy()
            thread = threading.Thread(target=self.process_image, args=(hr_img, sr_img, i, losses, per_time, wd_time, gudhi_time, cofaces_time, match_time, frangi_time, filter_time))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        print("gudhi time: ", sum(gudhi_time)/num_images)
        print("cofaces time: ", sum(cofaces_time)/num_images)
        print("match time: ", sum(match_time)/num_images)
        print("frangi time: ", sum(frangi_time)/num_images)
        print("filter time: ", sum(filter_time)/num_images)
        print("total persistence time: ", sum(per_time)/num_images)
        print("gudhi wd time: ", sum(wd_time)/num_images)
        return torch.tensor(sum(losses) / num_images)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        loss_name='wd',
        under_step_wd_loss=500,
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.loss_name = loss_name
        self.under_step_wd_loss = under_step_wd_loss
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        elif self.loss_type == 'wd':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
            # self.loss_func2 = TopologicalWDLoss().to(device)
            self.loss_wd = WassersteinDistanceLoss().to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return img

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        self.origin_loss = self.loss_func(noise, x_recon)
        if self.loss_name == 'wd':
            self.wd_loss = torch.tensor(0.0).to(x_start.device)
            if t < self.under_step_wd_loss:
                denoise_img = self.predict_start_from_noise(x_noisy, t=t-1, noise=x_recon)
                self.wd_loss = self.loss_wd(x_in['HR'], denoise_img)
                # self.wd_loss = self.loss_func2(x_in['HR'], denoise_img)
            loss = self.origin_loss + self.wd_loss
            # print("wd loss: ", self.wd_loss)
            # print("origin loss: ", self.origin_loss)
            # print("total loss: ", loss)
        elif self.loss_name == 'original':
            loss = self.origin_loss
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

    def get_each_loss(self):
        if self.loss_name == 'wd':
            return self.origin_loss, self.wd_loss
        elif self.loss_name == 'original':
            return self.origin_loss, torch.tensor(0.0).to(self.origin_loss.device)

import cv2
import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch, torchvision
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange, repeat
import time
from ldm.util import ismap
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from PIL import ImageEnhance
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def get_cond(mode, path):
    example = dict()
    if mode == 'real' or mode == 'v1':
        c_ = Image.open(path)
        c = c_.resize((576, 384))
    elif mode == 'syn':
        c = Image.open(path)
    # bright = ImageEnhance.Brightness(c)
    # c = bright.enhance(factor=1.2)
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    c = rearrange(c, '1 c h w -> 1 h w c')
    c = 2. * c - 1.
    # c += 0.1
    c = c.to(torch.device("cuda"))
    example["ll_image"] = c

    return example

def run(model, cond, custom_steps, num_sample):
    log = dict()
    eta = 0.  # 1.
    shape = cond['ll_image'].shape
    # height, width = cond['ll_image'].shape[1:3]
    split_input = False
    # if split_input:
    #     ks = 64 # 64
    #     stride = 32  # 32
    #     vqf = 4
    #     model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
    #                                 "vqf": vqf,
    #                                 "patch_distributed_vq": True,
    #                                 "tie_braker": False,
    #                                 "clip_max_weight": 0.5,
    #                                 "clip_min_weight": 0.01,
    #                                 "clip_max_tie_weight": 0.5,
    #                                 "clip_min_tie_weight": 0.01}

    # else:
    #     if hasattr(model, "split_input_params"):
    #         delattr(model, "split_input_params")

    if shape is not None:
        x_T = torch.randn(num_sample, shape[3], shape[1]//4, shape[2]//4).to(model.device)
        # x_T = repeat(x_T, '1 c h w -> b c h w', b=num_sample)

    # z, c = model.get_input(cond, model.cond_stage_key, force_c_encode=True)

    c = cond[model.cond_stage_key]
    if len(c.shape) == 3:
        c = c[..., None]
    c = rearrange(c, 'b h w c -> b c h w')
    c = c.to(memory_format=torch.contiguous_format).float()
    c = model.get_learned_conditioning(c)
    c = repeat(c, '1 c h w -> b c h w', b=num_sample)

    # if shape is not None:
    #     z = torch.randn(custom_shape)
    #     print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None
    with model.ema_scope("Plotting"):
        t0 = time.time()
        if opt.sample == 'dpm':
            ddim = DPMSolverSampler(model)
        elif opt.sample == 'plms':
            ddim = PLMSSampler(model)
        elif opt.sample == 'ddim':
            ddim = DDIMSampler(model)

        samples, intermediates = ddim.sample(custom_steps, batch_size=num_sample, shape=c.shape[1:], conditioning=c,
                                              eta=eta, x0=z0, verbose=False, x_T=x_T)
        t1 = time.time()

    # x_sample = model.decode_first_stage(samples, force_not_quantize=True)
    # log["sample"] = x_sample
    x_sample = model.decode_first_stage(samples)
    log["sample"] = x_sample
    log['time'] = t1 -t0

    return log


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        type=str,
        default='0',
        nargs="?",
        help="gpu id",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='v2-syn',
        nargs="?",
        help="choose the dataset:v1 or v2-real or v2-syn",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--nrun",
        type=int,
        default=10,
        help="the path from content root of the ckpt",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default='dpm',
        help="choose the sample way",
    )
    parser.add_argument(
        "--saveimg",
        action='store_false',
        help="save image",
    )
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.id

    if 'v2' in opt.dataset:
        if 'real' in opt.dataset:
            mode = 'real'
            indir = 'data/LOL-v2/Real_captured/Test/Low'
            ckpt = 'ckpt/real_2615.ckpt'
            cfg = 'configs/latent-diffusion/lolv2_real.yaml'

        elif 'syn' in opt.dataset:
            mode = 'syn'
            indir = 'data/LOL-v2/Synthetic/Test/Low'
            ckpt = 'ckpt/syn_4384.ckpt'
            cfg = 'configs/latent-diffusion/lolv2_syn.yaml'

        img = os.listdir(indir)
        # img.sort(key=lambda x: int(x.split('.')[0]))
        img.sort()

    elif opt.dataset == 'v1':
        mode = 'v1'
        indir = 'data/LOL/LOLdataset/eval15/low'
        img = os.listdir(indir)
        img.sort(key=lambda x: int(x.split('.')[0]))
        # img.sort()
        ckpt = 'ckpt/v1_1172.ckpt'
        cfg = 'configs/latent-diffusion/lolv1.yaml'

    else:
        AssertionError
        
    outdir = 'outputs/' + mode
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt, map_location='cuda:0')["state_dict"], strict=False)
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    model = model.to(device)

    custom_steps = opt.steps
    psnr_best = []
    ssim_best = []
    psnr_avg = []
    ssim_avg = []
    names = []
    logs = list()
    t = 0

    for i in img:
        names.append(i)

        if mode == 'v1':
            img = os.path.join(indir, i)
            img_gt = img.replace('low', 'high')
            gt = Image.open(img_gt).resize((576, 384))
            gt = np.array(gt)

        elif mode == 'real':
            img = os.path.join(indir, i)
            img_gt = img.replace('Low','Normal')
            gt = Image.open(img_gt).resize((576, 384))
            gt = np.array(gt)

        elif mode == 'syn':
            img = os.path.join(indir, i)
            img_gt = img.replace('Low','Normal')
            gt = np.array(Image.open(img_gt))
            


        cond = get_cond(mode, img)
        log = run(model, cond, custom_steps, num_sample=opt.nrun)
        sample = torch.clamp(log["sample"], -1., 1.)

        img_list = [np.round(((sample + 1.) / 2. * 255).cpu().numpy().transpose(0, 2, 3, 1)[i]).astype('uint8')
                    for i in range(sample.shape[0])]
        psnr_list = [PSNR(img_list[i], gt) for i in range(len(img_list))]
        ssim_list = [SSIM(gt, img_list[i], multichannel=True, channel_axis=2) for i in range(len(img_list))]
        idx = np.argmax(psnr_list)
        psnr_best.append(psnr_list[idx])
        ssim_best.append(ssim_list[idx])

        if opt.nrun > 1:
            psnr_avg.append(sum(psnr_list) / len(psnr_list))
            ssim_avg.append(sum(ssim_list) / len(ssim_list)) 
        logs.append(img_list[idx])
        t += log['time']

        if opt.saveimg:
            imgname = os.path.split(img)[1].split('.')[0]
            file_name = imgname + f"_psnr:{psnr_best[-1]}_ssim:{ssim_best[-1]}" + '.png'
            Image.fromarray(img_list[idx]).save(os.path.join(outdir, file_name))


        # sample = logs["sample_noquant"]
        # sample = torch.clamp(sample, -1., 1.)
        # sample = (sample + 1.)/2. * 255
        # sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0]
        #
        # path = img.replace('low', 'high')
        # gt = np.array(Image.open(path))
        # gt = (np.clip(gt, 0, 1)*255).astype(np.uint8)  # np.clip(x,0,1):make the value in [0,1]
        # sample_ = (np.clip(sample, 0, 1)*255).astype(np.uint8)
        # sample_ = sample.astype(np.uint8)
        # psnr.append(PSNR(sample_, gt))


    psnr_best_avg = np.mean(psnr_best)
    ssim_best_avg = np.mean(ssim_best)

    psnr_avg_avg = sum(psnr_avg) / len(psnr_avg)
    ssim_avg_avg = sum(ssim_avg) / len(ssim_avg)
    print(f"select best: PSNR_avg->{psnr_best_avg}, SSIM_avg->{ssim_best_avg}; avg every img: PSNR-> {psnr_avg_avg}, SSIM->{ssim_avg_avg}  spend {t}")
    for k, name in enumerate(names):
        print(f"{name}: PSNR_best->{psnr_best[k]},SSIM->{ssim_best[k]}; PSNR_avg->{psnr_avg[k]}, SSIM->{ssim_avg[k]} ")






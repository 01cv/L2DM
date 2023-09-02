<h1 >L<sup>2</sup>DM: A Diffusion Model for Low-Light Image Enhancement</h1>

<p>
This paper presents L<sup>2</sup>DM, a novel framework for low-light image enhancement using diffusion models.
<img src=https://github.com/Yore0/L2DM/blob/master/pic1.png>
  
</p>

## Requirements
A suitable conda environment can be created and activated with:
```
conda env create -f environment.yaml
conda activate ldm
```

## Data preparation
We used the LOL and LOL-v2 datasets, where the LOL-v2 dataset is divided into two parts: real and synthetic. The dataset and model weights are placed in Baidu Cloud for downloading.
Dataset files should be placed inside the ```data\```

## Pretrained Models
We need ```3``` network checkpoints, which are Auto-encoder checkpoints, COCO pre-training weights, and dataset-correlated weights.
Once downloaded, put the ```model.ckpt``` to ```ckpt/vq-f4/model.ckpt```; ```epoch=000099.ckpt``` to ```ckpt/coco/epoch=000099.ckpt```; ```<lol_>.ckpt``` to ```ckpt/<lol_>.ckpt```

## Training L<sup>2</sup>DM
In ```configs/latent-diffusion/``` we provide configs for training L<sup>2</sup>DM on the LOL, LOL-real, LOL-synthetic datasets. Training can be started by running
```CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,```

## Testing our results
Run```python d2l_ori.py --id 0 --dataset v1 --steps 20 --nrun 10 --sample dpm ```,<br>
```dataset``` are available in ```v1, v2-real, and v2-syn```.
<p>
  <img src=https://github.com/Yore0/L2DM/blob/master/pic2.png>
  <img src=https://github.com/Yore0/L2DM/blob/master/tab1.png>
</p>



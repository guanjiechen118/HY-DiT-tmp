# HunYuan-DiT with Skip-Branch

**The project is forked from [HunyuanDiT](https://github.com/tencent/HunyuanDiT)**

### 1. prepare your environments
We recommend CUDA versions 11.7 and 12.0+.

```shell
# 1. Prepare conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate HunyuanDiT

# 3. Install pip dependencies
python -m pip install -r requirements.txt

# 4. Install flash attention v2 for acceleration (requires CUDA 11.6 or above)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

### 2. Download Pretrained Models
To download the model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
mkdir ckpts
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.2 --local-dir ./ckpts
```

### 3. Download coco dataset for FID evaluation (optional)

I have prepared the code to download coco-validation. 
```
python coco/download_coco.py
```
Everythig will be done automatically

### 4. Where is the code of DeepCache
model module is at 
`./hydit/modules/models_cache.py`

### 5. Inference w/o cache
you can just following the example script at `./scripts/run_infer_test.sh`

Details can be found at [HunYuan-DiT](https://github.com/Tencent/HunyuanDiT/blob/main/README.md)

### 6. Inference with DeepCache

Two important parameters to introduce:
1. `cache-step`: This decide how many steps you wanna to cache. If cache-step=5, it will calculate transformer blocks only 1 time every 5 sample steps, while other 4 sample steps is calculated with feature cache.
2. `cache-at-branch`: This decide which skip-branch will be used in cache steps. If cache-at-branch=1, only the first and the last transformer block will be calculated in cache-steps, while other transformer blocks will be cached. 
3. The larger the `cache-step` is, the faster the inference will be, with larger quality loss. And the larger the `cache-at-branch` is, the more transformer blocks will be caculated in cache steps, the better the generation quality is, at the cost of inference speed.
4. To enable DeepCache, add `--deepcache` to your scritps.

I provide example to generate 10,000 image with caption of coco dataset with 1 or 8 GPUs.

`./scripts/infer-coco-1-gpu.sh`

`./scripts/infer-coco-8-gpus.sh`
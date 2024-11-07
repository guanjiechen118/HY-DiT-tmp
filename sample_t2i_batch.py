from pathlib import Path

from loguru import logger

from mllm.dialoggen_demo import DialogGen
from hydit.config import get_args
from hydit.inference import End2End


def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)

    # Try to enhance prompt
    enhancer = None

    return args, gen, enhancer


if __name__ == "__main__":
    args, gen, enhancer = inferencer()
    enhanced_prompt = None
    caption_path = args.caption_path
    # default as "./coco/data/captions/caption.json"
    import json
    import time
    import os
    import tqdm
    with open(caption_path, "r") as file:
        captions = json.load(file)
    # Run inference
    logger.info("Generating images...")
    height, width = args.image_size
    start_time = time.time()
    caption_id = 0
    pbar = tqdm.tqdm(len(captions))
    for caption in captions:
        index = args.index
        if not caption_id % 8 == index:
            caption_id += 1
            continue
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{caption_id}.png')
        if os.path.exists(save_path):
            caption_id += 1
            continue
        results = gen.predict(caption,
                            height=height,
                            width=width,
                            seed=args.seed,
                            enhanced_prompt=enhanced_prompt,
                            negative_prompt=args.negative,
                            infer_steps=args.infer_steps,
                            guidance_scale=args.cfg_scale,
                            batch_size=args.batch_size,
                            src_size_cond=args.size_cond,
                            use_style_cond=args.use_style_cond,
                            )
        image = results['images'][0]

        # Save images
        
        image.save(save_path)
        logger.info(f"Save to {save_path}")
        
        caption_id += 1
        pbar.update(1)
        pbar.set_description(f'{caption_id}/{len(captions)}')

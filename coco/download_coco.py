from datasets import load_dataset
import os
from PIL import Image
import json

if __name__ == '__main__':
    dataset = load_dataset("sayakpaul/coco-30-val-2014", cache_dir='./coco/data')
    # Specify the output directory
    output_dir = "./coco/data"
    images_dir = os.path.join(output_dir, "images")
    captions_dir = os.path.join(output_dir, "captions")

    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(captions_dir, exist_ok=True)

    captions = []
    # Iterate over the dataset and save images and captions
    for idx, sample in enumerate(dataset['train']):
        # Save the image
        image = sample['image']
        image_path = os.path.join(images_dir, f"image_{idx}.jpg")
        image.save(image_path)

        # Save the caption
        caption = sample['caption']
        captions.append(caption)

        # Optionally print progress
        if idx % 1000 == 0:
            print(f"Saved {idx} images and captions")

    
    caption_path = os.path.join(captions_dir, f"caption.json")
    with open(caption_path, "w") as file:
        json.dump(captions, file, indent=4)


    print("Finished saving all images and captions.")

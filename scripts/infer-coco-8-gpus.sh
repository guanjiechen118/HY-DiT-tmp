#!/bin/bash

task_name=deepcache-skip

for index in {0..7}
do
  CUDA_VISIBLE_DEVICES=$(($index)) python sample_t2i_batch.py  --no-enhance --infer-steps 100  --save-dir "./sample_results/${task_name}" --index $index \
   --deepcache --cache-step 4 --cache-at-branch 2 --caption-path ./coco/data/captions/caption.json \ 
   >> "logs/${task_name}_${index}.log" 2>&1 &
done

wait

# if don't need deepcache, remove :
# --deepcache --cache-step xx --cache-at-branch xx
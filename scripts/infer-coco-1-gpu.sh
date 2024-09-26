gpu_num=1
srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpu_num} -n1 --ntasks-per-node=1 --job-name=fvd_eval --kill-on-bad-exit=1 --quotatype=auto \
python sample_t2i_single.py  --no-enhance --infer-steps 100 --save-dir ./sample_results/test --index 0 --deepcache --cache-step 2 --cache-at-branch 1 --caption-path ./coco/data/captions/caption.json

# if need deepcache, add following command to the end of the script
# --deepcache --cache-step 2 --cache-at-branch 1
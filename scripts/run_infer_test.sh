gpu_num=1
srun --partition=MoE --mpi=pmi2 --gres=gpu:${gpu_num} -n1 --ntasks-per-node=1 --job-name=fvd_eval --kill-on-bad-exit=1 --quotatype=auto \
python sample_t2i.py --prompt "渔舟唱晚"  --no-enhance --infer-steps 100 --image-size 1024 1024
# if need deepcache, add following command to the end of the script
# --deepcache --cache-step 2 --cache-at-branch 1
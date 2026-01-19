#paths=("arcdetriomphe" "corner" "drone" "fountain" "mountain" "patio" "patio_high" "spot" "statue" "train" "train_station" "tree")
dataset_name="on_the_go_ours_v3_w_v_dinov2"
#root="/data2/wangxu/code/desplat-main/data/on-the-go"
root="/data2/wangxu/code/desplat-main/data/on-the-go"
export CUDA_VISIBLE_DEVICES=1
conda activate robustsplat


paths=("corner" "fountain" "mountain" "patio_high" "spot")
for dataset in "${paths[@]}"
do
    #python ./on_the_go_split.py --base_dir ${root}/${dataset}
    #python train.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 8 --eval --port 22625 -d depths --filtered_masks my_masks2
    # time python train_v3.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 8 --eval --port 22625 -d depths --densify_grad_threshold 0.00012
    time python train_v3_w_voxel.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 8 --eval --port 22625 -d depths --densify_grad_threshold 0.00012
    python render.py --model_path output/${dataset_name}/${dataset}
    python metrics.py --model_path output/${dataset_name}/${dataset}
done

# paths=("patio")
# for dataset in "${paths[@]}"
# do
# #     #python ./on_the_go_split.py --base_dir ${root}/${dataset}
# #     time python train_v3.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 4 --eval --port 26625 -d depths
#     time python train_v3.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 4 --eval --port 22625 -d depths --densify_grad_threshold 0.00012 #--densify_from_iter 500
#     python render.py --model_path output/${dataset_name}/${dataset}
#     python metrics.py --model_path output/${dataset_name}/${dataset}
# done

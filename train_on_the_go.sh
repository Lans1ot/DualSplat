#paths=("arcdetriomphe" "corner" "drone" "fountain" "mountain" "patio" "patio_high" "spot" "statue" "train" "train_station" "tree")
dataset_name="on_the_go_ours_time"
#root="/data2/wangxu/code/desplat-main/data/on-the-go"
root="/data2/wangxu/code/desplat-main/data/on-the-go"
export CUDA_VISIBLE_DEVICES=0


paths=("corner" "fountain" "mountain" "patio_high" "spot")
#paths=("drone" "statue" "train" "train_station" "tree")
for dataset in "${paths[@]}"
do
    #python ./on_the_go_split.py --base_dir ${root}/${dataset}
    #python train.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 8 --eval --port 22625 -d depths --filtered_masks my_masks2
    time python train_depths_try2.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 8 --eval --port 22625 -d depths
    python render.py --model_path output/${dataset_name}/${dataset}
    python metrics.py --model_path output/${dataset_name}/${dataset}
done

# # #paths=("patio" "arcdetriomphe")
# paths=("patio")
# #root="/data/wangxu/data/on-the-go"
# for dataset in "${paths[@]}"
# do
#     #python ./on_the_go_split.py --base_dir ${root}/${dataset}
#     time python train.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 4 --eval --port 26625 -d depths
#     # python render.py --model_path output/${dataset_name}/${dataset}
#     # python metrics.py --model_path output/${dataset_name}/${dataset}
# done

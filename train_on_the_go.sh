#paths=("arcdetriomphe" "corner" "drone" "fountain" "mountain" "patio" "patio_high" "spot" "statue" "train" "train_station" "tree")
dataset_name="on_the_go_2"
#root="/data2/wangxu/code/desplat-main/data/on-the-go"
root="/data2/wangxu/code/desplat-main/data/on-the-go"
export CUDA_VISIBLE_DEVICES=0


paths=("corner" "fountain" "mountain" "patio_high" "spot")
for dataset in "${paths[@]}"
do
    #python ./on_the_go_split.py --base_dir ${root}/${dataset}
    python train2.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 8 --eval --port 22625
    python render.py --model_path output/${dataset_name}/${dataset}
    python metrics.py --model_path output/${dataset_name}/${dataset}
done

paths=("patio")
#root="/data/wangxu/data/on-the-go"
for dataset in "${paths[@]}"
do
    #python ./on_the_go_split.py --base_dir ${root}/${dataset}
    python train2.py  --source_path ${root}/${dataset} --model_path output/${dataset_name}/${dataset} --resolution 4 --eval --port 26625 
    python render.py --model_path output/${dataset_name}/${dataset}
    python metrics.py --model_path output/${dataset_name}/${dataset}
done

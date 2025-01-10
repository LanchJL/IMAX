ulimit -n 65536
export CUDA_VISIBLE_DEVICES=1
python -u train.py \
--clip_arch ./clip_modules/ViT-L-14.pt \
--dataset_path ../dataset/CZSL/cgqa \
--save_path ./save_dir/cgqa \
--yml_path ./config/clip/cgqa.yml \
--num_workers 2 \
--seed 0 \
--adapter
export CUDA_VISIBLE_DEVICES=1
python -u train.py \
--clip_arch ./clip_modules/ViT-L-14.pt \
--dataset_path ../dataset/CZSL/ut-zap50k \
--save_path ./save_dir/ut-zappos \
--yml_path ./config/clip/ut-zappos.yml \
--num_workers 4 \
--seed 0 \
--adapter
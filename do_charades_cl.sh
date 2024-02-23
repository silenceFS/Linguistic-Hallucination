collection=charades
visual_feature=i3d_rgb_lgi
clip_scale_w=0.5
frame_scale_w=0.5
device_ids=$1
exp_id=$2
consistency_weight=$3
hallucination_weight=$4
# training

python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --dset_name $collection --exp_id $exp_id \
                    --device_ids $device_ids --consistency_weight $consistency_weight \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --support_set_number 1  --curriculum \
                    --hallucination_weight $hallucination_weight


python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --dset_name $collection --exp_id $exp_id \
                    --device_ids $device_ids --consistency_weight $consistency_weight \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --support_set_number 2 --curriculum \
                    --hallucination_weight $hallucination_weight

python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --dset_name $collection --exp_id $exp_id \
                    --device_ids $device_ids --consistency_weight $consistency_weight \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --support_set_number 3 --curriculum \
                    --hallucination_weight $hallucination_weight



                    # --train_support \
                    # --hidden_size 1024 --lr 1e-3 \
                    # --load_support_model

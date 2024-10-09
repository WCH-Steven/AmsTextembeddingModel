deepspeed --num_nodes=1 --num_gpus=8 run.py \
--model_name_or_path /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/save_models/sph_325_98_mean_sl64_uninstruct_t006_bs1024_epoch2 \
--overwrite_output_dir true \
--negatives_cross_device true \
--fp16 true \
--logging_strategy steps \
--logging_steps 100 \
--save_strategy epoch \
--output_dir /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/save_models/sph_325_98_mean_sl64_uninstruct_t006_bs1024_epoch5 \
--seq_max_len 64 \
--per_device_train_batch_size 128 \
--data_format knlin \
--use_inbatch_neg true \
--use_instruct false \
--sentence_pooling_method mean \
--remove_unused_columns false \
--temperature 0.06 \
--num_train_epochs 3 \
--max_example_num_per_dataset 20000000 \
--train_data /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/commodity.json\
??\
/cephfs/group/teg-openrecom-openrc/knlin/wxvideo/feed_100w/online_test/文本标签数据对2024汇总_500w_178w.jsonl \
--deepspeed /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/baai_general_embedding/finetune/ds_config.json




# deepspeed --num_nodes=1 --num_gpus=8 run.py \
# --model_name_or_path /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/bge-large-zh-v1.5 \
# --overwrite_output_dir true \
# --negatives_cross_device true \
# --fp16 true \
# --logging_strategy steps \
# --logging_steps 100 \
# --save_strategy epoch \
# --output_dir /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/save_models/sph_325_98_mean_sl64_instruct_t006_bs1024_epoch1 \
# --seq_max_len 64 \
# --per_device_train_batch_size 128 \
# --data_format knlin \
# --use_inbatch_neg true \
# --use_instruct true \
# --sentence_pooling_method mean \
# --remove_unused_columns false \
# --temperature 0.06 \
# --num_train_epochs 1 \
# --max_example_num_per_dataset 20000000 \
# --train_data /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/commodity.json\
# ??\
# /cephfs/group/teg-openrecom-openrc/knlin/wxvideo/feed_100w/online_test/文本标签数据对2024汇总_500w_178w.jsonl \
# --deepspeed /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/baai_general_embedding/finetune/ds_config.json




# deepspeed --num_nodes=1 --num_gpus=8 run.py \
# --model_name_or_path /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/bge-large-zh-v1.5 \
# --overwrite_output_dir true \
# --negatives_cross_device true \
# --fp16 true \
# --logging_strategy steps \
# --logging_steps 100 \
# --save_strategy epoch \
# --output_dir /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/save_models/sph_325_98_cls_sl64_instruct_t006_bs1024_epoch1 \
# --seq_max_len 64 \
# --per_device_train_batch_size 128 \
# --data_format knlin \
# --use_inbatch_neg true \
# --use_instruct true \
# --sentence_pooling_method cls \
# --remove_unused_columns false \
# --temperature 0.06 \
# --num_train_epochs 1 \
# --max_example_num_per_dataset 20000000 \
# --train_data /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/commodity.json\
# ??\
# /cephfs/group/teg-openrecom-openrc/knlin/wxvideo/feed_100w/online_test/文本标签数据对2024汇总_500w_178w.jsonl \
# --deepspeed /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/baai_general_embedding/finetune/ds_config.json





# deepspeed --num_nodes=1 --num_gpus=8 run.py \
# --model_name_or_path /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/bge-large-zh-v1.5 \
# --overwrite_output_dir true \
# --negatives_cross_device true \
# --fp16 true \
# --logging_strategy steps \
# --logging_steps 100 \
# --save_strategy epoch \
# --output_dir /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/save_models/sph_325_98_mean_sl64_uninstruct_t006_bs1024_epoch2 \
# --seq_max_len 64 \
# --per_device_train_batch_size 128 \
# --data_format knlin \
# --use_inbatch_neg true \
# --use_instruct false \
# --sentence_pooling_method mean \
# --remove_unused_columns false \
# --temperature 0.06 \
# --num_train_epochs 2 \
# --max_example_num_per_dataset 20000000 \
# --train_data /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/commodity.json\
# ??\
# /cephfs/group/teg-openrecom-openrc/knlin/wxvideo/feed_100w/online_test/文本标签数据对2024汇总_500w_178w.jsonl \
# --deepspeed /cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/baai_general_embedding/finetune/ds_config.json

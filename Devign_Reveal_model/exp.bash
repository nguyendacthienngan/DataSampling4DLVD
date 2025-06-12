# N=5
# (
# for thing in {10..19}; do
#     ((i=i%N)); ((i++==0)) && wait
#     python main.py --dataset msr --sampling origin --input_dir reveal_model_data/msr_data/origin/data_split_"$thing" --node_tag node_features --graph_tag graph --label_tag targets --feature_size 100 --data_split "$thing"  --model_type ggnn --batch_size 256 &
# done
# )

# N=5
# (
# for thing in {0..19}; do
#     ((i=i%N)); ((i++==0)) && wait
#     python main.py --dataset reveal --sampling oss --input_dir reveal_model_data/reveal_data/actual_ros/data_split_"$thing" --node_tag node_features --graph_tag graph --label_tag targets --feature_size 100 --data_split "$thing"  --model_type ggnn &
# done
# )
python main.py --dataset msr --sampling msr --input_dir reveal_model_data/msr_data/origin/data_split_0 --node_tag node_features --graph_tag graph --label_tag targets --feature_size 100 --data_split 0  --model_type ggnn --batch_size 256


!python main.py --dataset msr --sampling msr --input_dir /kaggle/input/processed-dataset-of-datasampling4dlvd/reveal_model_bigvul_dataset --output_dir /kaggle/working/output --processed_dir /kaggle/working/processed --node_tag node_features --graph_tag graph --label_tag targets --feature_size 100 --data_split 0  --model_type ggnn --batch_size 256
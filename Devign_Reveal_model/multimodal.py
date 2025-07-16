import argparse
from cmath import log
import os
import pickle
import sys

import numpy as np
import torch
# from torch.nn import BCELoss
from torch.optim import Adam
import torch.nn as nn

from data_loader.dataset import DataSet
# from modules.model import DevignModel, GGNNSum
from modules.model import CombinedModel  # thay vì chỉ DevignModel, GGNNSum

from my_trainer import my_train, my_evaluate_metrics
from utils import tally_param, debug
import logging

import csv
import pandas as pd

def save_result_to_csv(output_path, result_dict):
    file_exists = os.path.isfile(output_path)
    with open(output_path, mode='a', newline='') as csvfile:
        fieldnames = ['dataset', 'model_type', 'fusion_type', 'sampling', 'split',
                      'accuracy', 'precision', 'recall', 'f1', 'auc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result_dict)

if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_type', type=str, help='Type of the fusion (devign/ggnn)',
                        choices=['gmu', 'concat', 'crossattn'], default='gmu')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--sampling', type=str, required=True, help='sampling method')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--output_dir', type=str, required=True, help='Output Directory of the parser')
    parser.add_argument('--processed_dir', type=str, required=True, help='Input Directory of the processed data')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')
    parser.add_argument('--data_split', type=str, default='1')
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)

    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    model_dir = os.path.join(f'{args.output_dir}_{args.dataset}_result', f'{args.model_type}_model', args.sampling, args.data_split)
    print('out dir ', model_dir)

    processed_dir = args.processed_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        

    # if args.feature_size > args.graph_embed_size:
    #     print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
    #           'Setting graph embedding size to feature size', file=sys.stderr)
    #     args.graph_embed_size = args.feature_size

    input_dir = args.input_dir # Readonly (Kaggle Input)
    processed_data_path = os.path.join(processed_dir, 'processed.bin')
    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        logging.info('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
        logging.info(f'{len(dataset.train_examples)}, {len(dataset.valid_examples)}, {len(dataset.test_examples)}')
        
        if args.feature_size != dataset.feature_size:
            print(f"[INFO] Overriding --feature_size {args.feature_size} → dataset.feature_size = {dataset.feature_size}")
            args.feature_size = dataset.feature_size
    else:
        debug('generate new dataset')
        logging.info('generate new dataset')
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_GGNNinput.json'),
                          valid_src=os.path.join(input_dir, 'test_GGNNinput.json'),
                          test_src=os.path.join(input_dir, 'test_GGNNinput.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        # file = open(processed_data_path, 'wb')
        # pickle.dump(dataset, file)
        # file.close()
        with open(processed_data_path, 'wb') as f:
            pickle.dump(dataset, f)
        debug(f'processed file dump to {processed_data_path}')
        logging.info(f'processed file dump to {processed_data_path}')
        args.feature_size = dataset.feature_size
    # assert args.feature_size == dataset.feature_size, \
    #     'Dataset contains different feature vector than argument feature size. ' \
    #     'Either change the feature vector size in argument, or provide different dataset.'
        
    # Ensure graph_embed_size >= feature_size
    if args.feature_size > args.graph_embed_size:
        print(f"[WARN] graph_embed_size ({args.graph_embed_size}) < feature_size ({args.feature_size}). Adjusting.")
        args.graph_embed_size = args.feature_size

    debug('model: CombinedModel (Devign + CodeBERT)')
    logging.info('model: CombinedModel (Devign + CodeBERT)')
    model = CombinedModel(
        fusion_type=args.fusion_type,  # hoặc 'concat', 'crossattn' nếu bạn muốn thử nghiệm
        graph_dim=args.graph_embed_size,
        seq_dim=768,  # CodeBERT có output dim là 768
        fusion_dim=128,
        feature_size=dataset.feature_size,
        max_edge_type=dataset.max_edge_type
    )

    debug('Total Parameters : %d' % tally_param(model))
    logging.info('Total Parameters : %d' % tally_param(model))
    # model.cuda()
    model.to(args.device)
    # loss_function = BCELoss(reduction='sum')
    # loss_function = BCELoss(reduction='mean')
    loss_function = nn.CrossEntropyLoss()

    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    # train(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
    #       loss_function=loss_function, optimizer=optim,
    #       save_path=model_dir + '/GGNNSumModel', max_patience=100, log_every=None, device=args.device)
    my_train(model=model, epochs=50, dataset=dataset, loss_function=loss_function, optimizer=optim,
             save_path=model_dir + '/Model', device=args.device)
    # Đánh giá sau khi train
    model.eval()
    valid_batch_len = dataset.initialize_valid_batch()
    acc, pr, rc, f1, auc_score = my_evaluate_metrics(model, loss_function, valid_batch_len, dataset, device=args.device)

    print(f"Validation Result:\n"
        f"  Accuracy:  {acc:.4f}\n"
        f"  Precision: {pr:.4f}\n"
        f"  Recall:    {rc:.4f}\n"
        f"  F1 Score:  {f1:.4f}\n"
        f"  AUC:       {auc_score:.4f}")

    logging.info(f"Validation Metrics:\n"
                f"  Accuracy:  {acc:.4f}\n"
                f"  Precision: {pr:.4f}\n"
                f"  Recall:    {rc:.4f}\n"
                f"  F1 Score:  {f1:.4f}\n"
                f"  AUC:       {auc_score:.4f}")

    result_dict = {
        'dataset': args.dataset,
        'model_type': args.model_type,
        'fusion_type': args.fusion_type if hasattr(args, 'fusion_type') else 'na',
        'sampling': args.sampling,
        'split': args.data_split,
        'accuracy': acc,
        'precision': pr,
        'recall': rc,
        'f1': f1,
        'auc': auc_score
    }

    result_csv_path = os.path.join(args.output_dir, f"{args.dataset}_results_summary.csv")
    save_result_to_csv(result_csv_path, result_dict)

    # test_batch_len = dataset.initialize_test_batch()
    # acc, pr, rc, f1, auc_score = my_evaluate_metrics(model, loss_function, test_batch_len, dataset, device=args.device)
    # print("TEST RESULT:", acc, pr, rc, f1, auc_score)




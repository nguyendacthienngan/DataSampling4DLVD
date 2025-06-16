import argparse
from cmath import log
import os
import pickle
import sys
import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm.auto import tqdm
from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
import warnings
from torch import nn
from data_loader.batch_graph import GGNNBatchGraph
import copy
import numpy
from dgl.nn.pytorch.explain import GNNExplainer
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
parser.add_argument('--model_dir', type=str, required=True, help='Model Directory of the parser')
parser.add_argument('--output_dir', type=str, required=True, help='Output Directory of the parser')
args = parser.parse_args()


warnings.filterwarnings("ignore")

# change to proceed dataset dir as the input dataset for interpretation
input_dir = args.input_dir
model_dir = args.model_dir

processed_data_path = os.path.join(input_dir, 'processed.bin')
dataset = pickle.load(open(processed_data_path, 'rb'))

dataset.batch_size = 1
dataset.initialize_test_batch()
model = GGNNSum(input_dim=dataset.feature_size, output_dim=200,num_steps=6, max_edge_types=dataset.max_edge_type)

model_path = os.path.join(model_dir, 'Model_ep_49.bin')


# load the trained-model to interpretation
model.load_state_dict(torch.load(model_path))
model.to('cuda:0')


# use when explain model with Sampling_R
class GGNNSum_single(nn.Module):
    def __init__(self, GGNNSum):
        super(GGNNSum_single, self).__init__()
        self.net = GGNNSum

    def forward(self, graph, feat, eweight=None):
        batch_graph = GGNNBatchGraph()
        # graph must be on CPU before this step
        cpu_graph = graph.cpu()
        batch_graph.add_subgraph(copy.deepcopy(cpu_graph))

        # move everything to CUDA
        batch_graph.graph = batch_graph.graph.to('cuda:0')
        feat = feat.to('cuda:0')

        outputs = self.net(batch_graph,device='cuda:0')
        # return torch.tensor([[1-outputs, outputs]])
        return torch.stack([1 - outputs, outputs], dim=1).to(outputs.device)



# switch between sampling_L and R
exp_model = GGNNSum_single(model)

gnnexplainer = GNNExplainer(exp_model,num_hops=1,log =False)
TP_explaination_dict = {}
total_test_item = dataset.initialize_test_batch()
for index in tqdm(range(total_test_item)):
    target = dataset.test_examples[index].target
    if target == 1:
        graph = dataset.test_examples[index].graph
        if graph.num_edges() > 10 and graph.num_nodes() > 10:
            # features = graph.ndata['features']
            # pred = exp_model(graph, features)
            # graph = graph.to('cuda:0')
            features = graph.ndata['features']
            pred = exp_model(graph, features)


#             print(pred)
            #             break
            if pred[0][1] > 0.5:
                #                 print(index,'tp')
                _ ,edge_mask = gnnexplainer.explain_graph(graph=graph,feat=features)
                top_10 = np.argpartition(edge_mask.detach().cpu().numpy(), -10)[-10:]
                node_list = []
                for x in top_10:
                    node_1,node_2 = graph.find_edges(x)
                    node_list.append(node_1.cpu().numpy()[0])
                    node_list.append(node_2.cpu().numpy()[0])
                TP_explaination_dict[index] = node_list

print(TP_explaination_dict)
print(total_test_item)

print(len(TP_explaination_dict))

# save the explaination results for further analysis
output_dir = args.output_dir
output_path = os.path.join(output_dir, 'msr_4x_split_0_hop_1.pkl')

with open(output_path, 'wb') as fp:
    pickle.dump(TP_explaination_dict, fp)

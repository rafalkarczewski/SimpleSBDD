import sys

import torch.nn as nn

from torch_geometric.nn import aggr
# sys.path.append('../../hsbdd/Modular-Flows-Neurips-2022/')
import egnn_clean as eg

mean_aggr = aggr.MeanAggregation()


class CoMPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        self.layer = eg.EGNN(in_node_nf=in_dim, hidden_nf=hidden_dim,
                             out_node_nf=out_dim, n_layers=n_layers)

    def forward(self, data):
        data_inp = data.clone()
        edges = [data_inp.edge_index[0].long(), data_inp.edge_index[1].long()]
        h, pos = self.layer(data_inp.x.float(), data_inp.pos.float(), edges, None)
        return mean_aggr(pos, data_inp.batch)

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops

# From: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/general_conv.py

class GeneralConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True, normalize_adj=False, self_msg='concat', **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize_adj
        self.self_msg = self_msg

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if self.self_msg == 'concat':
            self.weight_self = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.self_msg == 'concat':
            glorot(self.weight_self)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        """"""
        if self.self_msg == 'concat':
            x_self = torch.matmul(x, self.weight_self)
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        x_msg = self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_feature)
        if self.self_msg == 'none':
            return x_msg
        elif self.self_msg == 'add':
            return x_msg + x
        elif self.self_msg == 'concat':
            return x_msg + x_self
        else:
            raise ValueError('self_msg {} not defined'.format(self.self_msg))

    def message(self, x_j, norm, edge_feature):
        if edge_feature is None:
            return norm.view(-1, 1) * x_j if norm is not None else x_j
        else:
            return norm.view(-1, 1) * (
                x_j + edge_feature) if norm is not None else (x_j + edge_feature)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GeneralEdgeConvLayer(MessagePassing):
    r"""General GNN layer, with edge features
    """
    def __init__(self, in_channels, out_channels, edge_dim, improved=False,
                 cached=False, bias=True, normalize_adj=False, self_msg='concat', aggr='sum', msg_direction='single', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize_adj
        self.msg_direction = msg_direction
        self.self_msg = self_msg

        if self.msg_direction == 'single':
            self.linear_msg = nn.Linear(in_channels + edge_dim, out_channels,
                                        bias=False)
        else:
            self.linear_msg = nn.Linear(in_channels * 2 + edge_dim,
                                        out_channels, bias=False)
        if self.self_msg == 'concat':
            self.linear_self = nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        x_msg = self.propagate(edge_index, x=x, norm=norm,
                               edge_feature=edge_feature)

        if self.self_msg == 'concat':
            x_self = self.linear_self(x)
            return x_self + x_msg
        elif self.self_msg == 'add':
            return x + x_msg
        else:
            return x_msg

    def message(self, x_i, x_j, norm, edge_feature):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        else:
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        x_j = self.linear_msg(x_j)
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

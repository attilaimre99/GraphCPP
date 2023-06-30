import torch
import torch.nn.functional as F
from torch.nn import Sequential, BatchNorm1d
import torch_geometric as pyg
from graphcpp.generalconv import GeneralConvLayer, GeneralEdgeConvLayer as GraphGymEdgeConvLayer

from graphcpp.act import act_dict
from graphcpp.pooling import pooling_dict
from graphcpp.utils import init_weights

# General classes
class GeneralLayer(torch.nn.Module):
    '''General wrapper for layers'''
    def __init__(self, name, dim_in, dim_out, has_act=True, dropout=0.0, act='relu', **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = kwargs['has_l2norm']
        self.dropout = dropout
        self.layer = layer_dict[name](dim_in, dim_out, bias=not kwargs['has_bn'], **kwargs)
        layer_wrapper = []
        if kwargs['has_bn']:
            layer_wrapper.append(BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(torch.nn.Dropout(p=dropout))
        if has_act:
            layer_wrapper.append(act_dict[act])
        self.post_layer = Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm: batch = F.normalize(batch, p=2.0, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm: batch.x = F.normalize(batch.x, p=2.0, dim=1)
        return batch


class GeneralMultiLayer(torch.nn.Module):
    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None, final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None, num_layers=2, **kwargs):
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(GeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner, dim_inner, final_act=True, **kwargs))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.model = Sequential(*layers)

    def forward(self, batch):
        import operator
        pre_layer = operator.attrgetter('0')
        final_layer = operator.attrgetter('1')
        
        embedding = pre_layer(self.model)(batch)
        batch = final_layer(self.model)(embedding)
        # for name, module in self.model.named_modules():
            # print(name)
            # batch = module(batch)
        # print([name for name, module in self.named_modules()])
        # if isinstance(batch, torch.Tensor):
        #     batch = self.model(batch)
        # else:
        #     batch.x = self.model(batch.x)
        return batch, embedding
    
class GNNStackStage(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_layers, stage_type, layer_type, **kwargs):
        super(GNNStackStage, self).__init__()
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.has_l2norm = kwargs['has_l2norm']
        for i in range(num_layers):
            if stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GeneralLayer(name=layer_type, dim_in=d_in, dim_out=dim_out, has_act=True, **kwargs)
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if self.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif self.stage_type == 'skipconcat' and i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if self.has_l2norm:
            batch.x = F.normalize(batch.x, p=2.0, dim=-1)
        return batch

class GNNGraphHead(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_layers, pooling='add', **kwargs):
        super(GNNGraphHead, self).__init__()
        self.pooling_fun = pooling_dict[pooling]
        self.layer_post_mp = MLP(dim_in=dim_in, dim_out=dim_out, num_layers=num_layers, bias=True, **kwargs)

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y
    
    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb, inbetween_embedding = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        # return pred, label, inbetween_embedding # for tsne
        return pred, label

# GENERIC LAYERS

class Linear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class GCNConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNConv, self).__init__()
        self.model = pyg.nn.GCNConv(in_channels=dim_in, out_channels=dim_out, bias=bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class SAGEConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, conv_aggr='mean', **kwargs):
        super(SAGEConv, self).__init__()
        self.model = pyg.nn.SAGEConv(in_channels=dim_in, out_channels=dim_out, bias=bias, aggr=conv_aggr)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class GATConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, conv_dropout=0.0, **kwargs):
        super(GATConv, self).__init__()
        self.model = pyg.nn.GATConv(dim_in, dim_out, bias=bias, edge_dim=11, dropout=conv_dropout)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class GATv2Conv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, conv_dropout=0.0, **kwargs):
        super(GATv2Conv, self).__init__()
        self.model = pyg.nn.GATv2Conv(dim_in, dim_out, bias=bias, edge_dim=11, dropout=conv_dropout)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch
    
class TransformerConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, conv_dropout=0.0, n_heads=1, **kwargs):
        super(TransformerConv, self).__init__()
        self.model = Sequential([
            pyg.nn.TransformerConv(in_channels=dim_in, out_channels=dim_out, bias=bias, heads=n_heads, edge_dim=11, dropout=conv_dropout),
            Linear(dim_in=dim_in*n_heads, dim_out=dim_out)
        ])

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch

class CustomGeneralConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(CustomGeneralConv, self).__init__()
        self.model = GeneralConvLayer(in_channels=dim_in, out_channels=dim_out, bias=bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
    
class GINConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, act='relu', **kwargs):
        super(GINConv, self).__init__()
        gin_nn = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_out), act_dict[act], torch.nn.Linear(dim_out, dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

class GeneralEdgeConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, conv_aggr="sum", **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GraphGymEdgeConvLayer(dim_in, dim_out, bias=bias, edge_dim=11, aggr=conv_aggr)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, edge_feature=batch.edge_attr)
        return batch


class GeneralSampleEdgeConv(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, conv_aggr="sum", **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GraphGymEdgeConvLayer(dim_in, dim_out, bias=bias, edge_dim=11, aggr=conv_aggr)

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < 0.5 # keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_attr[edge_mask, :]
        batch.x = self.model(batch.x, edge_index, edge_feature=edge_feature)
        return batch

# FINAL MODEL

class GCN(torch.nn.Module):
    def __init__(self, layers_pre_mp, mp_layers, layers_post_mp, hidden_channels, stage_type, **kwargs):
        super(GCN, self).__init__()

        if layers_pre_mp > 0:
            self.pre_mp = GeneralMultiLayer(name='linear', num_layers=layers_pre_mp, dim_in=32, dim_out=hidden_channels, dim_inner=hidden_channels, final_act=True, **kwargs)
        if mp_layers > 0:
            self.mp = GNNStackStage(dim_in=hidden_channels, dim_out=hidden_channels, num_layers=mp_layers, stage_type=stage_type, **kwargs)

        self.post_mp = GNNGraphHead(hidden_channels, 1, layers_post_mp, **kwargs)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
    

layer_dict = {
    'linear': Linear,
    'mlp': MLP,
    'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    'gatv2conv': GATv2Conv,
    'transformerconv': GATv2Conv,
    'generalconv': CustomGeneralConv,
    'generaledgeconv': GeneralEdgeConv,
    'generalsampleedgeconv': GeneralSampleEdgeConv,
    'ginconv': GINConv,
}
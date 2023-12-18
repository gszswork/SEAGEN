import torch
from torch import nn
import numpy as np
import math

#from model.temporal_attention import TemporalAttentionLayer
from model.sub_graph_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20,
                          use_time_proj=True):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
        super(TimeEmbedding, self).__init__(node_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_time_features,
                                            embedding_dimension, device, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                          use_time_proj=True):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

        return source_embeddings


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_time_features,
                                             embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device


    def compute_embedding(self, source_nodes, destination_nodes, source_timestamps, destination_timestamps, n_neighbors=50,
                             time_diffs=None, use_time_proj=True):

        '''
        (1) Get the node features and time embeddings of all source nodes and destination nodes.
        (2) Get the neighbors of only the source nodes.
        (3) Calculate the node features and time embedding for those sources' neighbors.
        (4) Do the temporal graph attention (called  interaction sub-graph encoder in my paper) and return.
        '''
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        destination_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
        source_timestamps_torch = torch.unsqueeze(torch.from_numpy(source_timestamps).double().to(self.device), dim=1)
        destination_timestamps_torch = torch.unsqueeze(torch.from_numpy(destination_timestamps).double().to(self.device), dim=1)

        source_node_time_embedding = self.time_encoder(torch.zeros_like(source_timestamps_torch)).float()
        destination_node_time_embedding = self.time_encoder(torch.zeros_like(destination_timestamps_torch)).float()
        source_node_features = self.node_features[source_nodes_torch, :]
        destination_node_features = self.node_features[destination_nodes_torch, :]

        #print(source_nodes.shape, source_timestamps.shape)
        neighbors, edge_idxs, edge_times, edge_times_spans = self.neighbor_finder.get_temporal_neighbor(source_nodes,
                                                                                                        source_timestamps,
                                                                                                        n_neighbors=n_neighbors)
        neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
        edge_deltas = edge_times_spans
        edge_deltas_torch = torch.from_numpy(edge_deltas).double().to(self.device)
        neighbors = neighbors.flatten()

        neighbor_features = self.node_features[neighbors, :]
        effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbor_features = neighbor_features.view(len(source_nodes), effective_n_neighbors, -1)
        neighbor_time_embedding = self.time_encoder(edge_deltas_torch).float()
        neighbor_mask = neighbors_torch == 0

        sub_graph_embedding = self.temporal_attention(0, source_node_features, source_node_time_embedding,
                                                      destination_node_features, destination_node_time_embedding,
                                                      neighbor_features, neighbor_time_embedding, neighbor_mask)
        return sub_graph_embedding


    def temporal_attention(self, n_layers, source_node_features, source_time_embedding,
                                           destination_node_features, destination_time_embedding,
                                           neighbor_features, neighbor_time_embedding, neighbor_mask):

        return None

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, mask):
        return None


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                memory=memory,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder, n_layers=n_layers,
                                                n_node_features=n_node_features,
                                                n_time_features=n_time_features,
                                                embedding_dimension=embedding_dimension,
                                                device=device,
                                                n_heads=n_heads, dropout=dropout,
                                                use_memory=use_memory)
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features
                                                             , embedding_dimension)
                                             for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                             embedding_dimension) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings],
                                       dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features)
            for _ in range(n_layers)])

    def temporal_attention(self, n_layers, source_node_features, source_time_embedding,
                                           destination_node_features, destination_time_embedding,
                                           neighbor_features, neighbor_time_embedding, neighbor_mask):
        attention_model = self.attention_models[n_layers - 1]
        sub_graph_embedding, _ = attention_model(source_node_features, source_time_embedding,
                                              destination_node_features, destination_time_embedding,
                                              neighbor_features, neighbor_time_embedding, neighbor_mask)
        return sub_graph_embedding

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              mask)

        return source_embedding


def get_embedding_module(module_type, node_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       memory=memory,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       n_node_features=n_node_features,
                                       n_time_features=n_time_features,
                                       embedding_dimension=embedding_dimension,
                                       device=device,
                                       n_heads=n_heads, dropout=dropout, use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 n_heads=n_heads, dropout=dropout, use_memory=use_memory)

    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout,
                             n_neighbors=n_neighbors)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))



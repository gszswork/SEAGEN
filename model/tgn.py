#import logging
import numpy as np
import torch
import math
from collections import defaultdict
from torch import nn
from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function_type="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="TGNF_with_memory",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = None
    self.device = device
    self.node_raw_features = None
    self.dropout = dropout
    self.n_heads = n_heads
    self.n_nodes = None
    self.n_node_features = 768
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    print("use_source_embedding_in_message",use_source_embedding_in_message)
    print("use_destination_embedding_in_message",use_destination_embedding_in_message)
    self.use_destination_embedding_in_message = use_destination_embedding_in_message# False
    self.use_source_embedding_in_message = use_source_embedding_in_message#False
    self.dyrep = dyrep#False
    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None
    self.message_aggregator = None
    self.message_function = None
    self.memory_updater = None
    
    self.memory_dimension = self.n_node_features
    self.memory_update_at_start = memory_update_at_start#True
    self.raw_message_dimension = 2 * self.memory_dimension +  self.time_encoder.dimension
    self.message_dimension = message_dimension if message_function_type != "identity" else self.raw_message_dimension  # 100
    self.aggregator_type = aggregator_type
    self.memory_updater_type = memory_updater_type

    self.embedding_module_type = embedding_module_type#graph_attention
    self.message_function_type = message_function_type
    self.memory_updater = get_memory_updater(module_type=self.memory_updater_type,#gru 
                                               memory=self.memory,
                                               message_dimension=self.message_dimension,#688
                                               memory_dimension=self.memory_dimension,#172
                                               device=self.device)#.to(self.device)
    self.embedding_module = get_embedding_module(module_type=self.embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=self.n_heads, dropout=self.dropout,
                                                 use_memory=self.use_memory,
                                                 n_neighbors=self.n_neighbors)#.to(self.device)
    
    self.message_function = get_message_function(module_type=self.message_function_type,
                                                   raw_message_dimension=self.raw_message_dimension,
                                                   message_dimension=self.message_dimension,
                                                   device=self.device)#.to(self.device)
    
    self.message_aggregator = get_message_aggregator(aggregator_type=self.aggregator_type, device=self.device)#to(self.device)
    
    
    self.class_classifier = nn.Sequential()
    self.class_classifier.add_module('c_fc1', nn.Linear(768, 2))
    self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))
    
    
    # MLP to compute probability on an edge given two node embeddings
    #self.affinity_score = MergeLayer(self.n_node_features)#, self.n_node_features)
    #                                 self.n_node_features,
    #                                 1)
    
    self.updated_embedding = None  #defaultdict(list)
    #self.embedding_output = None#defaultdict(list)
    self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
  
   

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times, edge_idxs, n_neighbors=10, positives = None):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes]) 
    timestamps = np.concatenate([edge_times, edge_times]) 
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:#True
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),#self.n_nodes 9228
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update
    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=None)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(nodes, self.memory.messages)#positives = np.concatenate([source_nodes, destination_nodes])
         
        #assert torch.allclose(memory[nodes], self.memory.get_memory(nodes), atol=1e-5), \
        #  "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(nodes)
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
                                                                             
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
    
    return node_embedding
    #return source_node_embedding, destination_node_embedding, node_embedding

  def forward(self, train_ngh_finder, batchsize, source_nodes, destination_nodes, edge_times, node_raw_features, edge_idxs):

    self.node_raw_features = (torch.from_numpy(node_raw_features.astype(np.float)).to(self.device)).float()#dtype=torch.float64                                        
    self.embedding_module.node_features = self.node_raw_features
    self.set_neighbor_finder(train_ngh_finder)
    self.n_nodes = self.node_raw_features.shape[0]

    if self.use_memory:#True
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension= self.memory_dimension,
                           input_dimension= self.message_dimension,
                           message_dimension= self.message_dimension,
                           device= self.device).to(self.device)

      self.memory_updater.memory = self.memory
    positives = np.concatenate([source_nodes, destination_nodes])

    freq = {}
    for i in positives:
      if i in freq:
          freq[i] = freq[i]+1
      else:
          freq[i] = 1

    del_positives = []

    for i in freq:
      if freq[i] == 1:
        del_positives.append(i)
    del_positives = np.array(del_positives)
    num_instance = len(source_nodes)
    num_batch = math.ceil(num_instance / batchsize) 
    self.updated_embedding = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
    tdn_loss = torch.tensor(0.).to(self.device)
    last_updated_embedding = torch.zeros((768,)).to(self.device)
    updated_idxs = np.array([])
    for k in range(0, num_batch):
            start_idx = k * batchsize
            end_idx = min(num_instance, start_idx + batchsize)
            source_batch, destination_batch = source_nodes[start_idx:end_idx], destination_nodes[start_idx:end_idx]
            # source_batch, destination_batch are all indexes like 1,2,3,4
            all_temp_nodes = np.concatenate((source_batch, destination_batch), 0)
            unique_nodes = np.unique(all_temp_nodes)
            updated_idxs = np.unique(np.concatenate((unique_nodes, updated_idxs)))
            temp = {}
            for i in unique_nodes:
              temp[i] = self.rindex(all_temp_nodes.tolist(), i)

            edge_idx_batch = edge_idxs[start_idx: end_idx]
            timestamp_batch = edge_times[start_idx:end_idx]

            node_embedding = self.compute_temporal_embeddings(source_batch, destination_batch, timestamp_batch, edge_idx_batch, self.n_neighbors, positives = del_positives)
            for i in unique_nodes:
                  self.updated_embedding[i] = node_embedding[temp[i]]
            # TDN
            S_i = last_updated_embedding
            S_i_plus1 = torch.mean(self.updated_embedding[torch.LongTensor(updated_idxs)], dim=0)
            res = self.cos(S_i, S_i_plus1)
            tdn_loss += res
            last_updated_embedding = S_i_plus1
    torch.cuda.empty_cache()
    tdn_loss = tdn_loss / torch.tensor(num_batch).to(self.device)
    return self.updated_embedding, tdn_loss

   




  def rindex(self, lst, value):
        lst.reverse()
        i = lst.index(value)
        lst.reverse()
        return len(lst) - i - 1

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)
    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)#return original unique_messages  

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).double().to(self.device)
    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding #self.use_source_embedding_in_message False
    

    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = (edge_times - self.memory.last_update[source_nodes])#.double()
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1).float()
    source_message = torch.cat([source_memory, destination_memory,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)
    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))
    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder

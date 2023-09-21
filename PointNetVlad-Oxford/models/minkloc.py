# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import models.dgcnn as dgcnn

from models.layers.pooling_wrapper import PoolingWrapper

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        # self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(32, 32))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        batch_size = embedding.shape[0]
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1)  # 0 # nxf -> f  bxnxf->bxf
        transformed_global = torch.tanh(global_context)  # f  bxf
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding, transformed_global.view(batch_size, -1,
                                                                                       1)))  # weights      nxf fx1  bxnxf bxfx1 bxnx1
        representation = torch.matmul(embedding.permute(0, 2, 1), sigmoid_scores)  # bxnxf bxfxn bxnx1 bxfx1
        return representation, sigmoid_scores

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # x4 = torch.randn(10, 100, 4)
    # x5 = x4.transpose(2, 1)
    # x3 = torch.matmul(x4, x5)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, cuda=0, idx=None, xyz=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if xyz:
            idx = knn(x[:, :3, :], k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda:' + str(cuda))  # 'cuda'

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature

class MinkLoc(torch.nn.Module):
    def __init__(self, backbone: nn.Module, pooling: PoolingWrapper, normalize_embeddings: bool = False):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}
        bias_bool = False
        self.filters_1 = 64
        self.filters_2 = 64
        self.filters_3 = 32
        self.global_space = 256
        self.global_filter = 288
        self.number_labels = 12

        self.attention = AttentionModule()

        self.dgcnn_s_conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, self.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv1 = nn.Sequential(
            nn.Conv2d(self.number_labels * 2, self.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv2 = nn.Sequential(
            nn.Conv2d(self.filters_1 * 2, self.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv2 = nn.Sequential(
            nn.Conv2d(self.filters_1 * 2, self.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_s_conv3 = nn.Sequential(
            nn.Conv2d(self.filters_2 * 2, self.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_f_conv3 = nn.Sequential(
            nn.Conv2d(self.filters_2 * 2, self.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.filters_3,
                                                      self.filters_3, kernel_size=1, bias=bias_bool),
                                            nn.BatchNorm1d(self.filters_3), nn.LeakyReLU(negative_slope=0.2))
        # self.global_conv = nn.Conv1d(self.filters_3, self.global_filter, kernel_size=1)
        # self.global_conv_2 = nn.Linear(384, self.global_filter)

    def dgcnn_conv_pass(self, x):
        self.k = 12  # 我们欧式聚类了12类
        xyz = x.permute(0, 2, 1)  # Bx3xN

        xyz = dgcnn.get_graph_feature(xyz, k=self.k, cuda=0)  # Bx6xNxk   128x6x100x12
        xyz = self.dgcnn_s_conv1(xyz)
        xyz1 = xyz.max(dim=-1, keepdim=False)[0]
        xyz = dgcnn.get_graph_feature(xyz1, k=self.k, cuda=0)
        xyz = self.dgcnn_s_conv2(xyz)
        xyz2 = xyz.max(dim=-1, keepdim=False)[0]
        xyz = dgcnn.get_graph_feature(xyz2, k=self.k, cuda=0)
        xyz = self.dgcnn_s_conv3(xyz)
        xyz3 = xyz.max(dim=-1, keepdim=False)[0]

        # sem = dgcnn.get_graph_feature(sem, k=self.k, cuda=0)  # Bx2fxNxk
        # sem = self.dgcnn_f_conv1(sem)
        # sem1 = sem.max(dim=-1, keepdim=False)[0]
        # sem = dgcnn.get_graph_feature(sem1, k=self.k, cuda=0)
        # sem = self.dgcnn_f_conv2(sem)
        # sem2 = sem.max(dim=-1, keepdim=False)[0]
        # sem = dgcnn.get_graph_feature(sem2, k=self.k, cuda=0)
        # sem = self.dgcnn_f_conv3(sem)
        # sem3 = sem.max(dim=-1, keepdim=False)[0]

        # x = torch.cat((xyz3, sem3), dim=1)
        # x = self.dgcnn_conv_all(x)
        x = self.dgcnn_conv_end(xyz3)
        # print(x.shape)

        x = x.permute(0, 2, 1)  # [node_num, 32]
        return x

    def forward(self, batch, batch_graph):
        ##################################  MinkLoc  ###########################################')
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        x = self.backbone(x)
        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.pooling.in_dim, f'Backbone output tensor has: {x.shape[1]} channels. ' \
                                                  f'Expected: {self.pooling.in_dim}'
        x = self.pooling(x)
        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)

        #x = x.flatten(1)
        assert x.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions.'
        assert x.shape[1] == self.pooling.output_dim, f'Output tensor has: {x.shape[1]} channels. ' \
                                                      f'Expected: {self.pooling.output_dim}'
        ### 128*256
        if self.normalize_embeddings:
            x = F.normalize(x, dim=1)
        ##################################  graph  ###########################################
        features_1 = batch_graph.float().cuda()
        # features_1 = data

        # features B x (3+label_num) x node_num
        features_1 = torch.unsqueeze(features_1, dim=0)
        abstract_features = self.dgcnn_conv_pass(features_1)
        pooled_features, _ = self.attention(abstract_features)
        pooled_features = torch.squeeze(pooled_features)
        # x is (batch_size, output_dim) tensor

        global_feature = torch.zeros([x.shape[0], self.global_filter])
        if x.shape[0] == pooled_features.shape[0]:
            global_feature = torch.cat((x, pooled_features), dim=1)
        else:
            global_feature[:, :self.global_space] = x
        return {'global': global_feature}

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')
        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')

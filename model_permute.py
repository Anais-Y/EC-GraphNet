import torch
import torch.nn as nn
import torch.nn.functional as F

class gcn_operation(nn.Module):
    def __init__(self, in_dim, out_dim, num_vertices, strides, activation='GLU'):
        """
        图卷积模块
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = nn.Parameter(torch.FloatTensor(num_vertices, num_vertices))
        nn.init.xavier_uniform_(self.adj)
        self.weight_cross_time = nn.Parameter(torch.FloatTensor(strides - 1, num_vertices, num_vertices))
        nn.init.xavier_uniform_(self.weight_cross_time)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation
        self.strides = strides
        self.bn = nn.BatchNorm1d(self.out_dim)

        assert self.activation in {'GLU', 'relu', 'leakyRelu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def construct_adj(self):
        N = self.num_vertices
        adj = self.adj
        local_adj = torch.zeros((N * self.strides, N * self.strides))
        for i in range(self.strides):
            local_adj[i * N:(i + 1) * N, i * N: (i + 1) * N] = adj

        for k in range(self.strides - 1):
            """两个切片之间完全连接"""
            local_adj[k*N:(k+1)*N, (k+1)*N:(k+2)*N] = self.weight_cross_time[k]  # 可更新的权重
            local_adj[(k+1)*N:(k+2)*N, k*N:(k+1)*N] = self.weight_cross_time[k].T

        return local_adj

    def forward(self, x):
        """
        :param x: (3*N, B, Cin)
        :return: (3*N, B, Cout)
        """
        adj = self.construct_adj()

        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)
        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 3*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            x = self.FC(x)  # 3*N, B, Cout
            x = torch.permute(x, (1, 2, 0))  # B, Cout, 3*N
            x = self.bn(x)
            x = torch.permute(x, (2, 0, 1))
            return torch.relu(x)

        elif self.activation == 'leakyRelu':
            # return torch.nn.functional.leaky_relu(self.FC(x))
            x = self.FC(x)  # 3*N, B, Cout
            x = torch.permute(x, (1, 2, 0))  # B, Cout, 3*N
            x = self.bn(x)
            x = torch.permute(x, (2, 0, 1))
            return F.leaky_relu(x)


class STSGCM(nn.Module):
    def __init__(self, in_dim, out_dims, num_of_vertices, strides, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.strides = strides

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation,
                strides=self.strides
            )
        )
# in_dim-->out_dims[0]-->out_dims[1]-->out_dims[...]-->out_dims[n-1]

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation,
                    strides=self.strides
                )
            )

    def forward(self, x):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x)
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]  # crop操作

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)

        del need_concat

        return out
# 这是把每一层的输出都连接起来然后取最大的


class STSGCL(nn.Module):
    def __init__(self,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=3,
                 activation='GLU',
                 temporal_emb=True,
                 spatial_emb=True):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation,
                    strides=self.strides
                )
            )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-2, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.history - self.strides + 1):
            t = x[:, i: i+self.strides, :, :]  # (B, 3, N, Cin)

            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
            # (B, 3*N, Cin)

            t = self.STSGCMS[i](t.permute(1, 0, 2))  # (3*N, B, Cin) -> (N, B, Cout)

            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)

            need_concat.append(t)

        out = torch.cat(need_concat, dim=1)  # (B, T-2, N, Cout)

        del need_concat, batch_size

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # 应用注意力得分
        return torch.matmul(attention_scores, V)


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim,
                 hidden_dim=128):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices  # 14
        self.history = history  # 3
        self.in_dim = in_dim  # 128
        self.hidden_dim = hidden_dim  # 128 by default

        self.FC1 = nn.Linear(self.num_of_vertices * self.hidden_dim, self.hidden_dim, bias=True)
        self.self_attention = SelfAttention(in_dim * history, hidden_dim)

        self.FC2 = nn.Linear(self.hidden_dim, 2, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B)
        """
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin
        x = x.reshape(batch_size, self.num_of_vertices, self.history*self.in_dim)  # (B*N, Tin, Cin)
        x = self.self_attention(x)  # Apply self-attention
        # out1 = x.reshape(batch_size, self.num_of_vertices * self.hidden_dim)  # Reshape to (B, N*hidden_dim)
        out1 = F.leaky_relu(self.FC1(x.reshape(batch_size, -1)))
        out2 = self.FC2(out1)  # (B, N*hidden_dim) -> (B, 2)

        del out1, batch_size

        return torch.squeeze(out2)  # B

class STSGCN(nn.Module):
    def __init__(self, history, num_of_vertices, in_dim, hidden_dims,
                 first_layer_embedding_size, out_layer_dim, activation='GLU',
                 temporal_emb=True, spatial_emb=True, strides=3):
        """

        :param adj: local时空间矩阵
        :param history:输入时间步长
        :param num_of_vertices:节点数量
        :param in_dim:输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param horizon:预测时间步长
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        """
        super(STSGCN, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.hidden_dims = hidden_dims
        self.out_layer_dim = out_layer_dim
        self.activation = activation

        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.strides = strides

        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=9)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]  # 这里的hidden_dim有四个元素，即有四个GCL
        history -= (self.strides - 1)  # 每经过一次GCL特征T都会少2

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )

            history -= (self.strides - 1)
            in_dim = hidden_list[-1]

        self.predictLayer = output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=in_dim,
                    hidden_dim=out_layer_dim
                )

    def forward(self, x):
        """
        :param x: B, Tin, N, Cin)
        :return: B
        """

        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin

        x = self.bn1(x)

        for model in self.STSGCLS:
            x = model(x)
        # (B, T - 8, N, Cout)  因为有4个串联的STSGCL

        out = self.predictLayer(x)  # (B, )

        return out



































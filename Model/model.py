import numpy as np
import torch
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Model.ST_Fusion import*


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = torch.diag(torch.sum(W, dim=1))

    L = D - W

    eigenvalues = torch.linalg.eig(L)[0] #torch.size([90,]) # 获取特征值
    magnitude = torch.abs(eigenvalues)
    lambda_max = torch.max(magnitude).item()  # 获取最大特征值
    # lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - torch.eye(W.shape[0]).to(L.device)

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde.clone()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def my_corrcoef(x):
    x = x - x.mean(dim=1, keepdim=True)
    y = x / (x.norm(dim=1, keepdim=True) + 1e-6)
    return y.mm(y.t())

def pearson_adj(node_features):
    bs, N, dimen = node_features.size() #20,90,195

    Adj_matrices = []
    for b in range(bs):
        corr_matrix = my_corrcoef(node_features[b]) #torch
        corr_matrix = (corr_matrix + 1) / 2
        L_tilde = scaled_Laplacian(corr_matrix)
        cheb_polynomials = cheb_polynomial(L_tilde, K=3)
        # cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor) for i in cheb_polynomial(L_tilde, K=3)]
        Adj_matrices.append(torch.stack(cheb_polynomials))
    Adj = torch.stack(Adj_matrices)

    return Adj

################K-order static GCN###################
class cheb_conv(nn.Module):
    '''
        K-order chebyshev graph convolution with static graph structure
        --------
        Input:  [x (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
                 SAtt(batch_size, num_of_vertices, num_of_vertices)]
        Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)

    '''

    def __init__(self, num_of_filters, k, num_of_timesteps, num_of_vertices, num_of_features):

        super(cheb_conv, self).__init__()
        self.k = k
        self.num_of_filters = num_of_filters
        self.num_of_features = num_of_features
        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        # self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.zeros(self.k, self.num_of_features, self.num_of_filters))
        nn.init.xavier_uniform_(self.Theta.data, gain=1.414)
        self.t = nn.Conv2d(self.num_of_timesteps, 1, 3, 1)

    def forward(self, x, cheb_polynomials):
        # _, num_of_timesteps, num_of_vertices, num_of_features = x.shape

        outputs = []
        for time_step in range(self.num_of_timesteps):
            # shape is (batch_size, V, F)
            # graph_signal = x[:, time_step, :, :] #torch.size([20,90,195])
            graph_signal = x
            # shape is (batch_size, V, F')
            output = torch.zeros(x.shape[0], self.num_of_vertices, self.num_of_filters) #torch.size([20,90,45])

            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[:, kk, :, :] #torch.size([20,90,90])

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk] #torch.size([20,195,45])

                # shape is (batch_size, V, F)
                rhs = torch.matmul(T_k.permute(0, 2, 1), graph_signal) #torch.size([20,90,195])
                output = output.to(rhs.device) + torch.matmul(rhs, theta_k) #torch.size([20,90,45])

            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        outputs = self.t(outputs)  # torch.Size([20, 1, 88, 43])
        outputs = F.relu(outputs) #torch.size([20,1,90,45])
        return outputs

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gcn = cheb_conv(num_of_filters=45, k=3, num_of_timesteps=1,
                             num_of_vertices=90,num_of_features=98)
        self.st_fusion = ST_Fusion(in_channels=98)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(7568, 1024)  # adni:7568 15136 22704;
        self.bn1 = nn.BatchNorm1d(1024)
        self.d1 = nn.Dropout(p=0.3)
        self.l2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.3)
        self.l3 = nn.Linear(256, 2)

    def forward(self, fdata, ddata):
        # fdata = fdata.unsqueeze(1)
        # X：（20，32，62，40）
        bs, tlen, num_nodes, seq = fdata.size() #20,2,90,98
        # fdata = fdata.permute(1, 0, 2, 3) #2,20,90,98

        # # Construct the adjacency matrix using Pearson correlation
        # A_input = tr.reshape(fdata, [bs * tlen, num_nodes, seq])#40,90,98
        # adj = pearson_adj(A_input) #40,3,90,90
        # adj_pro = adj.reshape(bs, tlen, 3, num_nodes, num_nodes)
        # adj_pro = adj_pro.mean(dim=(1, 2))  # => shape: (20, 90, 90)
        # adj = tr.reshape(adj, [tlen, bs, adj.shape[1], adj.shape[2], adj.shape[3]]) # 2,bs,3,90,90

        output = []
        # fdata = fdata.permute(1, 0, 2, 3) #20,2,90,98
        for i in range (tlen // 2):
            x1 = fdata[:, i, :, :]  # (B, V, F) t1
            x2 = fdata[:, 2*i+1, :, :]  # (B, V, F) t2
            output1, output2 = self.st_fusion(x1, x2, ddata)
            output.append(output1)
            output.append(output2)
        final_output = torch.stack(output)

        # Construct the adjacency matrix using Pearson correlation
        A_input = tr.reshape(final_output, [bs * tlen, num_nodes, seq])#40,90,98
        adj = pearson_adj(A_input) #40,3,90,90
        adj_pro = adj.reshape(bs, tlen, 3, num_nodes, num_nodes)
        adj_pro = adj_pro.mean(dim=(1, 2))  # => shape: (20, 90, 90)
        adj = tr.reshape(adj, [tlen, bs, adj.shape[1], adj.shape[2], adj.shape[3]]) # 2,bs,3,90,90

        out = []
        for i in range(adj.size(0)):
            adj_t = adj[i]
            out_t = self.gcn(final_output[i], adj_t)
            out.append(out_t)
        block_out = torch.stack(out) #torch.size([2,20,1,88,43])
        block_out = block_out.squeeze(2) #torch.size([2,20,88,43])
        block_out = block_out.squeeze(0)
        block_out = block_out.permute(1, 0, 2, 3)#20 2 88 43

        block_outs = self.f1(block_out) #20 7568
        block_outs = self.d1(self.bn1(self.l1(block_outs)))
        block_outs = self.d2(self.bn2(self.l2(block_outs)))
        out_logits = self.l3(block_outs)

        return F.softmax(out_logits, dim=1), block_out, block_outs,adj_pro


# dim = (20, 116, 220)
# dim = (20, 90, 195)
# test = torch.randn(dim)
# print(test.shape)
# model = Model()
# out = model(test)
# print(out.shape)
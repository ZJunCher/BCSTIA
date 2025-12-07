import torch
import torch.nn as nn
import torch.nn.functional as F


# Contains all ablation variants and the full model

class SpatiotemporalAttentionInteraction(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        super(SpatiotemporalAttentionInteraction, self).__init__()
        assert dimension in [2]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.energy_time_1_sf = nn.Softmax(dim=-1)
        self.energy_time_2_sf = nn.Softmax(dim=-1)
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1) #V1 torch.size(20,49,90)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)  # V2 torch.size(20,49,90)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)  # Q1 torch.size(20,49,90)
        theta_x2 = theta_x1.permute(0, 2, 1)  # K1^T torch.size(20,90,49)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)  # Q2 torch.size(20,49,90)
        phi_x2 = phi_x1.permute(0, 2, 1)  # K2^T torch.size(20,90,49)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)  # S1 = Q1 * K2^T  torch.size(20,49,49)
        energy_time_2 = energy_time_1.permute(0, 2, 1)  # S1^T  torch.size(20,49,49)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # S2 = K1^T * Q2 torch.size(20,90,90)
        energy_space_2 = energy_space_1.permute(0, 2, 1)  # S2^T

        energy_time_1s = self.energy_time_1_sf(energy_time_1)  # Softmax(S1)
        energy_time_2s = self.energy_time_2_sf(energy_time_2)  # Softmax(S1^T)
        energy_space_2s = self.energy_space_2s_sf(energy_space_1)  # Softmax(S2)
        energy_space_1s = self.energy_space_1s_sf(energy_space_2)  # Softmax(S2^T)

        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()  # Z1 torch.size(20,49,90)
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()  # Z2 torch.size(20,49,90)
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)


class SpatiotemporalAttentionCrossInteraction(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(SpatiotemporalAttentionCrossInteraction, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )
        self.phi = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm1d(self.in_channels)
        )

        # Cross Attention Projection
        self.q_time_proj = nn.Linear(self.in_channels, self.inter_channels)
        self.k_space_proj = nn.Linear(90, self.inter_channels)
        self.q_space_proj = nn.Linear(90, self.inter_channels)
        self.k_time_proj = nn.Linear(self.in_channels, self.inter_channels)

        self.softmax_t = nn.Softmax(dim=-1)
        self.softmax_s = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        B, F, V = x1.size()

        g_x1 = self.g(x1).reshape(B, self.inter_channels, -1)
        g_x2 = self.g(x2).reshape(B, self.inter_channels, -1)
        theta_x1 = self.theta(x1).reshape(B, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi(x2).reshape(B, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)  # (B, F, F)
        energy_time_2 = energy_time_1.permute(0, 2, 1)  # (B, F, F)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # (B, V, V)
        energy_space_2 = energy_space_1.permute(0, 2, 1)  # (B, V, V)

        attn_time_1 = self.softmax_t(energy_time_1)
        attn_time_2 = self.softmax_t(energy_time_2)
        attn_space_1 = self.softmax_s(energy_space_2)
        attn_space_2 = self.softmax_s(energy_space_1)

        z1 = torch.matmul(torch.matmul(attn_time_2, g_x1), attn_space_2).contiguous()
        z2 = torch.matmul(torch.matmul(attn_time_1, g_x2), attn_space_1).contiguous()

        z1 = z1.view(B, self.inter_channels, V)
        z2 = z2.view(B, self.inter_channels, V)

        ## T → S：Q_time(x1) × K_space(x2)
        q_time = self.q_time_proj(x1.mean(dim=2))    # (B, F) → (B, C)
        k_space = self.k_space_proj(x2.mean(dim=1))  # (B, V) → (B, C)
        attn_t2s = self.softmax_s(torch.bmm(q_time.unsqueeze(1), k_space.unsqueeze(2)))  # (B, 1, 1)
        x2 = x2 + attn_t2s.squeeze(2).unsqueeze(2) * x1.mean(dim=2).unsqueeze(2)  # broadcasting

        ## S → T：Q_space(x1) × K_time(x2)
        q_space = self.q_space_proj(x1.mean(dim=1))   # (B, V) → (B, C)
        k_time = self.k_time_proj(x2.mean(dim=2))     # (B, F) → (B, C)
        attn_s2t = self.softmax_t(torch.bmm(q_space.unsqueeze(1), k_time.unsqueeze(2)))  # (B, 1, 1)
        x1 = x1 + attn_s2t.squeeze(2).unsqueeze(2) * x2.mean(dim=1).unsqueeze(1)

        return x1 + self.W(z1), x2 + self.W(z2)

class SpatiotemporalAttentionCrossInteractionFuse(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(SpatiotemporalAttentionCrossInteractionFuse, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )
        self.phi = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm1d(self.in_channels)
        )

        # Cross Attention Projection
        self.q_time_proj = nn.Linear(self.in_channels, self.inter_channels)
        self.k_space_proj = nn.Linear(90, self.inter_channels)
        self.q_space_proj = nn.Linear(90, self.inter_channels)
        self.k_time_proj = nn.Linear(self.in_channels, self.inter_channels)

        self.softmax_t = nn.Softmax(dim=-1)
        self.softmax_s = nn.Softmax(dim=-1)

        # self.alpha = nn.Parameter(torch.tensor(0.33))
        # self.beta = nn.Parameter(torch.tensor(0.33))
        # self.gamma = nn.Parameter(torch.tensor(0.33))
        # self.omiga = nn.Parameter(torch.tensor(0.33))

    def forward(self, x1, x2):
        B, T, V = x1.size()

        #SITA
        g_x1 = self.g(x1).reshape(B, self.inter_channels, -1)
        g_x2 = self.g(x2).reshape(B, self.inter_channels, -1)
        theta_x1 = self.theta(x1).reshape(B, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi(x2).reshape(B, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)  # (B, F, F)
        energy_time_2 = energy_time_1.permute(0, 2, 1)  # (B, F, F)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # (B, V, V)
        energy_space_2 = energy_space_1.permute(0, 2, 1)  # (B, V, V)

        attn_time_1 = self.softmax_t(energy_time_1)
        attn_time_2 = self.softmax_t(energy_time_2)
        attn_space_1 = self.softmax_s(energy_space_2)
        attn_space_2 = self.softmax_s(energy_space_1)

        z1 = torch.matmul(torch.matmul(attn_time_2, g_x1), attn_space_2).contiguous()
        z2 = torch.matmul(torch.matmul(attn_time_1, g_x2), attn_space_1).contiguous()

        z1 = z1.view(B, self.inter_channels, V)
        z2 = z2.view(B, self.inter_channels, V)

        #XSTA
        # T → S
        q_time = self.q_time_proj(x1.mean(dim=2))  # (B, C)
        k_space = self.k_space_proj(x2.mean(dim=1))  # (B, C)
        attn_t2s = self.softmax_s(torch.bmm(q_time.unsqueeze(1), k_space.unsqueeze(2)))  # (B, 1, 1)
        cross_t2s = attn_t2s.squeeze(2).unsqueeze(2) * x1.mean(dim=2).unsqueeze(2)
        # x2 = x2 + cross_t2s

        # S → T
        q_space = self.q_space_proj(x1.mean(dim=1))  # (B, C)
        k_time = self.k_time_proj(x2.mean(dim=2))  # (B, C)
        attn_s2t = self.softmax_t(torch.bmm(q_space.unsqueeze(1), k_time.unsqueeze(2)))  # (B, 1, 1)
        cross_s2t = attn_s2t.squeeze(2).unsqueeze(2) * x2.mean(dim=1).unsqueeze(1)
        # x1 = x1 + cross_s2t

        # Adaptive Fusion
        res1 = x1
        proj1 = self.W(z1)
        f1 = res1.mean(dim=(1, 2))
        f2 = cross_s2t.mean(dim=(1, 2))
        f3 = proj1.mean(dim=(1, 2))
        w1 = F.softmax(torch.stack([f1, f2, f3], dim=1), dim=1)  # (B, 3)
        fused_x1 = (
                w1[:, 0].view(B, 1, 1) * res1 +
                w1[:, 1].view(B, 1, 1) * cross_s2t +
                w1[:, 2].view(B, 1, 1) * proj1
        )


        res2 = x2
        proj2 = self.W(z2)
        f1_2 = res2.mean(dim=(1, 2))
        f2_2 = cross_t2s.mean(dim=(1, 2))
        f3_2 = proj2.mean(dim=(1, 2))
        w2 = F.softmax(torch.stack([f1_2, f2_2, f3_2], dim=1), dim=1)  # (B, 3)
        fused_x2 = (
                w2[:, 0].view(B, 1, 1) * res2 +
                w2[:, 1].view(B, 1, 1) * cross_t2s +
                w2[:, 2].view(B, 1, 1) * proj2
        )

        return fused_x1, fused_x2



class SpatiotemporalAttentionSingleDir(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(SpatiotemporalAttentionSingleDir, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )
        self.phi = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm1d(self.in_channels)
        )

        # Cross Attention Projection
        self.q_time_proj = nn.Linear(self.in_channels, self.inter_channels)
        self.k_space_proj = nn.Linear(90, self.inter_channels)
        self.q_space_proj = nn.Linear(90, self.inter_channels)
        self.k_time_proj = nn.Linear(self.in_channels, self.inter_channels)

        self.softmax_t = nn.Softmax(dim=-1)
        self.softmax_s = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        B, F, V = x1.size()

        g_x2 = self.g(x2).reshape(B, self.inter_channels, -1)
        theta_x1 = self.theta(x1).reshape(B, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi(x2).reshape(B, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)  # (B, F, F)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # (B, V, V)
        energy_space_2 = energy_space_1.permute(0, 2, 1)  # (B, V, V)

        attn_time_1 = self.softmax_t(energy_time_1)
        attn_space_1 = self.softmax_s(energy_space_2)


        z2 = torch.matmul(torch.matmul(attn_time_1, g_x2), attn_space_1).contiguous()
        z2 = z2.view(B, self.inter_channels, V)

        ## T → S：Q_time(x1) × K_space(x2)
        q_time = self.q_time_proj(x1.mean(dim=2))    # (B, F) → (B, C)
        k_space = self.k_space_proj(x2.mean(dim=1))  # (B, V) → (B, C)
        attn_t2s = self.softmax_s(torch.bmm(q_time.unsqueeze(1), k_space.unsqueeze(2)))  # (B, 1, 1)
        x2 = x2 + attn_t2s.squeeze(2).unsqueeze(2) * x1.mean(dim=2).unsqueeze(2)  # broadcasting

        return x1, x2 + self.W(z2)

class SpatiotemporalAttentionCross(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(SpatiotemporalAttentionCross, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )
        self.phi = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm1d(self.in_channels)
        )

        # Cross Attention Projection
        self.q_time_proj = nn.Linear(self.in_channels, self.inter_channels)
        self.k_space_proj = nn.Linear(116, self.inter_channels)
        self.q_space_proj = nn.Linear(116, self.inter_channels)
        self.k_time_proj = nn.Linear(self.in_channels, self.inter_channels)

        self.softmax_t = nn.Softmax(dim=-1)
        self.softmax_s = nn.Softmax(dim=-1)

    def forward(self, x1, x2):

        ## T → S：Q_time(x1) × K_space(x2)
        q_time = self.q_time_proj(x1.mean(dim=2))    # (B, F) → (B, C)
        k_space = self.k_space_proj(x2.mean(dim=1))  # (B, V) → (B, C)
        attn_t2s = self.softmax_s(torch.bmm(q_time.unsqueeze(1), k_space.unsqueeze(2)))  # (B, 1, 1)
        x2 = x2 + attn_t2s.squeeze(2).unsqueeze(2) * x1.mean(dim=2).unsqueeze(2)  # broadcasting

        ## S → T：Q_space(x1) × K_time(x2)
        q_space = self.q_space_proj(x1.mean(dim=1))   # (B, V) → (B, C)
        k_time = self.k_time_proj(x2.mean(dim=2))     # (B, F) → (B, C)
        attn_s2t = self.softmax_t(torch.bmm(q_space.unsqueeze(1), k_time.unsqueeze(2)))  # (B, 1, 1)
        x1 = x1 + attn_s2t.squeeze(2).unsqueeze(2) * x2.mean(dim=1).unsqueeze(1)

        return x1 , x2

class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Conv1d(channels, channels, kernel_size=1)
        self.key   = nn.Conv1d(channels, channels, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: (B, C, L)
        Q = self.query(x)  # (B, C, L)
        K = self.key(x)    # (B, C, L)
        V = self.value(x)  # (B, C, L)

        attn_scores = torch.bmm(Q.transpose(1, 2), K) / (Q.shape[1] ** 0.5)  # (B, L, L)
        attn_weights = self.softmax(attn_scores)
        out = torch.bmm(attn_weights, V.transpose(1, 2))  # (B, L, C)
        out = out.transpose(1, 2)  # (B, C, L)
        return x + out

class SpatiotemporalAttentionFull(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(SpatiotemporalAttentionFull, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        )
        self.phi = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        )
        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels)
        )

        # Cross Attention
        self.q_time_proj = nn.Linear(in_channels, self.inter_channels)
        self.k_space_proj = nn.Linear(116, self.inter_channels)
        self.q_space_proj = nn.Linear(116, self.inter_channels)
        self.k_time_proj = nn.Linear(in_channels, self.inter_channels)

        self.softmax_t = nn.Softmax(dim=-1)
        self.softmax_s = nn.Softmax(dim=-1)

        # Self-Attention
        self.self_att_time = SelfAttention1D(in_channels)
        self.self_att_space = SelfAttention1D(116)

    def forward(self, x1, x2):
        B, F, V = x1.size()  # x1, x2: (B, F, V)

        x1_time_attn = self.self_att_time(x1)  # (B, C, V)
        x2_time_attn = self.self_att_time(x2)

        x1_space_attn = self.self_att_space(x1.transpose(1, 2))  # (B, V, C) -> transpose -> (B, C, F)
        x2_space_attn = self.self_att_space(x2.transpose(1, 2))

        x1_space_attn = x1_space_attn.transpose(1, 2)  # (B, C, V)
        x2_space_attn = x2_space_attn.transpose(1, 2)

        x1 = x1 + x1_time_attn + x1_space_attn
        x2 = x2 + x2_time_attn + x2_space_attn

        g_x1 = self.g(x1).reshape(B, self.inter_channels, -1)
        g_x2 = self.g(x2).reshape(B, self.inter_channels, -1)
        theta_x1 = self.theta(x1).reshape(B, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi(x2).reshape(B, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)  # (B, F, F)
        energy_time_2 = energy_time_1.permute(0, 2, 1)  # (B, F, F)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # (B, V, V)
        energy_space_2 = energy_space_1.permute(0, 2, 1)  # (B, V, V)

        attn_time_1 = self.softmax_t(energy_time_1)
        attn_time_2 = self.softmax_t(energy_time_2)
        attn_space_1 = self.softmax_s(energy_space_2)
        attn_space_2 = self.softmax_s(energy_space_1)

        z1 = torch.matmul(torch.matmul(attn_time_2, g_x1), attn_space_2).contiguous()
        z2 = torch.matmul(torch.matmul(attn_time_1, g_x2), attn_space_1).contiguous()

        z1 = z1.view(B, self.inter_channels, V)
        z2 = z2.view(B, self.inter_channels, V)

        # 2. Cross Attention
        q_time = self.q_time_proj(x1.mean(dim=2))    # (B, F) -> (B, C)
        k_space = self.k_space_proj(x2.mean(dim=1))  # (B, V) -> (B, C)
        attn_t2s = self.softmax_s(torch.bmm(q_time.unsqueeze(1), k_space.unsqueeze(2)))  # (B, 1, 1)
        x2 = x2 + attn_t2s.squeeze(2).unsqueeze(2) * x1.mean(dim=2).unsqueeze(2)

        q_space = self.q_space_proj(x1.mean(dim=1))   # (B, V) -> (B, C)
        k_time = self.k_time_proj(x2.mean(dim=2))     # (B, F) -> (B, C)
        attn_s2t = self.softmax_t(torch.bmm(q_space.unsqueeze(1), k_time.unsqueeze(2)))  # (B, 1, 1)
        x1 = x1 + attn_s2t.squeeze(2).unsqueeze(2) * x2.mean(dim=1).unsqueeze(1)

        # x1 = self.self_att_time(x1)  # (B, F, V)
        # x2 = self.self_att_space(x2)  # (B, F, V)

        return x1 + self.W(z1), x2 + self.W(z2)


class SpatiotemporalAttentionCrossInteractionFuse_DTI(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(SpatiotemporalAttentionCrossInteractionFuse_DTI, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )
        self.phi = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        )

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm1d(self.in_channels)
        )

        self.q_time_proj = nn.Linear(self.in_channels, self.inter_channels)
        self.k_space_proj = nn.Linear(90, self.inter_channels)
        self.q_space_proj = nn.Linear(90, self.inter_channels)
        self.k_time_proj = nn.Linear(self.in_channels, self.inter_channels)

        self.softmax_t = nn.Softmax(dim=-1)
        self.softmax_s = nn.Softmax(dim=-1)

    def forward(self, x1, x2, struct_graph=None):
        B, T, V = x1.size()

        # --- SITA ---
        g_x1 = self.g(x1).reshape(B, self.inter_channels, -1)
        g_x2 = self.g(x2).reshape(B, self.inter_channels, -1)
        theta_x1 = self.theta(x1).reshape(B, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        phi_x1 = self.phi(x2).reshape(B, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        # Temporal Attention
        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)

        # Spatial Attention (with Structural Graph Modulation)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # (B, V, V)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        # if struct_graph is not None:
        #     struct_graph = struct_graph.to(energy_space_1.device)
        #     energy_space_1 = energy_space_1 * struct_graph  # DTI调制
        #     energy_space_2 = energy_space_2 * struct_graph

        if struct_graph is not None:
            struct_graph = struct_graph.to(energy_space_1.device)  # (B, V, V)

            # Construct the identity matrix (residual path)
            I = torch.eye(V, device=struct_graph.device).unsqueeze(0).expand(B, -1, -1)

            # Gating coefficient α derived from attention values
            alpha = torch.sigmoid(energy_space_1)  # (B, V, V)

            # Generate the structure-guided modulation matrix
            modulated_graph = alpha * struct_graph + (1 - alpha) * I

            # The fused structural modulation is incorporated into the attention computation
            energy_space_1 = energy_space_1 * modulated_graph
            energy_space_2 = energy_space_2 * modulated_graph.transpose(1, 2)

        attn_time_1 = self.softmax_t(energy_time_1)
        attn_time_2 = self.softmax_t(energy_time_2)
        attn_space_1 = self.softmax_s(energy_space_2)
        attn_space_2 = self.softmax_s(energy_space_1)

        z1 = torch.matmul(torch.matmul(attn_time_2, g_x1), attn_space_2).contiguous()
        z2 = torch.matmul(torch.matmul(attn_time_1, g_x2), attn_space_1).contiguous()

        z1 = z1.view(B, self.inter_channels, V)
        z2 = z2.view(B, self.inter_channels, V)

        # --- XSTA ---
        q_time = self.q_time_proj(x1.mean(dim=2))
        k_space = self.k_space_proj(x2.mean(dim=1))
        attn_t2s = self.softmax_s(torch.bmm(q_time.unsqueeze(1), k_space.unsqueeze(2)))
        cross_t2s = attn_t2s.squeeze(2).unsqueeze(2) * x1.mean(dim=2).unsqueeze(2)

        q_space = self.q_space_proj(x1.mean(dim=1))
        k_time = self.k_time_proj(x2.mean(dim=2))
        attn_s2t = self.softmax_t(torch.bmm(q_space.unsqueeze(1), k_time.unsqueeze(2)))
        cross_s2t = attn_s2t.squeeze(2).unsqueeze(2) * x2.mean(dim=1).unsqueeze(1)

        # --- Adaptive Fusion ---
        res1 = x1
        proj1 = self.W(z1)
        f1 = res1.mean(dim=(1, 2))
        f2 = cross_s2t.mean(dim=(1, 2))
        f3 = proj1.mean(dim=(1, 2))
        w1 = F.softmax(torch.stack([f1, f2, f3], dim=1), dim=1)
        fused_x1 = (
            w1[:, 0].view(B, 1, 1) * res1 +
            w1[:, 1].view(B, 1, 1) * cross_s2t +
            w1[:, 2].view(B, 1, 1) * proj1
        )

        res2 = x2
        proj2 = self.W(z2)
        f1_2 = res2.mean(dim=(1, 2))
        f2_2 = cross_t2s.mean(dim=(1, 2))
        f3_2 = proj2.mean(dim=(1, 2))
        w2 = F.softmax(torch.stack([f1_2, f2_2, f3_2], dim=1), dim=1)
        fused_x2 = (
            w2[:, 0].view(B, 1, 1) * res2 +
            w2[:, 1].view(B, 1, 1) * cross_t2s +
            w2[:, 2].view(B, 1, 1) * proj2
        )

        return fused_x1, fused_x2

class ST_Fusion(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(ST_Fusion, self).__init__()
        # self.stf = SpatiotemporalAttentionCrossInteraction(in_channels, inter_channels)
        # self.stf = SpatiotemporalAttentionFull(in_channels, inter_channels)
        # self.stf = SpatiotemporalAttentionInteraction(in_channels, inter_channels)
        # self.stf = SpatiotemporalAttentionSingleDir(in_channels, inter_channels)
        # self.stf = SpatiotemporalAttentionCross(in_channels, inter_channels)
        # self.stf = SpatiotemporalAttentionCrossInteractionFuse(in_channels, inter_channels)
        self.stf = SpatiotemporalAttentionCrossInteractionFuse_DTI(in_channels, inter_channels)

    def forward(self, x1, x2, ddata):
        """
        :param x: (B, T, V, F)
        """
        # # Select features from two time steps
        # x1 = x[:, 0, :, :]  # (B, V, F) Extract features at time t1
        # x2 = x[:, 1, :, :]  # (B, V, F) Extract features at time t2
        #
        x1 = x1.permute(0, 2, 1)  # (B, F, V)
        x2 = x2.permute(0, 2, 1)  # (B, F, V)

        out1, out2 = self.stf(x1, x2, ddata)

        out1 = out1.permute(0, 2, 1)  # (B, V, F)
        out2 = out2.permute(0, 2, 1)  # (B, V, F)

        return out1, out2
# # Example input (B, T, V, F) Format，B=20, T=2, V=90, F=98
# input_data = torch.randn(20, 2, 90, 98)
# st_fusion = ST_Fusion(in_channels=98, inter_channels=49)
# output1, output2 = tiam_model(input_data)
#
# print(output1.shape)  #  (20, 90, 98)
# print(output2.shape)  #  (20, 90, 98)

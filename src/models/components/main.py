from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation,
    grouping_operation,
)
from torch import nn, einsum


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()
    new_xyz = gather_operation(
        xyz, furthest_point_sample(xyz_flipped, npoint)
    )
    if idx is None:
        _, idx = KNN(k, transpose_mode=True)(
            xyz_flipped, new_xyz.permute(0, 2, 1).contiguous()
        )
        idx = idx.int()

    grouped_xyz = grouping_operation(xyz, idx)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    b, _, nsample = xyz.shape
    device = xyz.device

    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        _, idx = KNN(k=k, transpose_mode=True)(
            x.transpose(1, 2), x.transpose(1, 2)
        )
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = (
        idx + idx_base
    )

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[
        idx, :
    ]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class MlpRes(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MlpRes, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class MlpConv(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MlpConv, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=(1, 1),
        stride=(1, 1),
        if_bn=True,
        activation_fn: Optional[Callable] = torch.relu,
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.conv(x)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class PointNetSAModuleKNN(nn.Module):
    def __init__(
        self,
        npoint,
        nsample,
        in_channel,
        mlp,
        if_bn=True,
        group_all=False,
        use_xyz=True,
        if_idx=False,
    ):
        super(PointNetSAModuleKNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(
            Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None)
        )
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz
            )
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(
                xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx
            )

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


class PointTransformerBlock(nn.Module):
    def __init__(
        self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4
    ):
        super(PointTransformerBlock, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

        self.query_knn = KNN(k=n_knn, transpose_mode=True)

    def forward(self, x, pos):

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()

        _, idx_knn = self.query_knn(pos_flipped, pos_flipped)
        idx_knn = idx_knn.int()
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(
            pos, idx_knn
        )
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum("b c i j, b c i j -> b c i", attention, value)
        y = self.linear_end(agg)

        return y + identity


class StructuralSemanticAttention(nn.Module):

    def __init__(
        self,
        in_channel,
        pos_channel,
        dim=256,
        n_knn=16,
        pos_hidden_dim=64,
        attn_hidden_multiplier=4,
    ):
        super(StructuralSemanticAttention, self).__init__()
        self.mlp_v = MlpRes(
            in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel
        )
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(pos_channel, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

        self.query_knn = KNN(k=self.n_knn, transpose_mode=True)

    def forward(self, pos, key, query):
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()

        _, idx_knn = self.query_knn(pos_flipped, pos_flipped)
        idx_knn = idx_knn.int()

        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(
            pos, idx_knn
        )
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum("b c i j, b c i j -> b c i", attention, value)
        y = self.conv_end(agg)

        return y + identity


class FeatureExtractor(nn.Module):
    def __init__(self, feat_channel=3, out_dim=1024):
        super(FeatureExtractor, self).__init__()
        self.feat_channel = feat_channel
        self.sa_module_1 = PointNetSAModuleKNN(
            512,
            16,
            self.feat_channel,
            [64, 128],
            group_all=False,
            if_bn=False,
            if_idx=True
        )
        self.transformer_1 = PointTransformerBlock(128, dim=64)
        self.sa_module_2 = PointNetSAModuleKNN(
            128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True
        )
        self.transformer_2 = PointTransformerBlock(256, dim=64)
        self.sa_module_3 = PointNetSAModuleKNN(
            None, None, 256, [512, out_dim], group_all=True, if_bn=False
        )

    def forward(self, point_cloud):
        l0_xyz = point_cloud[:, 0:3, :].contiguous()
        if self.feat_channel != 3:
            l0_points = point_cloud[:, 3:, :].contiguous()
        else:
            l0_points = point_cloud[:, 0:3, :].contiguous()

        l1_xyz, l1_points, idx1 = self.sa_module_1(
            l0_xyz, l0_points
        )
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(
            l1_xyz, l1_points
        )
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(
            l2_xyz, l2_points
        )

        return l3_points


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MlpRes(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MlpRes(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MlpRes(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.ReLU(), nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        x1 = self.ps(feat)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(
            torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1)
        )
        completion = self.mlp_4(x3)
        return completion


class StructuralSemanticJoint(nn.Module):

    def __init__(self, data_channel=15, dim_feat=512):
        super(StructuralSemanticJoint, self).__init__()
        self.semantic_feature_mlp = MlpConv(in_channel=data_channel, layer_dims=[64, 128])
        self.semantic_query_mlp = MlpConv(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])
        self.structural_semantic_attention = StructuralSemanticAttention(
            in_channel=128, pos_channel=3, dim=64
        )

    def forward(self, pcd_prev, feat_global, structural_feature_prev=None):
        xyz = pcd_prev[:, 0:3, :].contiguous()

        f_sem = self.semantic_feature_mlp(pcd_prev)
        f_sem = torch.cat(
            [
                f_sem,
                torch.max(f_sem, 2, keepdim=True)[0].repeat((1, 1, f_sem.size(2))),
                feat_global.repeat(1, 1, f_sem.size(2)),
            ],
            1,
        )
        f_sem = self.semantic_query_mlp(f_sem)


        f_stru = structural_feature_prev if structural_feature_prev is not None else f_sem
        f_ss = self.structural_semantic_attention(xyz, f_stru, f_sem)
        return f_ss, f_sem


class AwarenessOffsetEstimation(nn.Module):

    def __init__(self, data_channel=15, up_factor=2, i=0, radius=1.0):
        super(AwarenessOffsetEstimation, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.point_split = nn.ConvTranspose1d(256, 256, up_factor, up_factor, bias=False)
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.offset_feature_mlp = MlpRes(in_dim=512, hidden_dim=236, out_dim=128)
        self.offset_attention = StructuralSemanticAttention(in_channel=128, pos_channel=3, dim=64)
        self.offset_regressor = MlpConv(in_channel=128, layer_dims=[64, data_channel])

    def forward(self, pcd_prev, f_ss, f_rs, f_sem):
        f_fusion = torch.cat([f_ss, f_rs], 1)
        f_fusion_up = self.up_sampler(f_fusion)
        f_split = self.point_split(f_fusion)
        f_offset = self.offset_feature_mlp(torch.cat([f_split, f_fusion_up], 1))

        pcd_child = self.up_sampler(pcd_prev)
        f_sem_up = self.up_sampler(f_sem)
        f_offset = self.offset_attention(
            pcd_child[:, :3, :].contiguous(), f_offset, f_sem_up
        )
        offset = torch.tanh(self.offset_regressor(torch.relu(f_offset))) / (self.radius ** self.i)
        pcd_refined = pcd_child + offset
        return pcd_refined, offset, f_offset


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1.0):
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MlpConv(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MlpConv(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = StructuralSemanticAttention(in_channel=128, pos_channel=3, dim=64)

        self.mlp_ps = MlpConv(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(
            32, 128, up_factor, up_factor, bias=False
        )

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MlpRes(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MlpConv(in_channel=128, layer_dims=[64, 3])
        self.feat_extract = FeatureExtractor(3, dim_feat)

    def forward(self, pcd_prev, feat_global, k_prev=None):
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat(
            [
                feat_1,
                torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                feat_global.repeat(1, 1, feat_1.size(2)),
            ],
            1,
        )
        query = self.mlp_2(feat_1)

        hidden = self.skip_transformer(
            pcd_prev, k_prev if k_prev is not None else query, query
        )

        feat_child = self.mlp_ps(hidden)
        feat_child = self.ps(feat_child)
        hidden_up = self.up_sampler(hidden)
        k_curr = self.mlp_delta_feature(
            torch.cat([feat_child, hidden_up], 1)
        )

        delta = torch.tanh(self.mlp_delta(torch.relu(k_curr))) / (
            self.radius**self.i
        )
        pcd_child = self.up_sampler(pcd_prev)
        pcd_coarse = pcd_child + delta

        feat_coarse = self.feat_extract(pcd_coarse)

        return (
            pcd_coarse,
            pcd_child,
            delta,
            k_curr,
            feat_coarse,
        )


class HierarchicalRegionStructureExtraction(nn.Module):

    def __init__(self, k_num, in_channel):
        super(HierarchicalRegionStructureExtraction, self).__init__()
        self.k = k_num

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.mlp_1 = MlpConv(in_channel=1024, layer_dims=[512, 128])

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)


        x = self.mlp_1(x)

        return x


class LocalRegionRefinement(nn.Module):

    def __init__(self, class_num, i):
        super(LocalRegionRefinement, self).__init__()
        self.cls = class_num
        self.radius = 0.2
        self.i = i

        if self.i == 0:
            self.n_agent = 96
        elif self.i == 1:
            self.n_agent = 96
        elif self.i == 2:
            self.n_agent = 64
        elif self.i == 3:
            self.n_agent = 32
        else:
            self.n_agent = 16
        self.n_knn = 20

        self.agent_knn = KNN(k=self.n_knn, transpose_mode=False)

        self.local_coordinate_mlp = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1),

            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
        )

        self.local_semantic_mlp = nn.Sequential(
            nn.Conv2d(self.cls, 32, kernel_size=1, stride=1),

            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
        )

        self.coordinate_offset_head = nn.Sequential(
            nn.Conv1d(64, 128, 1, 1),

            nn.ReLU(),
            nn.Conv1d(128, 64, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1, 1),
        )

        self.semantic_residual_head = nn.Sequential(
            nn.Conv1d(64, 128, 1, 1),

            nn.ReLU(),
            nn.Conv1d(128, 64, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, self.cls, 1, 1),
        )

    def forward(self, pcd_coarse, trans_cord, k_prev):
        b, _, n_pts = pcd_coarse.shape
        if self.i == 0:

            trans_dist = torch.sum((trans_cord.transpose(1, 2)) ** 2, 2)
            _, idx_agent = torch.topk(trans_dist, self.n_agent, largest=True)
        else:

            trans_dist = torch.sum((trans_cord.transpose(1, 2)) ** 2, 2)
            _, idx_tk = torch.topk(trans_dist, self.n_agent // 2, largest=True)
            _, idx_bk = torch.topk(trans_dist, self.n_agent // 2, largest=False)
            idx_agent = torch.cat([idx_tk, idx_bk], dim=1)


        pcd_agent = gather_operation(pcd_coarse.contiguous(), idx_agent.int())


        _, idx_agent_knn = self.agent_knn(pcd_coarse[:, 0:3, :], pcd_agent[:, 0:3, :])
        idx_agent_knn = idx_agent_knn.transpose(1, 2).contiguous().int()


        agent_cord_patch = grouping_operation(pcd_coarse[:, 0:3, :].contiguous(), idx_agent_knn)
        agent_label_patch = grouping_operation(pcd_coarse[:, 3:, :].contiguous(), idx_agent_knn)


        agent_cord_patch = agent_cord_patch - agent_cord_patch[:, :, :, 0].unsqueeze(3).repeat(1, 1, 1, self.n_knn)
        agent_label_patch = agent_label_patch - agent_label_patch[:, :, :, 0].unsqueeze(3).repeat(1, 1, 1, self.n_knn)


        agent_cord_patch_feat = self.local_coordinate_mlp(agent_cord_patch)
        agent_label_patch_feat = self.local_semantic_mlp(agent_label_patch)


        agent_cord_patch_feat = torch.max(agent_cord_patch_feat, dim=3)[0]
        agent_label_patch_feat = torch.max(agent_label_patch_feat, dim=3)[0]


        child_cmp = torch.tanh(self.coordinate_offset_head(torch.relu(agent_cord_patch_feat))) \
            * torch.tensor(self.radius)
        child_label = torch.tanh(self.semantic_residual_head(torch.relu(agent_label_patch_feat))) \
            * torch.tensor(self.radius)


        local_trans = torch.cat([child_cmp, child_label], dim=1)


        pcd_local = pcd_agent + local_trans

        pcd_local = torch.cat([pcd_coarse, pcd_local], dim=2)


        k_prev_agent = gather_operation(k_prev.contiguous(), idx_agent.int())

        k_prev = torch.cat([k_prev, k_prev_agent], dim=2)

        return pcd_local, k_prev


class LabelBranch(nn.Module):
    def __init__(self, class_num, k_num, dim_feat, up_factor=2, i=0, probability=0.9):
        super(LabelBranch, self).__init__()
        self.i = i
        self.prob = probability
        self.up_factor = up_factor
        self.cls = class_num
        self.mlp_1 = MlpConv(in_channel=self.cls, layer_dims=[64, 128])

        self.label_feat_extract = HierarchicalRegionStructureExtraction(k_num, self.cls)
        self.d_k = 1.0
        self.key_start = nn.Conv1d(dim_feat, dim_feat, 1)
        self.query_start = nn.Conv1d(dim_feat, dim_feat, 1)
        self.dropout = nn.Dropout(0.1)
        self.linear_end = nn.Conv1d(dim_feat, dim_feat, 1)
        self.mlp_2 = MlpConv(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])
        self.mlp_ps = MlpConv(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(
            32, 128, up_factor, up_factor, bias=False
        )
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.skip_transformer = StructuralSemanticAttention(
            in_channel=128, pos_channel=self.cls, dim=64
        )
        self.mlp_delta_feature = MlpRes(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp_delta = MlpConv(in_channel=128, layer_dims=[64, self.cls])

    def forward(self, pcd, feat_pcd, feat_cord, k_cmp):

        xyz = pcd[:, 0:3, :].contiguous()
        pcd_label = pcd[:, 3:, :].contiguous()
        b, _, n = pcd_label.shape
        parent_feat = self.mlp_1(pcd_label)


        _, idx = KNN(k=20, transpose_mode=True)(
            xyz.transpose(1, 2), xyz.transpose(1, 2)
        )
        idx = idx.int()

        label_nn = grouping_operation(pcd_label, idx)
        label_nn = label_nn.permute(0, 2, 3, 1)
        pcd_label_flip = pcd_label.transpose(1, 2).unsqueeze(2).repeat(1, 1, 20, 1)

        label_nn = (
            torch.cat([label_nn - pcd_label_flip, pcd_label_flip], dim=3)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        feat_label = self.label_feat_extract(label_nn)


        key = feat_label
        query = feat_cord
        value = feat_pcd

        feat_query = self.query_start(query)
        feat_key = self.key_start(key)
        attention = torch.matmul(feat_query, feat_key.transpose(1, 2)) / np.sqrt(
            self.d_k
        )
        attention = torch.softmax(attention, -1)
        attention = self.dropout(attention)
        value = value
        out = einsum("b d d, b d i -> b d i", attention, value)
        out = self.linear_end(out)
        feat_fuse = out


        parent_feat = torch.cat(
            [
                parent_feat,
                torch.max(parent_feat, 2, keepdim=True)[0].repeat(
                    1, 1, parent_feat.shape[2]
                ),
                feat_fuse.repeat(1, 1, parent_feat.shape[2]),
            ],
            dim=1,
        )
        parent_feat = self.mlp_2(parent_feat)
        child_feat = self.mlp_ps(parent_feat)
        child_feat_up = self.ps(child_feat)
        child_label_up = self.up_sampler(pcd_label)

        child_trans_feat = self.skip_transformer(child_label_up, child_feat_up, k_cmp)
        child_trans_feat = self.mlp_delta_feature(
            torch.cat([child_trans_feat, child_feat_up], 1)
        )
        child_trans = torch.tanh(self.mlp_delta(torch.relu(child_trans_feat))) / (
            self.prob**self.i
        )

        child_label = child_label_up + child_trans

        return child_label, child_label_up, child_trans, child_trans_feat


class EfficientSSC(nn.Module):

    def __init__(
        self,
        class_num,
        dim_feat=1024,
        num_p0=1024,
        radius=1,
        up_factors=(2, 2, 2),
        k_num=20,
    ):
        super(EfficientSSC, self).__init__()
        self.cls = class_num
        self.num_p0 = num_p0
        self.k_num = k_num

        if up_factors is None:
            self.up_factors = [1, 1]
        else:
            self.up_factors = [1, 1] + list(up_factors)

        self.feature_extractor = FeatureExtractor(feat_channel=3, out_dim=dim_feat)
        self.step_feature_extractor = FeatureExtractor(feat_channel=self.cls, out_dim=dim_feat)


        self.SSJ = nn.ModuleList(
            [
                StructuralSemanticJoint(data_channel=3 + self.cls, dim_feat=dim_feat)
                for _ in self.up_factors
            ]
        )
        self.HRSE = nn.ModuleList(
            [
                HierarchicalRegionStructureExtraction(k_num, self.cls + 3)
                for _ in self.up_factors
            ]
        )
        self.AOE = nn.ModuleList(
            [
                AwarenessOffsetEstimation(
                    data_channel=3 + self.cls,
                    up_factor=factor,
                    i=i,
                    radius=radius,
                )
                for i, factor in enumerate(self.up_factors)
            ]
        )

        self.local_refiners = nn.ModuleList(
            [LocalRegionRefinement(class_num, i=i) for i, _ in enumerate(self.up_factors)]
        )
        self.hrse_knn = KNN(k=k_num, transpose_mode=True)

        self.initial_semantic_head = nn.Sequential(
            nn.Conv1d(dim_feat + 3, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, self.cls, kernel_size=1),
        )

    def _initialize_labeled_points(self, point_cloud, feat_global):
        b, n, _ = point_cloud.shape
        point_cloud_bcn = point_cloud.permute(0, 2, 1).contiguous()
        semantic_logits = self.initial_semantic_head(
            torch.cat((point_cloud_bcn, feat_global.repeat(1, 1, n)), dim=1)
        )
        return torch.cat([point_cloud, semantic_logits.transpose(1, 2)], dim=2)

    def _sample_seed_points(self, labeled_points):
        labeled_points_bcn = labeled_points.transpose(1, 2).contiguous()
        return gather_operation(
            labeled_points_bcn,
            furthest_point_sample(
                labeled_points_bcn[:, 0:3, :].transpose(1, 2).contiguous(),
                self.num_p0,
            ),
        )

    def _build_hrse_input(self, pcd):
        xyz = pcd[:, 0:3, :].contiguous()
        _, idx = self.hrse_knn(xyz.transpose(1, 2), xyz.transpose(1, 2))
        idx = idx.int()
        neighbor_feature = grouping_operation(pcd.contiguous(), idx)
        neighbor_feature = neighbor_feature.permute(0, 2, 3, 1)
        center_feature = pcd.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.k_num, 1)
        hrse_input = (
            torch.cat([neighbor_feature - center_feature, center_feature], dim=3)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return hrse_input

    def _resample_progressive_state(self, pcd, structural_feature):
        progressive_state = torch.cat((pcd, structural_feature), dim=1)
        progressive_state = gather_operation(
            progressive_state.contiguous(),
            furthest_point_sample(
                progressive_state[:, 0:3, :].transpose(1, 2).contiguous(),
                self.num_p0,
            ),
        )
        return (
            progressive_state[:, : 3 + self.cls, :],
            progressive_state[:, 3 + self.cls :, :],
        )

    def forward(self, point_cloud):
        point_cloud_bcn = point_cloud.permute(0, 2, 1).contiguous()
        feat_global = self.feature_extractor(point_cloud_bcn)
        pcd_labeled = self._initialize_labeled_points(point_cloud, feat_global)
        pcd = self._sample_seed_points(pcd_labeled)

        outputs = [pcd.permute(0, 2, 1).contiguous()]
        structural_feature_prev = None
        step_feature = feat_global

        for i, (ssj, hrse, aoe) in enumerate(zip(self.SSJ, self.HRSE, self.AOE)):
            f_ss, f_sem = ssj(pcd, step_feature, structural_feature_prev)
            f_rs = hrse(self._build_hrse_input(pcd))
            pcd, offset, structural_feature_prev = aoe(pcd, f_ss, f_rs, f_sem)

            if i < len(self.AOE) - 1:
                pcd, structural_feature_prev = self.local_refiners[i](
                    pcd, offset[:, :3, :], structural_feature_prev
                )

            if i < 2:
                pcd, structural_feature_prev = self._resample_progressive_state(
                    pcd, structural_feature_prev
                )

            step_feature = self.step_feature_extractor(pcd)
            outputs.append(pcd.permute(0, 2, 1).contiguous())

        return outputs

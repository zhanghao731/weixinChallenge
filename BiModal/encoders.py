import torch
import torch.nn as nn
from .blocks import LayerStack, PositionwiseFeedForward, ResidualConnection, clone
from .multihead_attention import MultiheadedAttention


class BiModalEncoderLayer(nn.Module):

    def __init__(self, d_model_M1, d_model_M2, d_model, dout_p, H, d_ff_M1, d_ff_M2):
        super(BiModalEncoderLayer, self).__init__()
        self.self_att_M1 = MultiheadedAttention(d_model_M1, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.self_att_M2 = MultiheadedAttention(d_model_M2, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M1 = MultiheadedAttention(d_model_M1, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M2 = MultiheadedAttention(d_model_M2, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.feed_forward_M1 = PositionwiseFeedForward(d_model_M1, d_ff_M1, dout_p)
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_M2, d_ff_M2, dout_p)
        self.res_layers_M1 = clone(ResidualConnection(d_model_M1, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_M2, dout_p), 3)

    def forward(self, x, masks):
        '''
        Inputs:
            x (M1, M2): (B, Sm, Dm)
            masks (M1, M2): (B, 1, Sm)
        Output:
            M1m2 (B, Sm1, Dm1), M2m1 (B, Sm2, Dm2),
        '''
        M1, M2 = x
        M1_mask, M2_mask = masks

        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)

        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)

        def sublayer_att_M1(M1): return self.bi_modal_att_M1(M1, M2, M2, M2_mask)

        def sublayer_att_M2(M2): return self.bi_modal_att_M2(M2, M1, M1, M1_mask)

        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        # 1. Self-Attention
        # both (B, Sm*, Dm*)
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)

        # 2. Multimodal Attention (var names: M* is the target modality; m* is the source modality)
        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)

        # 3. Feed-forward (var names: M* is the target modality; m* is the source modality)
        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[2](M1m2, sublayer_ff_M1)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)

        return M1m2, M2m1


class BiModalEncoder(nn.Module):

    def __init__(self, d_model_V, d_model_T, d_model, dout_p, H, d_ff_V, d_ff_T, N):
        super(BiModalEncoder, self).__init__()
        layer_VT = BiModalEncoderLayer(d_model_V, d_model_T, d_model, dout_p, H, d_ff_V, d_ff_T)
        self.encoder_VT = LayerStack(layer_VT, N)

    def forward(self, x, masks: dict):
        '''
        Input:
            x (A, V): (B, Sm, D)
            masks: {V_mask: (B, 1, Sv); A_mask: (B, 1, Sa)}
        Output:
            (Av, Va): (B, Sm1, Dm1)
        '''
        V, T = x

        # M1m2 (B, Sm1, D), M2m1 (B, Sm2, D) <-
        Vt, Tv = self.encoder_VT((V, T), (masks['V_mask'], masks['T_mask']))

        return Vt, Tv


class EncoderLayer(nn.Module):
    def __init__(self, feature_size, dropout, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadedAttention(feature_size, feature_size, feature_size, H, dropout, feature_size)
        self.feed_forward = PositionwiseFeedForward(feature_size, d_ff, dropout)
        self.residual_layers = clone(ResidualConnection(feature_size, dropout), 2)

    def forward(self, x, mask):
        x = self.residual_layers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.residual_layers[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, feature_size, dropout, H, d_ff, N):
        super(Encoder, self).__init__()
        encoderLayer = EncoderLayer(feature_size, dropout, H, d_ff)
        self.layers = LayerStack(encoderLayer, N)
        self.norm = nn.LayerNorm(feature_size)

    def forward(self, x, mask):
        x = self.layers(x, mask)
        x = self.norm(x)
        # mean pooling
        x = (x * mask.squeeze(1).unsqueeze(-1)).sum(1) / (mask.sum(2) + 1e-9)
        return x

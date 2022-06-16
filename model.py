import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from BiModal.encoders import BiModalEncoder, Encoder
from BiModal.blocks import PositionalEncoder
from category_id_map import CATEGORY_ID_LIST


# class MultiModal(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
#         self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
#                                  output_size = args.vlad_hidden_size, dropout = args.dropout)
#         self.enhance = SENet(channels = args.vlad_hidden_size, ratio = args.se_ratio)
#         bert_output_size = 768
#         self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
#         self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))
#
#     def forward(self, inputs, inference = False, focal_loss = False):
#         # bert_embedding = self.bert(inputs['text_input'], inputs['text_mask'])['pooler_output']
#         #  mean pooling
#         bert_embedding = self.bert(inputs['text_input'], inputs['text_mask'])['last_hidden_state']
#         bert_embedding = (bert_embedding * inputs['text_mask'].unsqueeze(-1)).sum(1) / (inputs['text_mask'].sum(
#             1).unsqueeze(-1) + 1e-9)
#         vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
#         vision_embedding = self.enhance(vision_embedding)
#
#         final_embedding = self.fusion([vision_embedding, bert_embedding])
#         prediction = self.classifier(final_embedding)
#
#         if inference:
#             return torch.argmax(prediction, dim = 1)
#         else:
#             return self.cal_loss(prediction, inputs['label'], focal_loss)
#
#     @staticmethod
#     def cal_loss(prediction, label, focal_loss):
#         label = label.squeeze(dim = 1)
#         if not focal_loss:
#             loss = F.cross_entropy(prediction, label)
#         else:
#             alpha = torch.tensor(0.25).to(prediction.device)
#             gamma = 2.0
#             smooth = 1e-4
#             prob = F.softmax(prediction, dim = 1)
#             target = label.view(-1, 1)
#             prob = prob.gather(1, target).view(-1) + smooth
#             logpt = torch.log(prob)
#             loss = -alpha * torch.pow(torch.sub(1.0, prob), gamma) * logpt
#             loss = torch.mean(loss)
#         with torch.no_grad():
#             pred_label_id = torch.argmax(prediction, dim = 1)
#             accuracy = (label == pred_label_id).float().sum() / label.shape[0]
#         return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size = 1024, expansion = 2, groups = 8, dropout = 0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias = False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std = 0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask = None):
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)  # [B,P,G]
        # mask
        if mask is not None:
            attention = torch.mul(attention, mask.unsqueeze(2))
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])  # [B,P*G,1]
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])  # [B*P,lamda*N]
        activation = self.cluster_linear(reshaped_input)  # [B*P,G*K]
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])  # [B,P*G,K]
        activation = torch.softmax(activation, dim = -1)  #
        activation = activation * attention  # [B,P*G,K]权重矩阵
        a_sum = activation.sum(-2, keepdim = True)  # [B,1,K]
        a = a_sum * self.cluster_weight  # [B,lamda*N/G,K]
        activation = activation.permute(0, 2, 1).contiguous()  # [B,K,P*G]
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])  # [B,P*G,lamda*N/G]
        vlad = torch.matmul(activation, reshaped_input)  # [B,K,lamda*N/G]
        vlad = vlad.permute(0, 2, 1).contiguous()  # [B,lamda*N/G,K]
        vlad = F.normalize(vlad - a, p = 2, dim = 1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])  # [B,vec]
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)  # [B,outputsize]
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio = 8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features = channels, out_features = channels // ratio, bias = False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels = hidden_size, ratio = se_ratio)
        # self.eca = eca_layer()

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim = 1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)
        # embedding = self.eca(embedding)
        return embedding


class eca_layer(nn.Module):
    def __init__(self, k_size = 3):
        super(eca_layer, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size = (k_size,), padding = (k_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x.unsqueeze(1))
        y = self.sigmoid(y.squeeze(1))
        return x * y


class outputLayer(nn.Module):
    def __init__(self, feature_size, cluster_size, se_ratio, fc_size, output_size = 1024, expansion = 2, groups = 8,
                 dropout = 0.2):
        super().__init__()
        # self.nextvlad = NeXtVLAD(feature_size, cluster_size, output_size, expansion, groups, dropout)
        self.drop = nn.Dropout(dropout)
        # self.bottleneck = nn.Linear(output_size, fc_size)
        # self.se_gate = SENet(feature_size, se_ratio)
        self.classifier = nn.Linear(feature_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, mask):
        # embeddings = self.nextvlad(inputs, mask)
        # embeddings = self.drop(embeddings)
        # embeddings = self.bottleneck(embeddings)
        # embeddings = self.se_gate(embeddings)
        embedding_mean = (inputs * mask.unsqueeze(-1)).sum(1) / (mask.sum(
            1).unsqueeze(-1) + 1e-9)
        # embedding_mean = self.se_gate(embedding_mean)
        embeddings = self.drop(embedding_mean)
        output = self.classifier(embeddings)
        return output


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        bert_output_size = 768
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        self.pos_enc_v = PositionalEncoder(args.frame_embedding_size, args.dropout)
        self.encoder = BiModalEncoder(args.frame_embedding_size, bert_output_size, bert_output_size, args.dropout,
                                      args.H, args.d_ff_v, args.d_ff_t, args.N)
        self.cross_modal_encoder = Encoder(bert_output_size, args.dropout, args.H, args.d_ff_v, 2)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.cross_modal_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.outputLayer_v = outputLayer(args.frame_embedding_size, args.vlad_cluster_size, args.se_ratio, args.fc_size,
        #                                  output_size = args.vlad_hidden_size, dropout = args.dropout)
        # self.fusion = ConcatDenseSE(args.frame_embedding_size + bert_output_size, args.fc_size, args.se_ratio,
        #                             args.dropout)
        # self.outputLayer_t = outputLayer(bert_output_size, args.vlad_cluster_size, args.se_ratio, args.fc_size,
        #                                  output_size = args.vlad_hidden_size, dropout = args.dropout)
        # self.se = SENet(bert_output_size, args.se_ratio)
        self.drop = nn.Dropout(args.dropout)
        self.outputLayer_v = outputLayer(args.frame_embedding_size, args.vlad_cluster_size, args.se_ratio, args.fc_size,
                                         dropout = args.dropout)
        self.outputLayer_t = outputLayer(bert_output_size, args.vlad_cluster_size, args.se_ratio, args.fc_size,
                                         dropout = args.dropout)
        self.classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))

        # 冻结bert前九层参数
        unfreeze_layer = ['layer.9', 'layer.10', 'layer.11']
        for n, p in self.bert.encoder.named_parameters():
            p.requires_grad = False
            if any(ele in n for ele in unfreeze_layer):
                p.requires_grad = True

    def forward(self, inputs, inference = False):
        bert_embedding = self.bert(inputs['text_input'], inputs['text_mask'])['last_hidden_state']
        vision_embedding = self.pos_enc_v(inputs['frame_input'])
        masks = dict()
        masks['V_mask'] = inputs['frame_mask'].unsqueeze(1)
        masks['T_mask'] = inputs['text_mask'].unsqueeze(1)
        Vt, Tv = self.encoder((vision_embedding, bert_embedding), masks)
        output_v = self.outputLayer_v(Vt, inputs['frame_mask'])
        output_t = self.outputLayer_t(Tv, inputs['text_mask'])
        fusion_embedding = torch.cat((Vt, Tv), dim = 1)
        mask = torch.cat((inputs['frame_mask'], inputs['text_mask']), dim = 1).unsqueeze(1)

        # output_v = self.outputLayer_v(Vt, inputs['frame_mask'])
        # output_t = self.outputLayer_t(Tv, inputs['text_mask'])
        # Vt = (Vt * inputs['frame_mask'].unsqueeze(-1)).sum(1) / (inputs['frame_mask'].sum(
        #     1).unsqueeze(-1) + 1e-9)
        # Tv = (Tv * inputs['text_mask'].unsqueeze(-1)).sum(1) / (inputs['text_mask'].sum(
        #     1).unsqueeze(-1) + 1e-9)
        # Tv = self.se(Tv)
        # Tv = self.drop(Tv)
        # output_t = self.classifier_text(Tv)
        # fusion_embedding = self.fusion([Vt, Tv])
        fusion_embedding = self.cross_modal_encoder(fusion_embedding, mask)
        fusion_embedding = self.drop(fusion_embedding)
        output = self.classifier(fusion_embedding)
        if inference:
            prob_v = F.softmax(output_v, dim = 1)
            prob_t = F.softmax(output_t, dim = 1)
            prob_fusion = F.softmax(output, dim = 1)
            prob_final = (prob_v + prob_t + prob_fusion) / 3.0
            return torch.argmax(prob_final, dim = 1)
            # return torch.argmax(output, dim = 1)
        else:
            return self.cal_loss(output_v, output_t, output, inputs['label'])

    @staticmethod
    def cal_loss(output_v, output_t, output_fusion, label):
        label = label.squeeze(1)
        # loss = F.nll_loss(torch.log(prob), label)
        loss_v = F.cross_entropy(output_v, label)
        loss_t = F.cross_entropy(output_t, label)
        loss_fusion = F.cross_entropy(output_fusion, label)
        loss = 0.3 * loss_v + 0.3 * loss_t + 0.4 * loss_fusion
        # loss = F.cross_entropy(pred, label)
        with torch.no_grad():
            prob_v = F.softmax(output_v, dim = 1)
            prob_t = F.softmax(output_t, dim = 1)
            prob_fusion = F.softmax(output_fusion, dim = 1)
            prob_final = (prob_v + prob_t + prob_fusion) / 3.0
            pred_label_id = torch.argmax(prob_final, dim = 1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

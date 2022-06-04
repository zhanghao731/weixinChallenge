import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        args.out_size = self.bert_config.hidden_size

        # 将frame feature变换到与text feature相同的维度
        self.vision_fc = nn.Linear(args.frame_embedding_size, self.bert_config.hidden_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.bert_config.hidden_size)
        self.drop_out = nn.Dropout(args.dropout)
        # self.enhance = SENet(channels = args.frame_embedding_size, ratio = args.se_ratio)
        # self.relu = nn.ReLU()
        # self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
        #                          output_size = args.vlad_hidden_size, dropout = args.dropout)
        # self.enhance = SENet(channels = args.vlad_hidden_size, ratio = args.se_ratio)
        # bert_output_size = 768
        # self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        # self.classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
        # self.classifier = MLP(args)
        # self.classifier = nn.Linear(self.bert_config.hidden_size, len(CATEGORY_ID_LIST))
        self.classifier = ClassificationHead(args)

    def forward(self, inputs, inference = False):
        # bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])['pooler_output']
        text_embedding = self.bert.embeddings(input_ids = inputs['text_input'])
        vision_embedding = self.relu(self.vision_fc(inputs['frame_input']))
        vision_embedding = self.bert.embeddings(inputs_embeds = vision_embedding)

        # 拼接text和vision embedding
        # fusion_embedding = torch.cat((text_embedding, vision_embedding), dim = 1)
        cls_emb = text_embedding[:, 0:1, :]
        text_emb = text_embedding[:, 1:, :]

        cls_mask = inputs['text_mask'][:, 0:1]
        text_mask = inputs['text_mask'][:, 1:]
        fusion_embedding = torch.cat([cls_emb, vision_embedding, text_emb], dim = 1)

        # 拼接attention_mask
        fusion_attention_mask = torch.cat([cls_mask, inputs['frame_mask'], text_mask], dim = 1)
        attention_mask = fusion_attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * (-10000.0)

        encoder_outputs = self.bert.encoder(fusion_embedding, attention_mask = attention_mask)['last_hidden_state']
        # encoder_outputs_mean = (encoder_outputs * fusion_attention_mask.unsqueeze(-1)).sum(
        #     1) / fusion_attention_mask.sum(1).unsqueeze(-1)
        encoder_outputs_mean = torch.mean(encoder_outputs, dim = 1)
        encoder_outputs_mean = encoder_outputs_mean.float()
        # encoder_outputs_max=encoder_outputs+(1-fusion_attention_mask).unsqueeze(-1)*(-1e10)
        # encoder_outputs_max=encoder_outputs_max.max(1)[0].float()
        # final_hidden_state=torch.cat((encoder_outputs_mean,encoder_outputs_max),dim = -1)
        # encoder_outputs_mean = self.bn(encoder_outputs_mean)
        # encoder_outputs_mean = self.drop_out(encoder_outputs_mean)
        prediction = self.classifier(encoder_outputs_mean)

        # vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        # vision_embedding = self.enhance(vision_embedding)

        # final_embedding = self.fusion([vision_embedding, bert_embedding])

        if inference:
            return torch.argmax(prediction, dim = 1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim = 1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim = 1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


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

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)  # [B,P,G]
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

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim = 1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class ClassificationHead(nn.Module):
    def __init__(self, args):
        super(ClassificationHead, self).__init__()
        self.norm_1 = nn.BatchNorm1d(args.out_size)
        # self.linear_1 = nn.Linear(args.out_size, 1024)
        # self.norm_2 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(args.dropout)
        self.linear_2 = nn.Linear(768, 256)
        self.norm_3 = nn.BatchNorm1d(256)
        self.linear_out = nn.Linear(256, len(CATEGORY_ID_LIST))

    def forward(self, features):
        x = self.norm_1(features)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = torch.relu(self.norm_3(x))
        x = self.dropout(x)
        # x=self.linear_2(x)
        # x=torch.relu(self.norm_3(x))
        # x=self.dropout(x)
        x = self.linear_out(x)
        return x


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.bottleneck = nn.Linear(args.out_size, 256)
        self.norm = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, len(CATEGORY_ID_LIST))

    def forward(self, features):
        x = self.bottleneck(features)
        x = self.norm(x)
        x = self.classifier(x)
        return x

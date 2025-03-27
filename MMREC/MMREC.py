import logging
import torch
from transformers import BertTokenizer
import transformers
from torch.nn import functional as F
from torch import nn

class LLMSummaryProcessor(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(LLMSummaryProcessor, self).__init__()
        # 添加输入投影层，将实体表示降维
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, entity_repr, llm_summary_repr):
        # 对输入进行降维处理
        projected_entity = self.input_projection(entity_repr)

        # 使用注意力机制
        attn_output, _ = self.attention(
            query=projected_entity.unsqueeze(1),
            key=llm_summary_repr.unsqueeze(1),
            value=llm_summary_repr.unsqueeze(1)
        )

        # 归一化并投影
        enhanced_repr = self.norm(attn_output.squeeze(1) + projected_entity)
        return self.output_projection(enhanced_repr)


class UnifiedRepresentationSpace(nn.Module):
    def __init__(self, text_dim, knowledge_dim, output_dim):
        super(UnifiedRepresentationSpace, self).__init__()

        # 统一投影层
        self.text_projector = nn.Linear(text_dim, output_dim)
        self.knowledge_projector = nn.Linear(knowledge_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, knowledge_features):
        # 投影到统一空间

        unified = self.knowledge_projector(knowledge_features)


        return unified


class DLRMStylePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DLRMStylePredictor, self).__init__()

        # 上游MLP
        self.upstream_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 下游MLP
        self.downstream_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # 上游MLP降维
        hidden = self.upstream_mlp(x)

        # 下游MLP预测
        output = self.downstream_mlp(hidden)
        return output


class WeightedFocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        # 额外增加对假阳性的惩罚
        # 假阳性：预测为正(非0类)但实际为负(0类)
        pred_classes = torch.argmax(inputs, dim=1)
        false_positive_mask = (pred_classes != 0) & (targets == 0)
        loss[false_positive_mask] = loss[false_positive_mask] * 2.0

        return loss.mean()



class MMREC(nn.Module):
    def __init__(self, args, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.hidden_size = 768
        self.args = args

        # 1. 文本编码器
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained('/root/autodl-tmp/TMR/Model/bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/TMR/Model/bert-base-uncased")
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[unused0]', '[unused1]', '[unused2]', '[unused3]']})

        # 2. LLM摘要处理
        # self.llm_summary_processor = LLMSummaryProcessor(self.hidden_size)
        self.llm_summary_processor = LLMSummaryProcessor(
            input_dim=self.hidden_size * 2,  # 实体对拼接后的维度
            hidden_size=self.hidden_size  # 目标隐藏层维度
        )

        # 3. 知识图谱编码器(保留原有)
        from ..CK_encoder import RECK
        self.reck_encoder = RECK(128, "/root/autodl-tmp/TMR/Model/bert-base-uncased")

        # 4. 统一表示空间
        self.unified_space = UnifiedRepresentationSpace(
            text_dim=self.hidden_size * 3,  # 实体对+LLM摘要
            knowledge_dim=self.hidden_size * 5,  # 知识图谱特征
            output_dim=self.hidden_size
        )

        # 5. DLRM风格预测器
        self.relation_predictor = DLRMStylePredictor(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size // 2,
            output_dim=23
        )

        # 6. 图像描述编码器
        self.phrase_encoder = nn.Linear(self.hidden_size*3, self.hidden_size)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()



    def forward(self, token, att_mask, pos1, pos2, token_phrase, att_mask_phrase, token_qwen, att_mask_qwen, pic, A, W, A_rev, W_rev):
        output_text = self.bert(token, attention_mask=att_mask)
        hidden_text = output_text[0]

        # 2. 提取实体表示
        onehot_head = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)
        onehot_tail = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden_text).sum(1)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden_text).sum(1)
        entity_pair_repr = torch.cat([head_hidden, tail_hidden], dim=-1)

        # 3. 处理图像描述(类似MMREC处理图像)
        output_phrases = self.bert(token_phrase, attention_mask=att_mask_phrase)
        hidden_phrases = output_phrases[0]
        image_repr = torch.mean(hidden_phrases, dim=1)

        output_qwen1 = self.bert(token_qwen, attention_mask=att_mask_qwen)
        hidden_qwen1 = output_qwen1[0]
        llm_summary_repr1 = torch.mean(hidden_qwen1, dim=1)
        hidden_text1 = torch.mean(hidden_text, dim=1)
        new_image_repr = torch.cat([llm_summary_repr1, hidden_text1, image_repr], dim=1)
        image_repr = self.phrase_encoder(new_image_repr)

        # 4. 处理LLM总结(MMREC核心思想)
        output_qwen = self.bert(token_qwen, attention_mask=att_mask_qwen)
        hidden_qwen = output_qwen[0]
        llm_summary_repr = torch.mean(hidden_qwen, dim=1)

        enhanced_entity_repr = self.llm_summary_processor(entity_pair_repr, llm_summary_repr)

        # 6. 文本模态组合(MMREC风格)
        text_repr = torch.cat([
            enhanced_entity_repr,  # 增强的实体表示
            image_repr,  # 图像描述表示
            llm_summary_repr  # LLM总结表示
        ], dim=1)


        reck_hidden_entity, W_final = self.reck_encoder(token, att_mask, pos1, pos2, pic, A, W, A_rev, W_rev)

        # 8. 知识模态组合
        knowledge_repr = torch.cat([
            entity_pair_repr,
            image_repr,
            W_final
        ], dim=1)

        unified_repr = self.unified_space(knowledge_repr)

        logits = self.relation_predictor(unified_repr)

        return logits, unified_repr





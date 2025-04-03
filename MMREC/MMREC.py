import logging
import torch
from transformers import BertTokenizer
import transformers
from torch.nn import functional as F
from torch import nn

class LLMSummaryProcessor(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(LLMSummaryProcessor, self).__init__()
        # Add input projection layer to reduce entity representation dimensions
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, entity_repr, llm_summary_repr):
        # Dimension reduction processing for the input
        projected_entity = self.input_projection(entity_repr)

        # Use attention mechanism
        attn_output, _ = self.attention(
            query=projected_entity.unsqueeze(1),
            key=llm_summary_repr.unsqueeze(1),
            value=llm_summary_repr.unsqueeze(1)
        )

        # Normalize and project
        enhanced_repr = self.norm(attn_output.squeeze(1) + projected_entity)
        return self.output_projection(enhanced_repr)


class UnifiedRepresentationSpace(nn.Module):
    def __init__(self, text_dim, knowledge_dim, output_dim):
        super(UnifiedRepresentationSpace, self).__init__()

        # Unified projection layer
        self.text_projector = nn.Linear(text_dim, output_dim)
        self.knowledge_projector = nn.Linear(knowledge_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, knowledge_features):
        # Project to unified space

        unified = self.knowledge_projector(knowledge_features)


        return unified


class DLRMStylePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DLRMStylePredictor, self).__init__()

        # Upstream MLP  
        self.upstream_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Downstream MLP
        self.downstream_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # Upstream MLP dimensionality reduction
        hidden = self.upstream_mlp(x)

        # Downstream MLP prediction
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

        # Additional penalty for false positives
        # False positives: predicted as positive (non-zero class) but actually negative (class 0)
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

        # 1. Text Encoder
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained('/root/autodl-tmp/TMR/Model/bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/TMR/Model/bert-base-uncased")
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[unused0]', '[unused1]', '[unused2]', '[unused3]']})

        # 2. LLM Summary Processing
        # self.llm_summary_processor = LLMSummaryProcessor(self.hidden_size)
        self.llm_summary_processor = LLMSummaryProcessor(
            input_dim=self.hidden_size * 2,  
            hidden_size=self.hidden_size  
        )

        # 3. Knowledge Encoder
        from ..CK_encoder import RECK
        self.reck_encoder = RECK(128, "/root/autodl-tmp/TMR/Model/bert-base-uncased")

        # 4. Unified Representation Space
        self.unified_space = UnifiedRepresentationSpace(
            text_dim=self.hidden_size * 3, 
            knowledge_dim=self.hidden_size * 5,  
            output_dim=self.hidden_size
        )

        # 5. DLRM-style Predictor
        self.relation_predictor = DLRMStylePredictor(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size // 2,
            output_dim=23
        )

        # 6. Image Description Encoder
        self.phrase_encoder = nn.Linear(self.hidden_size*3, self.hidden_size)

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()



    def forward(self, token, att_mask, pos1, pos2, token_phrase, att_mask_phrase, token_qwen, att_mask_qwen, pic, A, W, A_rev, W_rev):
        output_text = self.bert(token, attention_mask=att_mask)
        hidden_text = output_text[0]

        onehot_head = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)
        onehot_tail = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden_text).sum(1)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden_text).sum(1)
        entity_pair_repr = torch.cat([head_hidden, tail_hidden], dim=-1)

        output_phrases = self.bert(token_phrase, attention_mask=att_mask_phrase)
        hidden_phrases = output_phrases[0]
        image_repr = torch.mean(hidden_phrases, dim=1)

        output_qwen1 = self.bert(token_qwen, attention_mask=att_mask_qwen)
        hidden_qwen1 = output_qwen1[0]
        llm_summary_repr1 = torch.mean(hidden_qwen1, dim=1)
        hidden_text1 = torch.mean(hidden_text, dim=1)
        new_image_repr = torch.cat([llm_summary_repr1, hidden_text1, image_repr], dim=1)
        image_repr = self.phrase_encoder(new_image_repr)

        output_qwen = self.bert(token_qwen, attention_mask=att_mask_qwen)
        hidden_qwen = output_qwen[0]
        llm_summary_repr = torch.mean(hidden_qwen, dim=1)

        enhanced_entity_repr = self.llm_summary_processor(entity_pair_repr, llm_summary_repr)

        text_repr = torch.cat([
            enhanced_entity_repr,  
            image_repr, 
            llm_summary_repr  
        ], dim=1)


        reck_hidden_entity, W_final = self.reck_encoder(token, att_mask, pos1, pos2, pic, A, W, A_rev, W_rev)

        knowledge_repr = torch.cat([
            entity_pair_repr,
            image_repr,
            W_final
        ], dim=1)

        unified_repr = self.unified_space(knowledge_repr)

        logits = self.relation_predictor(unified_repr)

        return logits, unified_repr





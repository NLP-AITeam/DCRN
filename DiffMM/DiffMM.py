import logging
import torch
from transformers import BertTokenizer
import transformers
from torch.nn import functional as F
from torch import nn


class DiffMM(nn.Module):
    def __init__(self, args, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        max_length: max length of sentence
        pretrain_path: path of pretrain model
        blank_padding: need padding or not
        mask_entity: mask the entity tokens or not
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.hidden_size = 768 * 2

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained(
            '/root/autodl-tmp/TMR/Model/bert-base-uncased')  # get the pre-trained BERT model for the text
        self.tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/TMR/Model/bert-base-uncased")
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[unused0]', '[unused1]', '[unused2]', '[unused3]']})
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_pic = nn.Linear(768, self.hidden_size // 2)
        self.linear_final = nn.Linear(self.hidden_size * 3, self.hidden_size // 2)

        # the attention mechanism for fine-grained features
        self.linear_q_fine = nn.Linear(768, self.hidden_size // 2)
        self.linear_k_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_k_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        self.linear_weights = nn.Linear(self.hidden_size * 3, 3)
        self.linear_phrases = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.linear_qwen = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.linear_extend_pic = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.dropout_linear = nn.Dropout(0.46)

        # 新增
        self.args = args

        self.linear_final1 = nn.Linear(self.hidden_size, 768)
        self.linear_x = nn.Linear(self.hidden_size, self.hidden_size)

        from ..CK_encoder import RECK
        self.reck_encoder = RECK(128, "/root/autodl-tmp/TMR/Model/bert-base-uncased")


        self.diffusion_steps = args.diffusion_steps
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max


        self.register_buffer('betas', torch.linspace(self.beta_min, self.beta_max, self.diffusion_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        self.msi_text = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.msi_image = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.msi_fusion = nn.Linear(self.hidden_size, self.hidden_size // 2)

        self.denoise_net = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size * 2),  # +1 是时间步嵌入
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )


        self.modal_weight_text = nn.Parameter(torch.ones(1))
        self.modal_weight_image = nn.Parameter(torch.ones(1))

        self.proj_text = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.proj_image = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.temperature = args.temperature

    def diffusion_forward_process(self, x_0, t):
        """前向扩散过程：逐步添加噪声"""
        # 确保 t 在正确的设备上
        t = t.to(x_0.device)

        noise = torch.randn_like(x_0)

        alphas_cumprod = self.alphas_cumprod.to(x_0.device)

        alpha_cumprod_t = alphas_cumprod[t]
        if len(alpha_cumprod_t.shape) == 0:
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(0).repeat(x_0.shape[0])
        alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
        x_t = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
        return x_t, noise

    def diffusion_reverse_process(self, x_t, t, modality_cond=None):
        """反向扩散过程：去噪"""

        t = t.to(x_t.device)
        t_emb = t.float() / self.diffusion_steps  # [batch_size]
        t_emb = t_emb.unsqueeze(-1)  # [batch_size, 1]


        if modality_cond is not None:
            input_denoise = torch.cat([x_t, t_emb], dim=1)
        else:
            input_denoise = torch.cat([x_t, t_emb], dim=1)

        # 预测噪声
        pred_noise = self.denoise_net(input_denoise)


        betas = self.betas.to(x_t.device)
        alphas = self.alphas.to(x_t.device)
        alphas_cumprod = self.alphas_cumprod.to(x_t.device)


        beta_t = betas[t]
        if len(beta_t.shape) == 0:
            beta_t = beta_t.unsqueeze(0).repeat(x_t.shape[0])
        beta_t = beta_t.unsqueeze(-1)

        alpha_t = alphas[t]
        if len(alpha_t.shape) == 0:
            alpha_t = alpha_t.unsqueeze(0).repeat(x_t.shape[0])
        alpha_t = alpha_t.unsqueeze(-1)

        alpha_cumprod_t = alphas_cumprod[t]
        if len(alpha_cumprod_t.shape) == 0:
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(0).repeat(x_t.shape[0])
        alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)

        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        mean = (x_t - beta_t * pred_noise / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)


        if self.training:
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(beta_t) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1, x_0_pred, pred_noise

    def modality_signal_injection(self, text_feats, image_feats):
        """模态感知信号注入"""
        text_proj = self.msi_text(text_feats)
        image_proj = self.msi_image(image_feats)


        fusion = torch.cat([text_proj, image_proj], dim=-1)
        modality_signal = self.msi_fusion(fusion)

        return modality_signal

    def cross_modal_contrastive_loss(self, text_feats, image_feats, batch_size):
        """跨模态对比学习损失"""
        text_proj = self.proj_text(text_feats)
        image_proj = self.proj_image(image_feats)


        text_proj = F.normalize(text_proj, dim=-1)
        image_proj = F.normalize(image_proj, dim=-1)

        logits = torch.matmul(text_proj, image_proj.transpose(0, 1)) / self.temperature

        labels = torch.arange(batch_size, device=logits.device)

        loss_t2i = F.cross_entropy(logits, labels)

        loss_i2t = F.cross_entropy(logits.t(), labels)

        cl_loss = (loss_t2i + loss_i2t) / 2

        return cl_loss

    def multi_modal_graph_aggregation(self, text_feats, image_feats):
        """多模态图聚合"""
        # 使用可学习的模态权重
        text_weight = F.softplus(self.modal_weight_text)
        image_weight = F.softplus(self.modal_weight_image)

        # 归一化权重
        sum_weights = text_weight + image_weight
        text_weight = text_weight / sum_weights
        image_weight = image_weight / sum_weights


        if not hasattr(self, 'proj_image_to_text_dim'):
            self.proj_image_to_text_dim = nn.Linear(self.hidden_size // 2, self.hidden_size).to(text_feats.device)

        image_feats_projected = self.proj_image_to_text_dim(image_feats)

        # 加权聚合
        aggregated_feats = text_weight * text_feats + image_weight * image_feats_projected

        return aggregated_feats


    def forward(self, token, att_mask, pos1, pos2, token_phrase, att_mask_phrase, token_qwen, att_mask_qwen, pic, A, W, A_rev, W_rev):
        batch_size = token.size(0)

        output_text = self.bert(token, attention_mask=att_mask)
        hidden_text = output_text[0]

        output_phrases = self.bert(token_phrase, attention_mask=att_mask_phrase)
        hidden_phrases = output_phrases[0]

        output_qwen = self.bert(token_qwen, attention_mask=att_mask_qwen)
        hidden_qwen = output_qwen[0]

        hidden_phrases = torch.sum(hidden_phrases, dim=1)
        hidden_phrases = self.linear_phrases(hidden_phrases)

        hidden_qwen = torch.sum(hidden_qwen, dim=1)
        hidden_qwen = self.linear_qwen(hidden_qwen)


        onehot_head = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_tail = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden_text).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden_text).sum(1)  # (B, H)


        text_feats = torch.cat([head_hidden, tail_hidden], dim=-1)


        img_feature, reck_hidden_entity, W_final = self.reck_encoder(token, att_mask, pos1, pos2, pic, A, W, A_rev,
                                                                     W_rev)
        img_feature = torch.mean(img_feature, dim=1)


        image_feats = self.linear_pic(img_feature)

        noise_loss = torch.tensor(0.0, device=text_feats.device)
        cl_loss = torch.tensor(0.0, device=text_feats.device)


        modality_signal = self.modality_signal_injection(text_feats, image_feats)

        if self.training:

            t = torch.randint(0, self.diffusion_steps, (batch_size,), device=text_feats.device)

            text_feats_noisy, text_noise = self.diffusion_forward_process(text_feats, t)
            text_feats_denoised, text_feats_pred, pred_noise = self.diffusion_reverse_process(
                text_feats_noisy, t, modality_signal
            )

            noise_loss = F.mse_loss(pred_noise, text_noise)

            cl_loss = self.cross_modal_contrastive_loss(text_feats, image_feats, batch_size)

        else:
            text_feats_denoised = text_feats

        aggregated_feats = self.multi_modal_graph_aggregation(text_feats_denoised, image_feats)

        aggregated_feats_flat = aggregated_feats.view(batch_size, -1)
        TK = torch.cat([text_feats, hidden_qwen, aggregated_feats_flat], dim=-1)


        y = self.linear_final(self.dropout_linear(TK))

        if self.training:
            return y, noise_loss, cl_loss
        else:
            return y





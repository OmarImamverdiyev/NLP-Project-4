from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class HighwayNetwork(nn.Module):
    def __init__(self, size: int, num_layers: int = 2):
        super().__init__()
        self.transforms = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for transform, gate in zip(self.transforms, self.gates):
            transformed = F.relu(transform(outputs))
            gate_values = torch.sigmoid(gate(outputs))
            outputs = gate_values * transformed + (1.0 - gate_values) * outputs
        return outputs


class SequenceEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence.size(1),
        )
        return self.dropout(output)


class AttentionFlow(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.similarity = nn.Linear(hidden_size * 3, 1, bias=False)

    def forward(
        self,
        context: torch.Tensor,
        question: torch.Tensor,
        context_mask: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        context_length = context.size(1)
        question_length = question.size(1)

        expanded_context = context.unsqueeze(2).expand(-1, -1, question_length, -1)
        expanded_question = question.unsqueeze(1).expand(-1, context_length, -1, -1)
        combined = torch.cat(
            [
                expanded_context,
                expanded_question,
                expanded_context * expanded_question,
            ],
            dim=-1,
        )
        similarity = self.similarity(combined).squeeze(-1)

        masked_similarity = similarity.masked_fill(~question_mask.unsqueeze(1), -1e30)
        context_to_question = torch.softmax(masked_similarity, dim=-1)
        attended_question = torch.bmm(context_to_question, question)

        question_to_context_weights = torch.softmax(
            masked_similarity.max(dim=2).values.masked_fill(~context_mask, -1e30),
            dim=-1,
        )
        attended_context = torch.bmm(question_to_context_weights.unsqueeze(1), context)
        attended_context = attended_context.expand(-1, context_length, -1)

        return torch.cat(
            [
                context,
                attended_question,
                context * attended_question,
                context * attended_context,
            ],
            dim=-1,
        )


class BiDAFQuestionAnswering(nn.Module):
    def __init__(
        self,
        embedding_mode: str,
        embedding_dim: int,
        hidden_size: int,
        dropout: float = 0.2,
        vocab_size: int | None = None,
        pad_idx: int = 0,
        bert_model_name: str = "bert-base-uncased",
        freeze_bert: bool = True,
        pretrained_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        if embedding_mode not in {"static", "bert"}:
            raise ValueError("embedding_mode must be either 'static' or 'bert'.")

        self.embedding_mode = embedding_mode
        self.freeze_bert = freeze_bert

        if embedding_mode == "static":
            if vocab_size is None:
                raise ValueError("vocab_size is required for static embeddings.")
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            if pretrained_embeddings is not None:
                if pretrained_embeddings.shape != self.token_embedding.weight.data.shape:
                    raise ValueError("pretrained_embeddings has incompatible shape.")
                self.token_embedding.weight.data.copy_(pretrained_embeddings)
        else:
            self.bert = AutoModel.from_pretrained(bert_model_name)
            bert_hidden_size = int(self.bert.config.hidden_size)
            self.bert_projection = nn.Linear(bert_hidden_size, embedding_dim)
            if freeze_bert:
                for parameter in self.bert.parameters():
                    parameter.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.highway = HighwayNetwork(embedding_dim)
        self.contextual_encoder = SequenceEncoder(embedding_dim, hidden_size, dropout)
        self.attention_flow = AttentionFlow(hidden_size * 2)
        self.modeling_encoder = SequenceEncoder(hidden_size * 8, hidden_size, dropout)
        self.output_encoder = SequenceEncoder(hidden_size * 2, hidden_size, dropout)
        self.start_projection = nn.Linear(hidden_size * 10, 1)
        self.end_projection = nn.Linear(hidden_size * 10, 1)

    def embed_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.embedding_mode == "static":
            embeddings = self.token_embedding(input_ids)
        else:
            bert_mask = attention_mask.long()
            if self.freeze_bert:
                self.bert.eval()
                with torch.no_grad():
                    bert_outputs = self.bert(input_ids=input_ids, attention_mask=bert_mask)
            else:
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=bert_mask)
            embeddings = self.bert_projection(bert_outputs.last_hidden_state)

        embeddings = self.highway(embeddings)
        return self.dropout(embeddings)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_embeddings = self.embed_tokens(context_ids, context_mask)
        question_embeddings = self.embed_tokens(question_ids, question_mask)

        context_encoded = self.contextual_encoder(context_embeddings, context_mask)
        question_encoded = self.contextual_encoder(question_embeddings, question_mask)

        attention_output = self.attention_flow(
            context_encoded,
            question_encoded,
            context_mask,
            question_mask,
        )
        modeled_context = self.modeling_encoder(attention_output, context_mask)
        output_context = self.output_encoder(modeled_context, context_mask)

        start_logits = self.start_projection(torch.cat([attention_output, modeled_context], dim=-1)).squeeze(-1)
        end_logits = self.end_projection(torch.cat([attention_output, output_context], dim=-1)).squeeze(-1)

        start_logits = start_logits.masked_fill(~context_mask, -1e30)
        end_logits = end_logits.masked_fill(~context_mask, -1e30)
        return start_logits, end_logits

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse as sps
MAX_LENGTH = 32


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        dim: int = 128,
        max_len: int = MAX_LENGTH,
        padding_idx: int = 0,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout_rate: float = 0.3,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        min_event=None,
        k=None,
        soft_k: bool = False,
        eps: float = 1e-6,
        only_new: bool = False,
        basket_vectors=None,
        frequency_mode="mode_3",  # no, item_based, default, flexible,
        frequency_max=10,
    ) -> None:
        super().__init__()
        self.min_event = min_event

        self.padding_idx = padding_idx
        self.frequency_mode = frequency_mode
        self.max_basket_size = max_len
        self.frequency_max = frequency_max
        basket_vectors = sps.coo_matrix(basket_vectors)
        values = basket_vectors.data
        idxs = np.vstack((basket_vectors.row, basket_vectors.col))
        i = torch.LongTensor(idxs)
        v = torch.FloatTensor(values)
        shape = basket_vectors.shape

        basket_vectors = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        self.item_embedding = nn.Embedding.from_pretrained(basket_vectors, freeze=True, sparse=True)

        self.position_embedding = nn.Embedding(max_len, dim, padding_idx=0)
        self.embeddings_dropout = nn.Dropout(dropout_rate)
        self.embeddings_norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(basket_vectors.shape[1], dim)

        self.num_items = basket_vectors.shape[1]
        # 1/0

        self.freq_1d = nn.Embedding.from_pretrained(
            torch.from_numpy(np.arange(frequency_max) / frequency_max)[:, None].float(), freeze=False
        )

        self.freq_2d = nn.Embedding(frequency_max, dim)

        input_to_freq = self.max_basket_size
        if frequency_mode == "mode_7":
            input_to_freq += dim

        if frequency_mode in ["mode_8", "mode_9"]:
            input_to_freq += dim

        history_proj_out = 1
        if frequency_mode == "mode_8":
            history_proj_out = 1

        self.history_projection = nn.Sequential(
            # nn.LayerNorm(self.max_basket_size),
            nn.Linear(input_to_freq, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, history_proj_out),
        )

        self.item_output_emb = nn.Embedding(n_items, dim)
        self.item_output_emb2 = nn.Embedding(n_items, dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_heads=n_heads,
                    hidden_size=dim,
                    intermediate_size=dim,
                    hidden_dropout_prob=dropout_rate,
                    attn_dropout_prob=dropout_rate,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )

        self.sequential = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.initializer_range = initializer_range

        self.encode_layer_dims = [n_items, dim, dim]

        self.basket_encoder = self.mlp_layers(self.encode_layer_dims)

        self.k = k
        self.soft_k = soft_k
        self.eps = eps
        self.alpha = None
        if soft_k:
            alpha_min = math.exp(math.log(eps) / (k - 1))
            alpha_max = math.exp(math.log(eps) / k)
            self.alpha = (alpha_min + alpha_max) / 2
        self.only_new = only_new

        self.concat_modes = ["mode_6", "mode_7", "mode_8"]
        self.hidden_dim = dim

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.

        .. math:: x_{l+1} = x_0 {x_l^T} w_l + b_l + x_l

        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`w_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of cross network, [batch_size, num_feature_field * embedding_size]

        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
            xl_dot = (x_0.transpose(0, 1) * xl_w).transpose(0, 1)
            x_l = xl_dot + self.cross_layer_b[i] + x_l
        return x_l

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    @property
    def n_items(self) -> int:
        return self.item_embedding.num_embeddings

    @property
    def dim(self) -> int:
        return self.item_embedding.embedding_dim

    @property
    def max_len(self) -> int:
        return self.position_embedding.num_embeddings

    @property
    def n_heads(self) -> int:
        return self.transformer_blocks[0].multi_head_attention.num_attention_heads

    @property
    def n_blocks(self) -> int:
        return len(self.transformer_blocks)

    @property
    def dropout_rate(self) -> float:
        return self.embeddings_dropout.p

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.head(self.state(x))

    def state(self, x: torch.LongTensor) -> torch.FloatTensor:

        dim_vectors = []
        freqs = []
        freq_scores = []
        for user in x.long():
            vectors = self.item_embedding(user[-self.max_basket_size :])

            if self.frequency_mode in self.concat_modes:
                to_add = torch.sparse_coo_tensor(size=(self.max_basket_size - vectors.shape[0], vectors.shape[1])).to(
                    vectors.device
                )

                input_for_freq = torch.cat([to_add, vectors]).T

                if self.frequency_mode in ["mode_7", "mode_8"]:
                    input_for_freq = torch.cat([input_for_freq.to_dense(), self.item_output_emb.weight], dim=1)

                input_for_freq = self.history_projection(input_for_freq).squeeze(1)
                freq_scores.append(input_for_freq)

            freqs.append(vectors.sum(0).to_dense())

            h = F.normalize(vectors.to_dense())
            h = self.basket_encoder(h)

            dim_vectors.append(h.unsqueeze(0))

        seq = torch.cat(dim_vectors, axis=0)

        # freq = seq.sum(1)
        freq = torch.cat(freqs).view(-1, self.num_items)

        if self.frequency_mode in self.concat_modes:
            input_for_freq = torch.cat(freq_scores).view(-1, self.num_items)

        positions = torch.arange(self.max_len - x.size(1), self.max_len, device=x.device).repeat(x.size(0), 1)

        seq += self.position_embedding(positions)

        seq = self.embeddings_norm(seq)
        seq = self.embeddings_dropout(seq)

        mask = self.get_attention_mask(x)

        for block in self.transformer_blocks:
            seq = block(seq, mask)

        transformer_output = seq[:, -1, :]
        user_vector = self.sequential(transformer_output)

        if self.frequency_mode == "mode_1":
            return freq + 0 * torch.einsum(
                "bd,id->bi",
                x,
                self.item_output_emb.weight,
            )

        elif self.frequency_mode == "mode_2":
            return freq + torch.einsum(
                "bd,id->bi",
                x,
                self.item_output_emb.weight,
            )

        elif self.frequency_mode == "mode_3":
            return torch.einsum(
                "bd,id->bi",
                user_vector,
                self.item_output_emb.weight,
            )

        elif self.frequency_mode == "mode_4":
            freq_vectors = self.freq_1d(torch.clip(freq.long(), max=self.frequency_max - 1)).squeeze(
                2
            )  # b num_items dim
            x = torch.einsum(
                "bd,id->bi",
                x,
                self.item_output_emb.weight,
            )
            x += freq_vectors
            return x

        elif self.frequency_mode == "mode_5":
            scores = torch.einsum(
                "bd,id->bi",
                x,
                self.item_output_emb.weight,
            )

            freq_embs = self.freq_2d(torch.clip(freq.long(), max=self.frequency_max - 1))
            # print(x.shape, freq_embs.shape)
            scores += torch.einsum("bd,bid->bi", x, freq_embs)

            return scores

        elif self.frequency_mode == "mode_6":
            # x = torch.einsum('bd,id->bi',x, self.item_output_emb.weight,)
            # x +=
            # input_for_freq
            return input_for_freq

        elif self.frequency_mode == "mode_7":
            return input_for_freq
        elif self.frequency_mode == "mode_8":

            # print(user_vector.shape, self.item_output_emb.weight.shape )

            user_item_scores = torch.einsum("bd,id->bi", user_vector, self.item_output_emb2.weight)

            # print(user_vector.shape, input_for_freq.shape, user_item_scores.shape)

            scores = user_item_scores + input_for_freq
            # scores[freq>0] = input_for_freq[freq>0]

            return scores  # input_for_freq + user_item_scores

    def head(self, seq: torch.FloatTensor) -> torch.FloatTensor:
        return seq @ self.item_embedding.weight.T + self.output_bias

    def get_attention_mask(self, item_seq: torch.LongTensor) -> torch.FloatTensor:
        attention_mask = item_seq != self.padding_idx
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        return torch.where(extended_attention_mask, 0.0, -10000.0)

    def calculate_loss(self, batch) -> torch.FloatTensor:
        # batch params
        items = batch["items"]
        device = items.device
        batch_size, seq_len = items.size()
        n_samples = batch_size * (seq_len - 1)
        samples_idx = torch.arange(n_samples, device=device, dtype=torch.long).unsqueeze(dim=1)

        # calc causal forward for seq_len-1 tokens
        logits = self.forward(items[:, :-1]).reshape(n_samples, self.n_items)

        # build sub-sequences to predict from (x) and to predict to (y)
        x_idx = self.build_x_indices(batch_size, seq_len, device)
        y_idx = self.build_y_indices(batch_size, seq_len, device)

        seqs_repeated = items.unsqueeze(dim=1).expand(batch_size, seq_len, seq_len)
        x = (torch.gather(seqs_repeated, dim=2, index=x_idx) * (x_idx > 0)).reshape(n_samples, seq_len - 1)
        x[:, 0] = seqs_repeated[:, :-1, 0].reshape(n_samples)
        y = (torch.gather(seqs_repeated, dim=2, index=y_idx) * (y_idx > 0)).reshape(n_samples, seq_len - 1)

        # do not predict items after k-th item
        if self.k is not None:
            y_k = (y > 0).cumsum(dim=1)
            to_predict = (y_k > 0) & (y_k <= self.k)
            y *= to_predict

        # set labels based on sub-sequences to predict to
        labels = torch.zeros((n_samples, self.n_items), device=device, dtype=torch.float32)
        if self.soft_k:
            soft_y = self.alpha ** (y_k - 1) * to_predict
            labels[samples_idx, y.flip(1)] = soft_y.flip(1)  # flip to set maximum score for repetitions
        else:
            labels[samples_idx, y] = 1.0
        labels[:, 0] = 0.0  # padding

        # do not predict items already presented in sequence
        if self.only_new:
            labels[samples_idx, x] = 0.0

        # filter out useless samples
        has_positives_to_predict = (labels != 0).any(dim=1)
        has_non_padded_from_predict = (x != self.padding_idx).any(dim=1)
        logits = logits[has_positives_to_predict & has_non_padded_from_predict]
        labels = labels[has_positives_to_predict & has_non_padded_from_predict]

        return F.binary_cross_entropy_with_logits(logits, labels)

    def build_x_indices(self, batch_size: int, seq_len: int, device: torch.device) -> torch.LongTensor:
        return (
            torch.arange(seq_len - 1, device=device, dtype=torch.long)
            .expand(seq_len - 1, seq_len - 1)
            .tril()
            .unsqueeze(dim=0)
            .expand(batch_size, seq_len - 1, seq_len - 1)
        )

    def build_y_indices(self, batch_size: int, seq_len: int, device: torch.device) -> torch.LongTensor:
        return (
            torch.arange(1, seq_len, device=device, dtype=torch.long)
            .expand(seq_len - 1, seq_len - 1)
            .triu()
            .unsqueeze(dim=0)
            .expand(batch_size, seq_len - 1, seq_len - 1)
        )

    @torch.no_grad()
    def full_sort_predict(self, batch) -> np.ndarray:
        return self.forward(batch["items"])[:, -1].detach().cpu().numpy()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear)):  # , nn.Embedding)):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            inner_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.FloatTensor) -> torch.FloatTensor:
        return self.feed_forward(self.multi_head_attention(hidden_states, attention_mask))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.FloatTensor) -> torch.FloatTensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*new_x_shape)

    def forward(self, input_tensor: torch.FloatTensor, attention_mask: torch.FloatTensor) -> torch.FloatTensor:
        query = self.transpose_for_scores(self.query(input_tensor)).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(self.key(input_tensor)).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(self.value(input_tensor)).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key) / self.sqrt_attention_head_size + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.norm(input_tensor + hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.dense_1(input_tensor)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.norm(input_tensor + hidden_states)


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

            - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation

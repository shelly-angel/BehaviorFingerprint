import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
class SelfAttention(nn.Module):
    def __init__(self,
                 emb_len:int,
                 num_atten_heads: int,
                 atten_dp_prob: float,
                 dp_prob:float
                 ):
        super(SelfAttention, self).__init__()
        self.emb_len = emb_len
        self.num_atten_heads = num_atten_heads
        if self.emb_len % self.num_atten_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.emb_len, self.num_atten_heads))
        self.attention_head_size = int(self.emb_len / self.num_atten_heads)
        self.all_head_size = self.num_atten_heads * self.attention_head_size

        self.query = nn.Linear(self.emb_len, self.all_head_size)
        self.key = nn.Linear(self.emb_len, self.all_head_size)
        self.value = nn.Linear(self.emb_len, self.all_head_size)

        self.attn_dropout = nn.Dropout(atten_dp_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(self.emb_len, self.emb_len)
        self.LayerNorm = LayerNorm(self.emb_len, eps=1e-12)
        self.out_dropout = nn.Dropout(dp_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_atten_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        # attention_mask = [B,1,Len,Len]；每个矩阵[0,0,:,:]的下半三角和对角线的元素都是-0， 其他地方都是-10000
        mixed_query_layer = self.query(input_tensor) #[B,L,E]
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer) # 调整维度[B,num_heads,L,head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask #attetion_mask=[B,1,L,L], attention_scores=[B,num_heads,L,L]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
class FeedForward(nn.Module):
    def __init__(self,emb_len:int,hidden_act,hidden_size,dp):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(emb_len, hidden_size)
        if isinstance(hidden_act, str):
            self.activation = ACT2FN[hidden_act]
        else:
            self.activation = hidden_act

        self.dense_2 = nn.Linear(hidden_size,emb_len)
        self.LayerNorm = LayerNorm(emb_len, eps=1e-12)
        self.dropout = nn.Dropout(dp)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
class Layer(nn.Module):
    def __init__(self,args):
        super(Layer, self).__init__()
        self.emb_len = args.emb_len #整数item变成embedding的长度
        self.num_atten_heads = args.num_atten_heads #多头注意力 有多少个头
        self.atten_dp_prob = args.atten_dp_prob
        self.dp_prob = args.dp_prob
        self.hidden_act = args.hidden_act
        self.hidden_size = args.hidden_size

        self.attention = SelfAttention(self.emb_len,self.num_atten_heads,self.atten_dp_prob,self.dp_prob)
        self.feedforward = FeedForward(self.emb_len,self.hidden_act,self.hidden_size,self.dp_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        feedforward_output = self.feedforward(attention_output)
        return feedforward_output
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_encoder_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        hidden_states = torch.mean(hidden_states,dim=1) # hidden_states的输出维度是[B,L,H] 求mean之后为[B,H]
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

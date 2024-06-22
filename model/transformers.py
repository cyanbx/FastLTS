from model.tf_submod import *


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, mem=None):
        "Pass the input (and mask) through each layer in turn."
        # if mem is not None:
        #     mem = torch.cat([mem, x], dim=1)  # 这样upper layer还在attn原来的x
        for layer in self.layers:
            if mem is None:
                x = layer(x, mask) #+ x
            else:
                x, mem, attn = layer(x, mask, mem)
        return x if mem is None else (x, mem, attn)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), N=3)
        self.size = size

    def forward(self, x, mask, mem=None):
        "Follow Figure 1 (left) for connections."
        if mem is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        else:
            x, mem_, attn = self.sublayer[0](x, lambda x: self.self_attn(x, mem, mem, mask, with_mem=True), with_mem=True)
            attn = attn.mean(2).mean(1)[:, :mem.size(1)]
            mem = mem[:, :mem.size(1)]

        x = self.sublayer[1](x, self.feed_forward)
        return x if mem is None else (x, mem, attn)


def make_transformer_encoder(N_layer=4, d_model=256, d_ff=1024, heads=8, dropout=0.1, ffn_layer='conv_relu_conv', first_kernel_size=9):
    "Helper: Construct a transformer encoder from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(heads, d_model)
    if ffn_layer == 'conv_relu_conv':
        ff = ConvReluConvFFN(d_model, d_ff, first_kernel_size, second_kernel_size=1, dropout=dropout)
    else:
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_layer)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


import numpy as np
import matplotlib as plt
import os
import copy
from torch import nn
import torch
import math
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed= src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator


    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        return out

    '''
    def decoder(self, src, src_mask) : # z-> src // c -> src_mask
        out = self.decoder(src, src_mask)
        return out
    '''

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask): # 추기
        out =self.decode(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)
        return out

    '''
    def forward(self, src, tgt, src_mask): # forward() 인자에 src_mask를 추가하고, encoder의 forward()에 넘겨준다.
        # x -> src  # z-> tgt  + src_mask를 추가해준다.
        encoder_out = self.encode(src, src_mask)
        y = self.decoder(tgt, encoder_out)
        return y
    '''
    def forward(self,src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt) #추가
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)  # 추가
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1) # 마지막 dimension인 len(vocab)에 대한 확률값을 구하기 위함
        return out, decoder_out # generator 추가하면서 바뀜

class Encoder(nn.Module) :
    def __init__(self, encoder_block, n_layer): # 여기서 n_layer는 encoder_block의 개수를 의미한다.
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
    '''
    def forward(self, x):
        out = x
        for layer in self.layers:
            out=layer(out)
        return out
    # encoder block들을 순서대로 실행하면서, 이전 block의 output을 이후 block의 input으로 넣는다.
    # 첫 block의 input은 Encoder 전체의 input x가 된다. 이후 가장 마지막 block의 output, 즉 context를 return한다.
    '''
    def forword(self, src, src_mask): # 외부에서 생성된 mask 인자를 추가하고, 이를 각 sublayer의 forward()에 넘겨준다.
        out=src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)] # 추가

    '''
    def forward(self, x):
        out= x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return out
    '''
    '''
    def forward(self, src, src_mask):
        out = src
        # 실제로 query, key, value를 받아야하므로 x -> query, key, value로 수정해준다.
        out = self.self_attention(query =out, key=out, value =out, mask= src_mask)
        out = self.position_ff(out)
        return out
    '''
    def forward(self, src, src_mask):
        out = src
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key= out, value=out, mask = src_mask))
        out = self.residuals[1](out, self.position_ff)

def calculate_attention(query, key, value, mask):
    # query, key, value: (n_batch, seq_len, d_k)
    # mask : (n_batch, seq_len, seq_len) -> mask : (n_batch, 1, seq_len, seq_len)

    # dk는 key의 shape에서 가로축을 의미한다.
    d_k = key.shape[-1]

    # Query x K^T
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, seq_len, seq_len) -> (n_batch, h, swq_len, seq_len)


    # Scaling
    attention_score = attention_score / math.sqrt(d_k)
    # Scaling을 해줘야 하는 이유 : Query와 Key의 길이가 커질수록 내적 값 역시 커질 가능성이 높기 때문에
    # softmax의 기울기가 0인 영역에 도달할 가능성이 높다.
    # 참조 : https://tigris-data-science.tistory.com/entry/%EC%B0%A8%EA%B7%BC%EC%B0%A8%EA%B7%BC-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-Transformer1-Scaled-Dot-Product-Attention


    # mask가 있다면 Masking된 부위 -1e9으로 채우기
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len) -> (n_batch, h, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k) - > (n_batch, h, seq_len, d_k)
    return out
    # Q, K, V 의 마지막 dimensiton은 반드시 d_k이여야만 한다.

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc              # (d_model, d_embed)

    def forward(self, *args, query, key, value, mask=None):
        # query, key, value : (n_batch, seq_len, d_embed)
        # mask : (n_batch, seq_len, seq_len)
        # return value : (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc) : # (n_batch, seq_len, d_embed)
            out = fc(x)        # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_bacth, h, seq_len, d_k)
        key = transform(key, self.k_fc) # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1,2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out

# Pad Mask Code in Pytorch
def make_pad_mask(self, query, key, pad_idx = 1):
    # query: (n_batch, query_seq_len)
    # key : (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    # unsqueeze 함수는 squeeze 함수의 반대로 1인 차원을 생성하는 함수다.
    # ne 함수는 =!의 의미로 key.ne(pad_idx)는 key와 pad_idx는 ==!이러는 의미로 해석할 수 있다.
    # 참조 : https://kongdols-room.tistory.com/127
    # 여기서는 만약 다르면 unsequeeze(1)을 하라는 의미같다.
    # 따라서 1번, 2번이 각각 1로 된다.
    # repeat()는 1이라 선언된걸 그대로 가져오고, 바꾸길 원하는 차원에 값을 넣어 바꾸는 mask다.
    key_mask = key.ne(pad_idx).unsqueeze(1).unszueeze(2) # (n_batch, 1, 1, key_seq_len)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1) # (n_batch, 1, query_seq_len, key_seq_len)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3) # (n_batch, 1, query_seq_len, 1)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len) # (n_batch, 1, query_seq_len, key_seq_len)

    # & 연산자 : 비트 연산
    mask = key_mask & query_mask # ???
    mask.requires_grad = False
    return mask

# 인자 : query, key ; n_batch x seq_len의 shape를 갖는다.
# <pad>의 index를 의미하는 pad_idx(대개 1)와 일치하는 token들은 모두 0, 그외에는 모두 1인 mask를 생성한다.

def make_src_mask(self, src):
    pad_mask = self.make_pad_mask(src, src)
    return pad_mask

class PositionWiseFeedForwardLayer(nn.module):
    def __int__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1 # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2 # (d_ff, d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    # 생성자의 인자로 받는 두 FC Layer는 (dembed x dff), (dff x dembed)의 shape를 가져야만 한다.

class ResidualConnectionLayer(nn.Module):
    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out+x
        return out


def make_subsequent_mask(query, key):
    # query : (n_batch, query_seq_len)
    # key : (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    # np.tril 함수는 하삼각행렬 (Lower triangular matrix)를 반환한다. k는 0(default) 또는 이외의 수를 정해주는 수이다.
    # np.zeros : 0으로 가득 찬 array를 생성
    # np.ones : 1로 가득 찬 array를 생성
    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
    mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    return mask

def make_tgt_mask(self, tgt):
    pad_mask = self.make_pad_mask(tgt, tgt)
    seq_mask = self.make_subsequent_mask(tgt, tgt)
    mask = pad_mask & seq_mask
    return pad_mask & seq_mask # ???

class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])

    def forward(self,tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out=layer(out, encoder_out, tgt_mask, src_tgt_mask)

def make_src_tgt_mask(self, src, tgt):
    pad_mask = self.make_pad_mask(tgt, src)
    return pad_mask

def make_pad_mask(self, query, key):
    return 0

class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key = out, value=out, mask= tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(query=out, key = encoder_out, value= encoder_out, mask = src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out

# Embedding
# 원래 Transformer의 input shape는 (n_batch x seq_len)인데, Encoder와 Decoder의 Input은
# n_batch x seq_len x d_embed의 shape를 가진 것으로 가정했다. 이는 Embedding 과정을 생략해서 그렇다.
# Embedding을 구현해주자

class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        # nn.Sequential은 순서를 갖는 모듈 컨테이너이다. 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달된다.


    def forward(self, x):
        out = self.embedding(x)
        return out

# embedding에도 scaling을 적용한다는 걸 note!
class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self,x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len= 256, device= torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        # 이렇게 False로 하는 이유는 생성자에서 만든 encoding을 forward() 내부에서 slicing해서 사용ㅎ나ㅡㄴ데
        # 이 Encoding이 학습되지 않도록 False를 부여해야 한다는 것이다.
        # Positional Encoding은 학습되는 parameter가 아니기 때문이다.
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0)/d_embed))
        encoding[:, 0::2] =torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsequuze(0).to(device)

    def forward(self,x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x+pos_embed
        return out
# Positional Encoding의 목적은 Positional 정보(token index number 등)을 정규화시키기 위한 것이다.
# 단순하게 index number로만 Positional Encoding으로 사용하게 된다면, 만약 Training data에서는 최대 문장의길이가 30인데
# test data에서 길이 50인 문장이 나오게 된다면 30 ~49의 index는 model이 학습한 적 없는 정보가 된다.
# 따라서 positional 정보를 일정한 범위 안의 실수로 제약해둔다.
# 여기서 sin, cos함수를 사용하는데, 짝수 index에는 sin 함수를, 홀수 index에는 cos함수를 사용한다.
# 이를 사용할 경우 항상 -1에서 1 사이의 값만이 positional 정보로 사용된다.


########
# 이렇게 decoder를 거쳐서 나온 shape는 최종 output이 아니다. 왜냐하면 n_batch x seq_len x d_embed 인데
# 우리가 원하는 output은 target sentence인 n_batch x seq_len이기 때문이다.

# 따라서 추가적인 FC Layer를 거친다. 이를 Generator라고 부른다.
# Generator가 하는 일은 Decoder output의 마지막 dimension을 d_embed에서 len(vocab)으로 변경하는 것이다.

# Transformer를 생성하는 build_model()
def build_model(src_vocab_size, tgt_vocab_size, device= torch.device("cpu"),max_len =256, d_embed=512,
                n_layer=6, d_model=512, h=8, d_ff=2048):

    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
        d_embed= d_embed,
        vocab_size= src_vocab_size
    )

    tgt_token_embed = TokenEmbedding(
        d_embed = d_embed,
        vocab_size = tgt_vocab_size
    )
    pos_embed =PositionalEncoding (
        d_embed = d_embed,
        max_len = max_len,
        device= device
        )

    src_embed= TransformerEmbedding(
        token_embed = src_token_embed,
        pos_embed = copy(pos_embed))

    tgt_embed = TransformerEmbedding(
        token_embed= tgt_token_embed,
        pos_embed = copy(pos_embed)
    )
    attention = MultiHeadAttentionLayer(
        d_model= d_model,
        h=h,
        qkv_fc = nn.Linear(d_embed, d_model),
        out_fc = nn.Linear(d_model, d_embed)
    )
    position_ff = PositionWiseFeedForwardLayer(
        fc1 = nn.Linear(d_embed, d_ff),
        fc2 = nn.Linear(d_ff, d_embed)
    )
    encoder_block = EncoderBlock(
        self_attention = copy(attention),
        position_ff= copy(position_ff)
    )
    decoder_block = DecoderBlock(
        self_attention = copy(attention),
        cross_attention= copy(attention),
        position_ff = copy(position_ff)
    )
    encoder = Encoder(
        encoder_block = encoder_block,
        n_layer = n_layer
    )
    decoder = Decoder(
        decoder_block = decoder_block,
        n_layer = n_layer
    )
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
        src_embed= src_embed,
        tgt_embed= tgt_embed,
        encoder = encoder,
        decoder= decoder,
        generator = generator).to(device)
    model.device = device

    return model
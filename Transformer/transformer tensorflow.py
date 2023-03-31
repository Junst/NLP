import numpy as np
import matplotlib as plt
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1/tf.pow(10000, (2*(i//2))/ tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.range(position, dtype=tf.float32)[:,tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # 배열의 짝수 인덱스 (2i)에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스 (2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:,1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1],:]

# 문장의 길이 50, 임베딩 벡터의 차원 128
sample_pos_encoding = PositionalEncoding(50,128)

# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0],cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 128))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()


def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)

    # Q와 K의 곱. Attention Score Matrix
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    ## 원소 간 곱 (element-wise product : tf.math.multiply(x, y)
    ## 행렬 간 곱 (Matrix Multiplication : tf.matmul)

    # Scaling
    # dk의 루트값으로 나눠준다.
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # Masking. Attention Score Matrix의 Masking 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 Softmax 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None :
        logits += (mask*-1e9)

    # Softmax 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
    # attention_weight : (batch_size, num_heads, query의 문장 길이, key으 ㅣ문장 길이)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # Output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights
'''
# 임의의 Query, Key, Value인 Q, K, V 행렬 생성
np.set_printoptions(suppress=True)
temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

# 함수 실행
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
print(temp_out) # 어텐션 값

temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
print(temp_out) # 어텐션 값
'''

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model, num_heads, name="multi-head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # d_model을 num_heads로 나눈 값
        # 논문 기준 : 64
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # WO에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(units=d_model)

    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['keys'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 (Dense Layer) 지나기
        # q : (batach_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        # 참고) Encoder(k, v)- Decoder(q) Attention에서는 Query 길이와 Key, Value의 길이는 다를 수 있따.

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. Head 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)


        # 3. Scaled Dot Product Attention / 앞서 구현한 함수 사용
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. Head Concatenate
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 5. WO에 해당하는 Dense Layer 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs= self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    ## tf.cast() -> Tensor를 새로운 형태로 Casting하는데 사용한다. 부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.
    ## Boolean 형태인 경우 True이면 1, False이면 0을 출력한다.
    return mask[:, tf.newaxis, tf.newaxis, :]

# print(create_padding_mask(tf.constant({{1, 21, 777, 0, 0]])))

# 다음의 코드는 Encoder와 Decoder 내부에서 사용할 예정이다.
# outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
# outputs = tf.keras.layers.Dense(units=d_model, outputs)


def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # Encdoer는 Padding Mask를 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # Multi-Head Attention (첫번째 서브층 / Self-Attention)
    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query' : inputs, 'key' : inputs, 'value' : inputs, # Q = K = V
            'mask' : padding_mask # Padding mask 사용
        })
        # Dropout + Residual Connection과 Layer Normalization
        attention = tf.keras.layers.Dropout(rate=dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs+attention)

        # Position Wise Feed Forward Neural Network (두번째 서브층)
        outputs = tf.keras.layers.Dense(units=dff, activation= 'relu')(attention)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)

        # Dropout + Residual Connection, Layer Normalization
        outputs = tf.keras.layers.Dropout(rate=dropout)(output)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention+outputs)

        return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name
    )

def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # Encoder는 Padding Mask 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # Positional Encoding + Dropout
    embeddings = tf.keras.laters.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs= tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # Encoder를 num_layers 개 쌓기
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model= d_model, num_heads=num_heads,
                                dropout=dropout, name="encoder_layer_{}".format(i),)([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask =  create_padding_mask(x) # Padding Mask도 포함
    return tf.maximum(look_ahead_mask, padding_mask)

# print(create_look_ahead_mask(tf.constant([[1, 2, 0, 4, 5]])))

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs= tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name = "encoder_outputs")

    # Look-Ahead Mask (첫 번째 서브층)
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name = "look_ahead_mask")

    # Padding Mask (두 번째 서브층)
    padding_mask = tf.keras.Input(shape=(1, 1, None), name = 'padding_mask')

    # Multi-Head Attention (첫 번째 서브층 / Masked Self Attention)
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
        'query': inputs, 'keys': inputs, 'value':inputs, # Q = K  = V
        'mask' : look_ahead_mask # Look-Ahead Mask
    })

    # Residual Connection과 Layer Normalization
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1+inputs)

    # Multi-Head Attention (두 번째 서브층 / Decoder-Encdoer Attention)
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
        'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
        'mask': padding_mask # 패딩 마스크
    })

    # Dropout + Residual Connection, Layer Normalization
    attention2= tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2+attention1)

    # Position Wise Feed Forward Neural Network (세번째 서브층)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # Dropout + Residual Connection, Layer Normalization
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs+attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    # Decoder는 Look-Ahead Mask(첫 번째 서브층)와 Padding Mask(두 번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name= 'look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # Positional Encoding + Dropout
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # Decoder를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                name='decoder_layer_{}'.format(i),)(inputs= [outputs, enc_outputs,
                                                                             look_ahead_mask, padding_mask])
    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"):

    # Encoder의 Input
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # Decoder의 Input
    dec_inputs = tf.keras.Input(shape=(None, ), name="dec_inputs")

    # Encoder의 Padding MMask
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name= 'enc_padding_mask')(inputs)

    # Decoder의 Look-Ahead Mask(첫번째 서브층)
    look_ahead_mask= tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name= 'look_ahead_mask')(dec_inputs)

    # Decoder의 Padding Mask (두번째 서브층)
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    # Encoder의 출력은 enc_outputs. Decoder로 전달된다.
    enc_outputs= encoder(vocab_size=vocab_size, num_layers=num_layers, dff = dff, d_model = d_model,
                         num_heads=num_heads, dropout=dropout,)(inputs=[inputs, enc_padding_mask])
    # Encoder의 입력은 입력 문장과 패딩 마스크

    # Decoder의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads = num_heads, dropout=dropout,)\
        (inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # 다음 단어 예측을 위한 출력층
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


small_transformer = transformer(
    vocab_size = 9000,
    num_layers = 4,
    dff = 512,
    d_model = 128,
    num_heads = 4,
    dropout = 0.3,
    name="small_transformer")

tf.keras.utils.plot_model(
    small_transformer, to_file='small_transformer.png', show_shapes=True)

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH -1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction = 'none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step**(self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

sample_learning_rate = CustomSchedule(d_model=128)

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")




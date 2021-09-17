# 第49页代码----------------------------------------------------------------
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, num_units, **kwargs):
    super().__init__(**kwargs)
    self.num_heads = num_heads
    self.num_units_per_head = num_units // num_heads
    self.linear_queries = common.Dense(num_units)
    self.linear_keys = common.Dense(num_units)
    self.linear_values = common.Dense(num_units)
    self.linear_output = common.Dense(num_units)


# 第49-51页代码----------------------------------------------------------------
def call(self, inputs, memory=None, mask=None, cache=None): 
    """Runs the layer.
    Args:
      inputs: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
      memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
        If ``None``, computes self-attention.
      mask: The dot product mask. A boolean tensor of shape :math:`[B, T_2]` or
        :math:`[B, T_1, T_2]`.
      cache: An optional tuple containing projected keys and values from the
        previous step. Tensors of shape :math:`[B, H, T_2, D / H]`.
      training: Run in training mode.
    Returns:
      A tuple with the attention context, the updated cache and the attention
      probabilities of the first head (if :obj:`return_attention` is ``True``).
    """

    def _compute_kv(x):
      keys = self.linear_keys(x)
      keys = split_heads(keys, self.num_heads)
      values = self.linear_values(x)
      values = split_heads(values, self.num_heads)
      return keys, values

    # Compute queries.
    queries = self.linear_queries(inputs)
    queries = split_heads(queries, self.num_heads)
    queries *= self.num_units_per_head**-0.5

    # Compute keys and values.
    if memory is None:
      keys, values = _compute_kv(inputs)
    else:
      keys, values = _compute_kv(memory)

    # Dot product attention.
    dot = tf.matmul(queries, keys, transpose_b=True)
    
    if mask is not None:
      mask = tf.cast(mask, tf.float32)
      mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
      dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)
    
    attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
    heads = tf.matmul(attn, values)

    # Concatenate all heads output.
    combined = combine_heads(heads)
    outputs = self.linear_output(combined)
    return outputs
  
def split_heads(inputs, num_heads):
  """Splits a tensor in depth.
  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
    num_heads: The number of heads :math:`H`.
  Returns:
    A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
  """
  shape = misc.shape_list(inputs)
  outputs = tf.reshape(inputs, [shape[0], shape[1], num_heads, shape[2] // num_heads])
  outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
  return outputs


# 第57页代码----------------------------------------------------------------
def _encode(self, positions, depth):
    if depth % 2 != 0:
      raise ValueError("SinusoidalPositionEncoder expects the depth to be divisble "
                       "by 2 but got %d" % depth)

    batch_size = tf.shape(positions)[0]
    positions = tf.cast(positions, tf.float32)

    log_timescale_increment = math.log(10000) / (depth / 2)
    inv_timescales = tf.exp(tf.range(depth / 2, dtype=tf.float32) * -log_timescale_increment)
    inv_timescales = tf.reshape(tf.tile(inv_timescales, [batch_size]), [batch_size, depth // 2])
    scaled_time = tf.expand_dims(positions, -1) * tf.expand_dims(inv_timescales, 1)
    encoding = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
    return tf.cast(encoding, self.dtype)


# 第59页代码----------------------------------------------------------------
def _lower_triangle_mask(sequence_length, maximum_length=None, dtype=tf.bool):
  batch_size = tf.shape(sequence_length)[0]
  if maximum_length is None:
    maximum_length = tf.reduce_max(sequence_length)
  mask = tf.ones([batch_size, maximum_length, maximum_length], dtype=dtype)
  mask = tf.linalg.band_part(mask, -1, 0)
  return mask

def future_mask(sequence_length, maximum_length=None):
  """Builds the dot product mask for future positions.
  Args:
    sequence_length: The sequence length.
    maximum_length: Optional size of the returned time dimension. Otherwise
      it is the maximum of :obj:`sequence_length`.
    dtype: The type of the mask tensor.
  Returns:
    A ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[batch_size, max_length, max_length]``.
  """
  sequence_mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  sequence_mask = tf.expand_dims(sequence_mask, axis=1)
  mask = _lower_triangle_mask(sequence_length, maximum_length=maximum_length, dtype=dtype)
  return mask * sequence_mask


# 第65-66页代码----------------------------------------------------------------
# 根据tensorflow2的执行模式选择不同的求梯度方式
if tf.executing_eagerly():
  with tf.GradientTape() as tape:
    training_loss, reported_loss = self._run_model(source, target)
  gradients = tape.gradient(training_loss, self._model.trainable_variables)
else:
  training_loss, reported_loss = self._run_model(source, target)
  gradients = self._optimizer.get_gradients(training_loss, self._model.trainable_variables)
    
# 将求得的梯度存储存储起来
self._gradient_accumulator(gradients)
    
# 构造存储梯度的实例，即为上面的self._gradient_accumulator
class GradientAccumulator(object):
  def __init__(self):
    """Initializes the accumulator."""
    self._gradients = []
    self._accum_steps = None
    
  def __call__(self, gradients):
    for accum_gradient, gradient in zip(self._gradients, gradients):
      accum_gradient.assign_add(gradient, read_value=False)
    self._accum_steps.assign_add(1)

  def reset(self):
    """Resets the accumulated gradients on the current replica."""
    if not self._gradients:
      return
    self._accum_steps.assign(0)
    for gradient in self._gradients:
      gradient.assign(tf.zeros(gradient.shape, dtype=gradient.dtype), read_value=False)

# 更新梯度
gradients = list(gradient.value() for gradient in self._gradient_accumulator._gradients)
self._optimizer.apply_gradients(list(zip(gradients, self._model.trainable_variables)))

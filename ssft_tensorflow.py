import numpy as np
import tensorflow as tf
import tf_keras as keras
from keras import Model, Input
from keras import layers
from keras.constraints import max_norm
from keras.saving import register_keras_serializable
import yaml

@register_keras_serializable(package="CustomLayers")
class SimpleSelfAttention(layers.Layer):
    def __init__(self, depth: int, **kwargs):
        super(SimpleSelfAttention, self).__init__(**kwargs)
        self.depth = depth
        self.q_dense_layer = None
        self.k_dense_layer = None
        self.v_dense_layer = None
        self.output_dense_layer = None

    def build(self, input_shape):
        self.q_dense_layer = layers.Dense(self.depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = layers.Dense(self.depth, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = layers.Dense(self.depth, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = layers.Dense(self.depth, use_bias=False, name='output_dense_layer')
        super(SimpleSelfAttention, self).build(input_shape)

    def call(self, inputs):
        memory = inputs
        q = self.q_dense_layer(inputs)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)
        logit = tf.matmul(q, k, transpose_b=True)

        attention_weight = layers.Softmax(name='attention_weight')(logit)
        attention_output = tf.matmul(attention_weight, v)
        return self.output_dense_layer(attention_output)

    def get_config(self):
        config = super(SimpleSelfAttention, self).get_config()
        config.update({"depth": self.depth})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def SleepSatelight(
    segment_length: int,
    n_channels: int,
    fact_conv_size: int,
    n_fact_conv: int,
    D: int,
    dropout_rate: float,
    n_hop: int,
    *args,
    **kwargs
):
    input_t = Input(shape=(n_channels, segment_length//2, 1))
    input_f = Input(shape=(n_channels, segment_length//4, 1))

    # Temporal stream embedding block
    block1 = layers.Conv2D(
        n_fact_conv,
        (1, fact_conv_size),
        padding='same',
        use_bias=False
    )(input_t)
    
    block1 = layers.DepthwiseConv2D(
        (n_channels, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.)
    )(block1)
    
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('relu')(block1)
    block1 = layers.Dropout(dropout_rate)(block1)
    block1 = layers.AveragePooling2D((1, 4))(block1)
    block1 = layers.Reshape((-1, n_fact_conv*D))(block1)

    # Temporal stream SA blocks
    block2 = block1
    for hop in range(n_hop):
        curr_dim = n_fact_conv*D*(hop+1)
        next_dim = n_fact_conv*D*(hop+2)
        
        # Project skip connection to match dimensions
        skip = layers.Dense(curr_dim)(block2) if hop > 0 else block2
        
        block2 = SimpleSelfAttention(curr_dim)(block2)
        block2 = layers.BatchNormalization(axis=-1)(block2)
        block2 = layers.Dropout(dropout_rate)(block2)
        block2 = layers.Add()([skip, block2])
        
        # Prepare for dimension increase
        block2 = layers.Dense(next_dim, use_bias=True)(block2)
        block2 = layers.BatchNormalization()(block2)
        block2 = layers.Activation('relu')(block2)
        block2 = layers.Dropout(dropout_rate)(block2)
        block2 = layers.AveragePooling1D(4)(block2)

    # Frequency stream embedding block
    block3 = layers.Conv2D(
        n_fact_conv,
        (1, fact_conv_size),
        padding='same',
        use_bias=False
    )(input_f)
    
    block3 = layers.DepthwiseConv2D(
        (n_channels, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.)
    )(block3)
    
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation('relu')(block3)
    block3 = layers.Dropout(dropout_rate)(block3)
    block3 = layers.AveragePooling2D((1, 4))(block3)
    block3 = layers.Reshape((-1, n_fact_conv*D))(block3)

    # Frequency stream SA blocks
    block4 = block3
    for hop in range(n_hop):
        curr_dim = n_fact_conv*D*(hop+1)
        next_dim = n_fact_conv*D*(hop+2)
        
        # Project skip connection to match dimensions
        skip = layers.Dense(curr_dim)(block4) if hop > 0 else block4
        
        block4 = SimpleSelfAttention(curr_dim)(block4)
        block4 = layers.BatchNormalization(axis=-1)(block4)
        block4 = layers.Dropout(dropout_rate)(block4)
        block4 = layers.Add()([skip, block4])
        
        # Prepare for dimension increase
        block4 = layers.Dense(next_dim, use_bias=True)(block4)
        block4 = layers.BatchNormalization()(block4)
        block4 = layers.Activation('relu')(block4)
        block4 = layers.Dropout(dropout_rate)(block4)
        block4 = layers.AveragePooling1D(4)(block4)

    # Output layers
    flatten_t = layers.Flatten(name='flatten_t')(block2)
    flatten_f = layers.Flatten(name='flatten_f')(block4)
    combined = layers.Concatenate()([flatten_t, flatten_f])
    dense1 = layers.Dense(300, name='output_dense1')(combined)
    dense2 = layers.Dense(5, name='output_dense2')(dense1)
    softmax = layers.Activation('softmax', name='softmax')(dense2)

    return Model(inputs=[input_t, input_f], outputs=softmax)


def build_model(
    segment_length: int = 1500,
    n_channels: int = 1,
    fact_conv_size: int = 250,
    n_fact_conv: int = 16,
    D: int = 2,
    dropout_rate: float = 0.2,
    n_hop: int = 3,
    learning_rate: float = 1e-4,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7,
):

    model = SleepSatelight(
        segment_length=segment_length,
        n_channels=n_channels,
        fact_conv_size=fact_conv_size,
        n_fact_conv=n_fact_conv,
        D=D,
        dropout_rate=dropout_rate,
        n_hop=n_hop,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model_params_from_config(config_path: str = "config.yaml") -> dict:
    """
    Helper to load the "model" section from config.yaml
    and return a dict that can be passed directly to build_model.

    - For items not specified in YAML, fall back to build_model defaults.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = (cfg or {}).get("model", {}) or {}

    params = {
        "segment_length": model_cfg.get("segment_length", 1500),
        "n_channels": model_cfg.get("n_channels", 1),
        "fact_conv_size": model_cfg.get("fact_conv_size", 250),
        "n_fact_conv": model_cfg.get("n_fact_conv", 16),
        "D": model_cfg.get("D", 2),
        "dropout_rate": model_cfg.get("dropout_rate", 0.2),
        "n_hop": model_cfg.get("n_hop", 3),
        # Optimizer parameters (optional in YAML, with safe defaults)
        "learning_rate": model_cfg.get("learning_rate", 1e-4),
        "beta_1": model_cfg.get("beta_1", 0.9),
        "beta_2": model_cfg.get("beta_2", 0.999),
        "epsilon": model_cfg.get("epsilon", 1e-7),
    }
    return params


def build_model_from_config(config_path: str = "config.yaml") -> Model:
    """
    Utility to construct a model according to the "model" section
    of config.yaml.
    """
    params = load_model_params_from_config(config_path)
    return build_model(**params)


if __name__ == "__main__":
    """
    When this module is executed directly, load config.yaml
    and print the model summary (training is done in train_*.py).
    """
    model = build_model_from_config("config.yaml")
    model.summary()

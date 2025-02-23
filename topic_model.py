import tensorflow as tf


def create_model(num_classes, emb_table_size, model_chkpt, topk=5):
    """
    Function to create full model.

    Input:
    num_classes: number of classes
    emb_table_size: size of embedding table
    model_chkpt: path to model checkpoint
    topk: number of predictions to return

    Output:
    model: full model
    """
    # Inputs
    citation_0 = tf.keras.layers.Input((16,), dtype=tf.int64, name="citation_0")
    citation_1 = tf.keras.layers.Input((128,), dtype=tf.int64, name="citation_1")
    journal = tf.keras.layers.Input((384,), dtype=tf.float32, name="journal_emb")
    language_model_output = tf.keras.layers.Input(
        (
            512,
            768,
        ),
        dtype=tf.float32,
        name="lang_model_output",
    )

    # Create a multi-class classification model using functional API
    pooled_language_model_output = tf.keras.layers.GlobalAveragePooling1D()(
        language_model_output
    )
    citation_emb_layer = tf.keras.layers.Embedding(
        input_dim=emb_table_size,
        output_dim=256,
        mask_zero=True,
        trainable=True,
        name="citation_emb_layer",
    )

    citation_0_emb = citation_emb_layer(citation_0)
    citation_1_emb = citation_emb_layer(citation_1)

    pooled_citation_0 = tf.keras.layers.GlobalAveragePooling1D()(citation_0_emb)
    pooled_citation_1 = tf.keras.layers.GlobalAveragePooling1D()(citation_1_emb)

    concat_data = tf.keras.layers.Concatenate(name="concat_data", axis=-1)(
        [pooled_language_model_output, pooled_citation_0, pooled_citation_1, journal]
    )

    # Dense layer 1
    dense_output = tf.keras.layers.Dense(
        2048, activation="relu", kernel_regularizer="L2", name="dense_1"
    )(concat_data)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_1")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="layer_norm_1"
    )(dense_output)

    # Dense layer 2
    dense_output = tf.keras.layers.Dense(
        1024, activation="relu", kernel_regularizer="L2", name="dense_2"
    )(dense_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_2")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="layer_norm_2"
    )(dense_output)

    # Dense layer 3
    dense_output_l3 = tf.keras.layers.Dense(
        512, activation="relu", kernel_regularizer="L2", name="dense_3"
    )(dense_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_3")(dense_output_l3)
    dense_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="layer_norm_3"
    )(dense_output)

    output_layer = tf.keras.layers.Dense(
        num_classes, activation="sigmoid", name="output_layer"
    )(dense_output)
    topk_outputs = tf.math.top_k(output_layer, k=topk)

    model = tf.keras.Model(
        inputs=[citation_0, citation_1, journal, language_model_output],
        outputs=topk_outputs,
    )

    model.load_weights(model_chkpt)
    model.trainable = False

    return model

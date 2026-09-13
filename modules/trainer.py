"""Modul Trainer untuk melatih model regresi estimasi harga rumah."""

import tensorflow as tf
import tensorflow_transform as tft
import tf_keras as keras
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'price'

NUMERICAL_FEATURES = [
    'bathrooms', 'bedrooms', 'condition', 'floors',
    'sqft_above', 'sqft_basement', 'sqft_living', 'sqft_lot',
    'view', 'waterfront', 'yr_built', 'yr_renovated'
]

CATEGORICAL_FEATURES = ['city', 'statezip']


def transformed_name(key: str) -> str:
    """Mengembalikan nama fitur yang telah ditransformasi."""
    return f"{key}_xf"


def _gzip_reader_fn(filenames):
    """Membaca berkas TFRecord terkompresi GZIP."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Menyiapkan dataset batch untuk proses training dan validasi."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=None,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset


def _build_keras_model(hp_dict: dict) -> keras.Model:
    """Membangun arsitektur Keras Deep Neural Network multi-input."""
    inputs = {}

    for feat in NUMERICAL_FEATURES:
        inputs[transformed_name(feat)] = keras.layers.Input(
            shape=(1,), name=transformed_name(feat), dtype=tf.float32
        )

    for feat in CATEGORICAL_FEATURES:
        inputs[transformed_name(feat)] = keras.layers.Input(
            shape=(1,), name=transformed_name(feat), dtype=tf.int64
        )

    cat_embeddings = []
    for feat in CATEGORICAL_FEATURES:
        cleaned = keras.layers.Lambda(
            lambda x: tf.clip_by_value(
                x, tf.cast(0, x.dtype), tf.cast(149, x.dtype)
            )
        )(inputs[transformed_name(feat)])
        embed = keras.layers.Embedding(input_dim=150, output_dim=8)(cleaned)
        cat_embeddings.append(keras.layers.Flatten()(embed))

    num_layers = [inputs[transformed_name(f)] for f in NUMERICAL_FEATURES]
    all_features = keras.layers.concatenate(num_layers + cat_embeddings)

    layer_x = keras.layers.Dense(
        hp_dict.get('units_1', 64), activation='relu'
    )(all_features)
    layer_x = keras.layers.Dropout(hp_dict.get('dropout_rate', 0.2))(layer_x)
    layer_x = keras.layers.Dense(
        hp_dict.get('units_2', 32), activation='relu'
    )(layer_x)
    output = keras.layers.Dense(1, activation='linear')(layer_x)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp_dict.get('learning_rate', 0.001)
        ),
        loss='mean_squared_error',
        metrics=[keras.metrics.MeanAbsoluteError(name='mean_absolute_error')]
    )
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Menghasilkan fungsi serving signature untuk TFRecord mentah."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        ]
    )
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Fungsi utama eksekusi pelatihan model oleh komponen Trainer TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, batch_size=64
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, batch_size=64
    )

    hp_dict = (
        fn_args.hyperparameters.get('values', {})
        if fn_args.hyperparameters else {}
    )

    model = _build_keras_model(hp_dict)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=15
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ),
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )

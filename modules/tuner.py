"""Modul Tuner untuk hyperparameter tuning model estimasi harga rumah."""

# pylint: disable=duplicate-code

from typing import Any, Dict, NamedTuple, Text
import keras_tuner as kt
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
    """Menyiapkan dataset batch untuk proses hyperparameter tuning."""
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


def model_builder(hp: kt.HyperParameters) -> keras.Model:
    """Membangun arsitektur model Keras untuk pencarian hyperparameter."""
    inputs = {}

    # Layer input numerik
    for feat in NUMERICAL_FEATURES:
        inputs[transformed_name(feat)] = keras.layers.Input(
            shape=(1,), name=transformed_name(feat), dtype=tf.float32
        )

    # Layer input kategorikal
    for feat in CATEGORICAL_FEATURES:
        inputs[transformed_name(feat)] = keras.layers.Input(
            shape=(1,), name=transformed_name(feat), dtype=tf.int64
        )

    # Konversi categorical index ke embedding representation dengan index clipping
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

    # Tuning jumlah unit layer dense pertama
    units_1 = hp.Int(
        'units_1', min_value=32, max_value=128, step=32, default=64
    )
    layer_x = keras.layers.Dense(units_1, activation='relu')(all_features)

    # Tuning dropout
    dropout_rate = hp.Float(
        'dropout_rate', min_value=0.1, max_value=0.4, step=0.1, default=0.2
    )
    layer_x = keras.layers.Dropout(dropout_rate)(layer_x)

    # Tuning jumlah unit layer dense kedua
    units_2 = hp.Int(
        'units_2', min_value=16, max_value=64, step=16, default=32
    )
    layer_x = keras.layers.Dense(units_2, activation='relu')(layer_x)
    output = keras.layers.Dense(1, activation='linear')(layer_x)

    model = keras.Model(inputs=inputs, outputs=output)

    # Tuning learning rate optimizer
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=[keras.metrics.MeanAbsoluteError(name='mean_absolute_error')]
    )
    return model


TunerFnResult = NamedTuple(
    'TunerFnResult',
    [
        ('tuner', kt.engine.base_tuner.BaseTuner),
        ('fit_kwargs', Dict[Text, Any]),
    ],
)


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama eksekusi hyperparameter tuning oleh komponen Tuner TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(
        fn_args.train_files, tf_transform_output, batch_size=64
    )
    eval_set = _input_fn(
        fn_args.eval_files, tf_transform_output, batch_size=64
    )

    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective=kt.Objective('val_mean_absolute_error', direction='min'),
        max_trials=3,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='house_price_tuning'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': eval_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'epochs': 5
        }
    )

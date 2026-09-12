from typing import NamedTuple, Dict, Text, Any
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'price'

NUMERICAL_FEATURES = [
    'bathrooms', 'bedrooms', 'condition', 'floors',
    'sqft_above', 'sqft_basement', 'sqft_living', 'sqft_lot',
    'view', 'waterfront', 'yr_built', 'yr_renovated'
]

CATEGORICAL_FEATURES = ['city', 'statezip']

def transformed_name(key: str) -> str:
    return f"{key}_xf"

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
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

def model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
    inputs = {}

    # Layer input numerik
    for feat in NUMERICAL_FEATURES:
        inputs[transformed_name(feat)] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name(feat), dtype=tf.float32
        )

    # Layer input kategorikal
    for feat in CATEGORICAL_FEATURES:
        inputs[transformed_name(feat)] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name(feat), dtype=tf.int64
        )

    # Konversi categorical index ke embedding representation
    cat_embeddings = []
    for feat in CATEGORICAL_FEATURES:
        embed = tf.keras.layers.Embedding(
            input_dim=150, output_dim=8
        )(inputs[transformed_name(feat)])
        cat_embeddings.append(tf.keras.layers.Flatten()(embed))

    num_layers = [inputs[transformed_name(f)] for f in NUMERICAL_FEATURES]
    all_features = tf.keras.layers.concatenate(num_layers + cat_embeddings)

    # Tuning jumlah unit layer dense
    units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32, default=64)
    x = tf.keras.layers.Dense(units_1, activation='relu')(all_features)

    # Tuning dropout
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1, default=0.2)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16, default=32)
    x = tf.keras.layers.Dense(units_2, activation='relu')(x)

    output = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    # Tuning learning rate
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error')]
    )
    return model

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', kt.engine.base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_set = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

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

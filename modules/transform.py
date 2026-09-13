"""Modul Transform untuk rekayasa fitur data prediksi harga rumah menggunakan TFT."""

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'price'

NUMERICAL_FEATURES = [
    'bathrooms',
    'bedrooms',
    'condition',
    'floors',
    'sqft_above',
    'sqft_basement',
    'sqft_living',
    'sqft_lot',
    'view',
    'waterfront',
    'yr_built',
    'yr_renovated'
]

CATEGORICAL_FEATURES = [
    'city',
    'statezip'
]

def transformed_name(key: str) -> str:
    """Mengubah nama fitur asli menjadi nama fitur hasil transformasi"""
    return f"{key}_xf"

def preprocessing_fn(inputs):
    """Fungsi transformasi fitur menggunakan TensorFlow Transform"""
    outputs = {}

    # Standarisasi fitur numerik dengan z-score
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            tf.cast(inputs[feature], tf.float32)
        )

    # Encoding fitur kategorikal ke vocabulary index
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            tf.cast(inputs[feature], tf.string)
        )

    # Label target dipastikan bertipe float32
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)

    return outputs

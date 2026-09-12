FROM tensorflow/serving:latest

COPY ./serving_model_dir/house_pricing_model /models/house_pricing_model
ENV MODEL_NAME=house_pricing_model

EXPOSE 8501
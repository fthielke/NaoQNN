FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

ADD pyproject.toml README.md ball_model_experiments/requirements.txt /build/
ADD naoqnn /build/naoqnn
RUN python -m pip install --upgrade pip \
 && python -m pip install /build \
 && python -m pip install -r /build/requirements.txt \
 && rm -rf /build

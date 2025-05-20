# NaoQNN: Quantized Neural Networks for NAO V6 robots

This repository contains code for the paper "Quantized Neural Networks for Ball Detection on the NAO Robot: An Optimized Implementation", presented at the RoboCup 2024 Symposium.

## Paper

Thielke, F. (2025). Quantized Neural Networks for Ball Detection on the NAO Robot: An Optimized Implementation. In: Barros, E., Hanna, J.P., Okada, H., Torta, E. (eds) RoboCup 2024: Robot World Cup XXVII. RoboCup 2024. Lecture Notes in Computer Science(), vol 15570. Springer, Cham. https://doi.org/10.1007/978-3-031-85859-8_9

## How to reproduce results from the paper

1. Open ball_model_experiments and either start the devcontainer or install NaoQNN and requirements.txt into your Python environment.
2. Download [b-alls-2019.hdf5](https://b-human.informatik.uni-bremen.de/public/datasets/b-alls-2019/b-alls-2019.hdf5).
3. Run `python prepare_dataset.py` to prepare the TensorFlow datasets.
4. Train the models by executing `python train_ball_model.py 0` to `python train_ball_model.py 7`. Since the models are very small, depending on your hardware these 8 processes can easily run in parallel on the same machine.
5. Run `python inference_test.py` to perform predictions on the test set.
6. Run `test_models/export_models.py` to convert the quantized models to asm and the float models to H5 files.
7. Compile `test_models/measure_speed.cpp` and link it with [CompiledNN](https://github.com/bhuman/CompiledNN). Run the resulting executable on a NAO V6 robot to get inference timings.
8. Check that the resulting inference times match those in `MEASURED_INFERENCE_TIME` in `calculate_metrics.py` and correct them if necessary.
9. Run `python calculate_metrics.py` to generate plots and a LaTeX table with the results.
10. Run `python plot_ball_detections.py` to recreate Fig. 4 from the paper. Note that by default the plotted cases are chosen randomly.

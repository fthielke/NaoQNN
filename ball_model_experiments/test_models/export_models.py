# SPDX-License-Identifier: MIT
from ball_model import BallModel
from train_ball_model import MODEL_CONFIGURATIONS, RANGE_CENTER, RANGE_RADIUS, load_data
from generate_asm import ball_model_to_asm, dataset_to_asm


def export_float_models():
    filenames = []

    for config in MODEL_CONFIGURATIONS[:4]:
        weights_path = '../weights/' + '/'.join(str(c.value) for c in config) + '/weights_0'
        filename = '_'.join(str(c.value) for c in config) + '.h5'

        model = BallModel(RANGE_CENTER, RANGE_RADIUS, *config)
        model.load_weights(weights_path).expect_partial()

        model.to_float_model().save(filename, save_format='h5')
        filenames.append(filename)

    return filenames

def export_quantized_models():
    models = []
    model_names = []

    for config in MODEL_CONFIGURATIONS[4:]:
        weights_path = '../weights/' + '/'.join(str(c.value) for c in config) + '/weights_0'
        model_name = '_'.join(str(c.value) for c in config)

        model = BallModel(RANGE_CENTER, RANGE_RADIUS, *config)
        model.load_weights(weights_path).expect_partial()
        models.append(model)
        model_names.append(model_name)

    for model, model_name in zip(models, model_names):
        with open(f'{model_name}.s', 'w', encoding='utf-8', newline='') as f:
            print(ball_model_to_asm(model, inference_symbol='run_' + model_name), file=f)

    return model_names

def export_testset():
    train, val, test = load_data()
    del train
    del val

    with open('test_dataset.s', 'w', encoding='utf-8', newline='') as f:
        print(dataset_to_asm(test.as_numpy_iterator()), file=f)


if __name__ == '__main__':
    h5_filenames = export_float_models()
    quantized_model_names = export_quantized_models()
    export_testset()

    print('\n{' + ', '.join(f'"{filename}"' for filename in h5_filenames) + '}')

    for model_name in quantized_model_names:
        print(f'void run_{model_name}(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);')

    print('\n{' + ', '.join(f'std::make_pair<>("{model_name}", run_{model_name})' for model_name in quantized_model_names) + '}')

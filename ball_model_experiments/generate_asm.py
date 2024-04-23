# SPDX-License-Identifier: MIT
from typing import Iterable, Tuple
import numpy as np
import tensorflow as tf
from naoqnn import QuantizedConv1x1Softmax
from naoqnn.generate_asm import dispatch_layer, LabelManager, State, asm_dense_16_argmax, asm_dense_32_argmax
from naoqnn.generate_asm.copy import asm_copy_memory

from ball_model import BallModel, BallModelEncoder, BallModelClassifier, BallModelDetector


TEST_MODEL_CONFIG=(BallModelEncoder.GERRIT_ORIGINAL_QUANTIZED, BallModelClassifier.GERRIT_ORIGINAL_QUANTIZED, BallModelDetector.GERRIT_ORIGINAL_QUANTIZED)
TEST_MODEL_WEIGHTS_PATH='weights/' + '/'.join(str(c.value) for c in TEST_MODEL_CONFIG) + '/weights_0'


def ball_model_to_asm(model: BallModel, num_layers=-1, output_classification_logits: bool = False, additional_classification_offset: int = 0, inference_symbol: str = 'run_quantized_model') -> str:
    if num_layers < 1:
        num_layers = len(model.encoder.layers) + len(model.classification_model.layers) + len(model.detection_model.layers)
    label_manager = LabelManager()

    # Parameters:
    # Input address (RDI)
    # Classification output address (RSI)
    # Detection output address (RDX)
    # Batch size (ECX -> R8D)
    state = State(
        data_offset=0,
        width=32,
        height=32,
        text=f"""    .text
.globl {inference_symbol}
{inference_symbol}:
    pushq %r13
    pushq %r12
    movl %ecx, %r8d
    leaq model_weights(%rip), %rax""", # RAX points to model weights
        data="""    .data
    .align 64
model_weights:"""
    )
    return_statement = """
    popq %r12
    popq %r13
    retq"""

    # Register map after input layer:
    # RDI detection output address (buffer for all following in-place operations)
    # RSI classification output address (only needed to store the results of the softmax layer)
    # RAX weights pointer (-data_offset)
    # RCX unused
    # RDX input ptr (detection output address)
    # R8D batch size
    # R9 unused
    # R10 unused
    # R11 output ptr (detection output address)

    # Encoder
    first_layer = True
    for layer in model.encoder.layers:
        dispatch_layer(layer, label_manager, state, first_layer=first_layer)
        first_layer = False

        num_layers -= 1
        if num_layers == 0:
            return state.text + f'\n{return_statement}\n\n' + state.data

    encoded_shape = (state.width, state.height)

    # Classification head
    classifier_layers = [layer for layer in model.classification_model.layers if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten)]
    assert isinstance(classifier_layers[-1], QuantizedConv1x1Softmax)
    if len(classifier_layers) > 1:
        classifier_input_size = np.prod(classifier_layers[0].input_shape[1:])
        state.text += f"""
    movq %rdx, %r13
    imull ${classifier_input_size}, %r8d, %r11d
    addq %rdx, %r11
    movq %r11, %rdi"""
        asm_copy_memory(
            in_ptr='%rdx',
            out_ptr='%r11',
            num_bytes_per_sample=classifier_input_size,
            label_manager=label_manager,
            state=state
        )

        state.text += """
    movq %rdi, %rdx
    movq %rdi, %r11"""

        for layer in classifier_layers[:-1]:
            dispatch_layer(layer, label_manager, state)
            num_layers -= 1
            if num_layers == 0:
                asm_copy_memory(
                    in_ptr='%rdx',
                    out_ptr='%r13',
                    num_bytes_per_sample=np.prod(layer.output_shape[1:]),
                    label_manager=label_manager,
                    state=state
                )
                return state.text + f'\n{return_statement}\n\n' + state.data

    if classifier_layers[-1].input_shape[-1] == 16:
        asm_dense_16_argmax(classifier_layers[-1], label_manager, state, additional_offset=additional_classification_offset, output_classification_logits=output_classification_logits)
    elif classifier_layers[-1].input_shape[-1] == 32:
        asm_dense_32_argmax(classifier_layers[-1], label_manager, state, additional_offset=additional_classification_offset, output_classification_logits=output_classification_logits)
    else:
        raise NotImplementedError()
    num_layers -= 1
    if num_layers == 0:
        return state.text + f'\n{return_statement}\n\n' + state.data

    if len(classifier_layers) > 1:
        state.text += """
    movq %r13, %rdi
    movq %r13, %rdx
    movq %r13, %r11"""

    state.width, state.height = encoded_shape
    print(f'{(state.width, state.height)=}')

    # Detection head
    detector_layers = [layer for layer in model.detection_model.layers if not isinstance(layer, tf.keras.layers.SpatialDropout2D) and not isinstance(layer, tf.keras.layers.Flatten)]
    for layer in detector_layers:
        dispatch_layer(layer, label_manager, state)
        num_layers -= 1
        if num_layers == 0:
            return state.text + f'\n{return_statement}\n\n' + state.data

    # Return
    return state.text + f'\n{return_statement}\n\n' + state.data

def dataset_to_asm(dataset: Iterable[Tuple[np.ndarray, np.ndarray]]) -> str:
    return f"""    .data
.globl test_data
    .align 64
test_data:
    .byte {', '.join(str(x) for image, _ in dataset for x in np.ravel(image, order='C'))}"""


def test_model(n_layers=None):
    test_data = tf.data.Dataset.load('balls_ds_validation', compression='GZIP')
    image, _ = next(test_data.as_numpy_iterator())

    model = BallModel((0, 255), (0, 255), *TEST_MODEL_CONFIG)
    if n_layers is None:
        n_layers = model.num_layers()
    model.load_weights(TEST_MODEL_WEIGHTS_PATH).expect_partial()
    reference = model.get_up_to_nth_layer(n_layers-1)(image[None,...], training=False).numpy().squeeze(axis=0).astype(np.uint8)
    print(reference.shape)
    is_classification_output = reference.shape[-1] == 2
    is_detection_output = reference.shape[-1] == 3

    import os
    os.makedirs('test_per_layer', exist_ok=True)

    with open('test_per_layer/model.s', 'w', encoding='utf-8', newline='') as f:
        print(ball_model_to_asm(model, num_layers=n_layers), file=f)

    with open('test_per_layer/test_data.s', 'w', encoding='utf-8', newline='') as f:
        print(f"""    .data
.globl test_data
    .align 64
test_data:
    .byte {', '.join(str(x) for x in np.ravel(image, order='C'))}
    .byte {', '.join(str(x) for x in np.ravel(image, order='C'))}""", file=f)

    if is_classification_output:
        reference = np.argmax(reference, axis=-1)
        with open('test_per_layer/main.cpp', 'w', encoding='utf-8', newline='') as f:
            print(f"""#include <array>
#include <fstream>
#include <iostream>

extern "C" {{
    void run_quantized_model(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);
    extern const unsigned char test_data[{np.size(image) * 2}];
}}

unsigned char output_class[1];
alignas(64) unsigned char output_buffer[{np.size(image) * 2 * 16}];

int main() {{
    run_quantized_model(test_data, output_class, output_buffer, 2);
    std::ofstream out("test_per_layer/results.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(output_class), 1);
    return 0;
}}
    """, file=f)
    elif is_detection_output:
        with open('test_per_layer/main.cpp', 'w', encoding='utf-8', newline='') as f:
            print(f"""#include <array>
#include <fstream>
#include <iostream>

extern "C" {{
    void run_quantized_model(const unsigned char* input, unsigned char* output_class, float* output_detection, unsigned int batch_size);
    extern const unsigned char test_data[{np.size(image) * 2}];
}}

unsigned char output_class[1];
alignas(64) float output_buffer[{np.size(image) * 2 * 4}];

int main() {{
    run_quantized_model(test_data, output_class, output_buffer, 2);
    std::ofstream out("test_per_layer/results.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(output_buffer), {np.size(reference) * 4});
    return 0;
}}
""", file=f)
    else:
        with open('test_per_layer/main.cpp', 'w', encoding='utf-8', newline='') as f:
            print(f"""#include <array>
#include <fstream>
#include <iostream>

extern "C" {{
    void run_quantized_model(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);
    extern const unsigned char test_data[{np.size(image) * 2}];
}}

unsigned char output_class[1];
alignas(64) unsigned char output_buffer[{np.size(image) * 2 * 16}];

int main() {{
    run_quantized_model(test_data, output_class, output_buffer, 2);
    std::ofstream out("test_per_layer/results.bin", std::ios::binary);
    out.write(&reinterpret_cast<char*>(output_buffer)[{np.size(reference)}], {np.size(reference)});
    return 0;
}}
""", file=f)

    if os.path.exists('test_per_layer/results.bin'):
        os.remove('test_per_layer/results.bin')

    import subprocess
    subprocess.call(['g++', 'test_per_layer/test_data.s', 'test_per_layer/model.s', 'test_per_layer/main.cpp', '-march=silvermont', '-o', 'test_per_layer/test_app'])
    subprocess.call(['test_per_layer/test_app'])

    import struct
    with open('test_per_layer/results.bin', 'rb') as f:
        if is_classification_output:
            result = np.asarray(struct.unpack('=B', f.read())).reshape(reference.shape)
        elif is_detection_output:
            result = np.asarray(struct.unpack('=' + ''.join(('f',) * np.prod(reference.shape)), f.read())).reshape(reference.shape)
        else:
            result = np.asarray(struct.unpack('=' + ''.join(('B',) * np.prod(reference.shape)), f.read())).reshape(reference.shape)

    with np.printoptions(threshold=sys.maxsize):
        print(reference[:2])
        print(' ')
        print(result[:2])
        print(' ')
        print((reference - result)[:2])

    if not is_classification_output:
        import matplotlib.pyplot as plt
        sq_reference = np.squeeze(reference)
        sq_result = np.squeeze(result)
        max_value = np.max(np.maximum(reference, result))

        if np.ndim(sq_reference) > 2:
            fig, axes = plt.subplots(sq_reference.shape[-1], 2, sharex=True, sharey=True, squeeze=True)
            fig.set_size_inches(4.5, 10*sq_reference.shape[-1]/8)
        else:
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=True)
            fig.set_size_inches(5, 6)
        fig.tight_layout()

        if np.ndim(sq_reference) > 2:
            for c in range(reference.shape[-1]):
                axes[c, 0].imshow(sq_reference[...,c], vmin=0., vmax=max_value)
                axes[c, 1].imshow(sq_result[...,c], vmin=0., vmax=max_value)
        else:
            if np.ndim(sq_reference) == 1:
                sq_reference = sq_reference[:, None]
            if np.ndim(sq_result) == 1:
                sq_result = sq_result[:, None]
            axes[0].imshow(sq_reference, vmin=0., vmax=max_value)
            axes[1].imshow(sq_result, vmin=0., vmax=max_value)
        plt.savefig('test_per_layer/result.png')


def test_classification():
    test_data = tf.data.Dataset.load('balls_ds_validation', compression='GZIP')
    labels = test_data.batch(len(test_data)).get_single_element()[1][:,0].numpy()

    import os
    os.makedirs('test_classification', exist_ok=True)

    with open('test_classification/test_data.s', 'w', encoding='utf-8', newline='') as f:
        print(dataset_to_asm(test_data.as_numpy_iterator()), file=f)

    model = BallModel((0, 255), (0, 255), *TEST_MODEL_CONFIG)
    model.load_weights(TEST_MODEL_WEIGHTS_PATH).expect_partial()
    reference = np.ravel(np.argmax(model(test_data.batch(len(test_data)).get_single_element()[0], training=False)['scores'].numpy(), axis=-1).astype(np.uint8))
    print(reference.shape)

    with open('test_classification/model.s', 'w', encoding='utf-8', newline='') as f:
        print(ball_model_to_asm(model), file=f)

    with open('test_classification/main.cpp', 'w', encoding='utf-8', newline='') as f:
            print(f"""#include <array>
#include <fstream>
#include <iostream>

extern "C" {{
    void run_quantized_model(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);
    extern const unsigned char test_data[{len(test_data) * 32 * 32}];
}}

unsigned char output_class[{len(reference)}];
alignas(64) unsigned char output_buffer[{len(test_data) * 32 * 32 * 16}];

int main() {{
    run_quantized_model(test_data, output_class, output_buffer, {len(reference)});
    std::ofstream out("test_classification/results.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(output_class), {len(reference)});
    return 0;
}}
    """, file=f)

    if os.path.exists('test_classification/results.bin'):
        os.remove('test_classification/results.bin')

    import subprocess
    subprocess.call(['g++', 'test_classification/test_data.s', 'test_classification/model.s', 'test_classification/main.cpp', '-march=silvermont', '-o', 'test_classification/test_app'])
    subprocess.call(['test_classification/test_app'])

    import struct
    with open('test_classification/results.bin', 'rb') as f:
        result = np.asarray(struct.unpack('=' + ''.join(('B',) * np.prod(reference.shape)), f.read())).reshape(reference.shape)

    print(np.count_nonzero(reference))
    print(np.count_nonzero(result))

    print(np.count_nonzero(result == reference) / len(reference))

    try:
        from sklearn.metrics import f1_score
        print(f1_score(labels, reference))
        print(f1_score(labels, result))
    except ImportError:
        pass

if __name__ == '__main__':
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test_model(n_layers=n_layers)
    #test_classification()

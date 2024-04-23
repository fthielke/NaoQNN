# SPDX-License-Identifier: MIT
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, precision_recall_curve
from scipy.stats import ttest_rel
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from train_ball_model import MODEL_CONFIGURATIONS, BallModelEncoder, BallModelClassifier, BallModelDetector, load_data


matplotlib.rcParams['font.family'] = ['cmss10']
matplotlib.rcParams['mathtext.fontset'] = 'cm'


MEASURED_INFERENCE_TIME = {
    (BallModelEncoder.GERRIT_ORIGINAL, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 197.892,
    (BallModelEncoder.GERRIT_VALIDPAD, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 80.1793,
    (BallModelEncoder.GERRIT_VALIDPAD_SPLITCONVS, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 58.0327,
    (BallModelEncoder.V4_FLOAT, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 17.6513,
    (BallModelEncoder.GERRIT_ORIGINAL_QUANTIZED, BallModelClassifier.GERRIT_ORIGINAL_QUANTIZED, BallModelDetector.GERRIT_ORIGINAL_QUANTIZED): 162.288,
    (BallModelEncoder.GERRIT_VALIDPAD_QUANTIZED, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED): 40.6693,
    (BallModelEncoder.GERRIT_VALIDPAD_SPLITCONVS_QUANTIZED, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED): 28.3787,
    (BallModelEncoder.V4, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED): 7.29623,
}

MODEL_NAME = {
    (BallModelEncoder.GERRIT_ORIGINAL, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 'a)',
    (BallModelEncoder.GERRIT_VALIDPAD, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 'b)',
    (BallModelEncoder.GERRIT_VALIDPAD_SPLITCONVS, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 'c)',
    (BallModelEncoder.V4_FLOAT, BallModelClassifier.GERRIT_ORIGINAL, BallModelDetector.GERRIT_ORIGINAL): 'd)',
    (BallModelEncoder.GERRIT_ORIGINAL_QUANTIZED, BallModelClassifier.GERRIT_ORIGINAL_QUANTIZED, BallModelDetector.GERRIT_ORIGINAL_QUANTIZED): 'a) quantized',
    (BallModelEncoder.GERRIT_VALIDPAD_QUANTIZED, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED): 'b) quantized',
    (BallModelEncoder.GERRIT_VALIDPAD_SPLITCONVS_QUANTIZED, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED): 'c) quantized',
    (BallModelEncoder.V4, BallModelClassifier.GERRIT_QUANTIZED, BallModelDetector.GERRIT_QUANTIZED): 'd) quantized',
}

def recall_at_precision(y_true, y_scores, prec) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    try:
        return recall[np.arange(precision.shape[0])[precision >= prec][0]]
    except:
        return 0.

def mean_circle_iou(circles_true, circles_pred):
    grid = np.transpose(np.mgrid[0:32,0:32][::-1], axes=(1,2,0))

    y_true = circles_true[:,2,None,None] >= np.linalg.norm(grid[None,:,:,:] - circles_true[:,None,None,:2], axis=-1)
    y_pred = circles_pred[:,2,None,None] >= np.linalg.norm(grid[None,:,:,:] - circles_pred[:,None,None,:2], axis=-1)
    tp = np.count_nonzero(np.logical_and(y_true, y_pred), axis=(1, 2))
    fp_plus_fn = np.count_nonzero(np.logical_xor(np.logical_not(y_true), np.logical_not(y_pred)), axis=(1, 2))
    iou = tp / (tp + fp_plus_fn)
    return np.mean(iou)


def calculate_metrics(model_config, test):
    # results_original = np.load('weights/gerrit_original/gerrit_original/gerrit_original/results_test.npz')
    try:
        results = np.load('weights/' + '/'.join(str(c.value) for c in model_config) + '/results_test.npz')
    except:
        return
    
    print(f'{model_config}:')

    # auc_original = np.fromiter((roc_auc_score(test[:,0], results_original['predictions'][i]) for i in range(results_original['predictions'].shape[0])), dtype=float)
    auc = np.fromiter((roc_auc_score(test[:,0], results['predictions'][i]) for i in range(results['predictions'].shape[0])), dtype=float)
    print(f' AUC: {np.mean(auc)} +- {np.std(auc)}')
    # print(f'  {ttest_rel(auc_original, auc, alternative="greater")}')

    r_at_p100 = np.fromiter((recall_at_precision(test[:,0], results['predictions'][i], prec=1.) for i in range(results['predictions'].shape[0])), dtype=float)
    print(f' Recall@P100: {np.mean(r_at_p100)} +- {np.std(r_at_p100)}')

    pos_mask = test[:,0] > 0
    circ_pred = results['circles'][:,pos_mask]
    circ_test = test[pos_mask,1:]
    mae_center = np.fromiter((mean_absolute_error(np.zeros(circ_test.shape[0]), np.linalg.norm(circ_test[:,:2] - circ_pred[i,:,:2], axis=1)) for i in range(circ_pred.shape[0])), dtype=float)
    print(f' Center MAE: {np.mean(mae_center)} +- {np.std(mae_center)}')

    mae_radius = np.fromiter((mean_absolute_error(circ_test[:,2], circ_pred[i,:,2]) for i in range(circ_pred.shape[0])), dtype=float)
    print(f' Radius MAE: {np.mean(mae_radius)} +- {np.std(mae_radius)}')

    circle_iou = np.fromiter((mean_circle_iou(circ_test, circ_pred[i]) for i in range(circ_pred.shape[0])), dtype=float)
    print(f' Mean circle IoU: {np.mean(circle_iou)} +- {np.std(circle_iou)}')


def create_latex_table(test):
    latex = r"""\begin{table}
    \centering
    \caption{Resulting metrics of the trained models on the test set.}\label{tab:results}
    \begin{tabular}{llrrr}
        \hline
        \multicolumn{2}{l}{Architecture} & \multicolumn{1}{l}{AUROC} & \multicolumn{1}{l}{Mean ball IoU} & \multicolumn{1}{l}{Latency ($\mu{}s$)} \\
        \hline"""
    
    pos_mask = test[:,0] > 0
    circ_test = test[pos_mask,1:]

    for model in (model_config for i in range(4) for model_config in MODEL_CONFIGURATIONS[i::4]):
        results = np.load('weights/' + '/'.join(str(c.value) for c in model) + '/results_test.npz')

        circ_pred = results['circles'][:,pos_mask]

        auc = np.fromiter((roc_auc_score(test[:,0], results['predictions'][i]) for i in range(results['predictions'].shape[0])), dtype=float)
        circle_iou = np.fromiter((mean_circle_iou(circ_test, circ_pred[i]) for i in range(circ_pred.shape[0])), dtype=float)
        inference_time = MEASURED_INFERENCE_TIME[model]
        name = MODEL_NAME[model] if 'quan' not in MODEL_NAME[model] else ''
        model_type = 'float' if 'quan' not in MODEL_NAME[model] else 'quantized'

        latex += f"""
{name} & {model_type} & ${np.mean(auc):0.4f} \\pm {np.std(auc):0.4f}$ & ${np.mean(circle_iou):0.4f} \\pm {np.std(circle_iou):0.4f}$ & ${inference_time:0.3f}$ \\\\"""

    latex += """
        \hline
    \end{tabular}
\end{table}"""

    print(latex)


def create_plots(test):
    pos_mask = test[:,0] > 0
    circ_test = test[pos_mask,1:]

    data = []
    for model_config in MODEL_CONFIGURATIONS:
        if not MEASURED_INFERENCE_TIME[model_config]:
            continue
        results = np.load('weights/' + '/'.join(str(c.value) for c in model_config) + '/results_test.npz')
        circ_pred = results['circles'][:,pos_mask]
        for i, (auc, circle_iou) in enumerate((roc_auc_score(test[:,0], results['predictions'][i]), mean_circle_iou(circ_test, circ_pred[i])) for i in range(results['predictions'].shape[0])):
            data.append({
                'name': MODEL_NAME[model_config],
                'Inference time [$\\mu$s]': MEASURED_INFERENCE_TIME[model_config],
                'Model type': 'quantized' if 'quantized' in MODEL_NAME[model_config] else 'float',
                'AUROC': auc,
                'Mean ball IoU': circle_iou,
                'model_index': i
            })
    data = pd.DataFrame(data)

    fig = plt.figure(figsize=(4.5,3.5))
    ax = sns.pointplot(
        data=data,
        orient='h',
        y='Inference time [$\\mu$s]',
        x='AUROC',
        hue='Model type',
        linestyles='none',
        #errorbar='sd',
        errorbar=('ci', 95),
        markers=['o', 'D'],
        markersize=4,
        native_scale=True
    )
    #ax.set_xlim(0.98, 1.)
    ax.set_ylim(0, 225)
    plt.savefig('plot_time_auc.pdf')

    fig = plt.figure(figsize=(4.5,3.5))
    fig.tight_layout()
    ax = sns.pointplot(
        data=data,
        orient='h',
        y='Inference time [$\\mu$s]',
        x='Mean ball IoU',
        hue='Model type',
        linestyles='none',
        #errorbar='sd',
        errorbar=('ci', 95),
        markers=['o', 'D'],
        markersize=4,
        native_scale=True
    )
    # ax.set_xlim(0.7, 1.)
    ax.set_ylim(0, 225)
    plt.savefig('plot_time_meaniou.pdf')

def create_diff_plots(test):
    pos_mask = test[:,0] > 0
    circ_test = test[pos_mask,1:]

    data = []

    for float_model, quan_model in zip(MODEL_CONFIGURATIONS[:4], MODEL_CONFIGURATIONS[4:]):
        results_float = np.load('weights/' + '/'.join(str(c.value) for c in float_model) + '/results_test.npz')
        results_quan = np.load('weights/' + '/'.join(str(c.value) for c in quan_model) + '/results_test.npz')

        circ_pred_float = results_float['circles'][:,pos_mask]
        circ_pred_quan = results_quan['circles'][:,pos_mask]

        for float_auc, float_circle_iou in ((roc_auc_score(test[:,0], results_float['predictions'][i]), mean_circle_iou(circ_test, circ_pred_float[i])) for i in range(results_float['predictions'].shape[0])):
            for quan_auc, quan_circle_iou in ((roc_auc_score(test[:,0], results_quan['predictions'][i]), mean_circle_iou(circ_test, circ_pred_quan[i])) for i in range(results_quan['predictions'].shape[0])):
                data.append({
                    'Model': MODEL_NAME[float_model],
                    'name_quan': MODEL_NAME[quan_model],
                    'Inference time reduction factor': MEASURED_INFERENCE_TIME[float_model] / MEASURED_INFERENCE_TIME[quan_model],
                    'AUROC reduction': float_auc - quan_auc,
                    'Mean ball IoU reduction': float_circle_iou - quan_circle_iou,
                })
    data = pd.DataFrame(data)

    fig = plt.figure(figsize=(4.5,3.5))
    ax = sns.pointplot(
        data=data,
        orient='v',
        x='Model',
        y='Inference time reduction factor',
        linestyles='none',
        errorbar=('ci', 95),
        markersize=4
    )
    #ax.set_xlim(0.98, 1.)
    #ax.set_ylim(0, 225)
    plt.savefig('plot_diff_time.pdf')

    fig = plt.figure(figsize=(4.5,3.5))
    ax = sns.pointplot(
        data=data,
        orient='v',
        x='Model',
        y='AUROC reduction',
        linestyles='none',
        errorbar=('ci', 95),
        markersize=4
    )
    #ax.set_xlim(0.98, 1.)
    #ax.set_ylim(0, 225)
    plt.savefig('plot_diff_auc.pdf')

    fig = plt.figure(figsize=(4.5,3.5))
    ax = sns.pointplot(
        data=data,
        orient='v',
        x='Model',
        y='Mean ball IoU reduction',
        linestyles='none',
        errorbar=('ci', 95),
        markersize=4
    )
    #ax.set_xlim(0.98, 1.)
    #ax.set_ylim(0, 225)
    plt.savefig('plot_diff_meaniou.pdf')


def quantized_t_tests(test):
    pos_mask = test[:,0] > 0
    circ_test = test[pos_mask,1:]

    for float_model, quan_model in zip(MODEL_CONFIGURATIONS[:4], MODEL_CONFIGURATIONS[4:]):
        results_float = np.load('weights/' + '/'.join(str(c.value) for c in float_model) + '/results_test.npz')
        results_quan = np.load('weights/' + '/'.join(str(c.value) for c in quan_model) + '/results_test.npz')

        circ_pred_float = results_float['circles'][:,pos_mask]
        circ_pred_quan = results_quan['circles'][:,pos_mask]

        float_auc = np.fromiter((roc_auc_score(test[:,0], results_float['predictions'][i]) for i in range(results_float['predictions'].shape[0])), dtype=float)
        quan_auc = np.fromiter((roc_auc_score(test[:,0], results_quan['predictions'][i]) for i in range(results_quan['predictions'].shape[0])), dtype=float)
        
        float_mean_circle_iou = np.fromiter((mean_circle_iou(circ_test, circ_pred_float[i]) for i in range(circ_pred_float.shape[0])), dtype=float)
        quan_mean_circle_iou = np.fromiter((mean_circle_iou(circ_test, circ_pred_quan[i]) for i in range(circ_pred_quan.shape[0])), dtype=float)

        print(MODEL_NAME[float_model])
        print(f' AUC: {ttest_rel(float_auc, quan_auc, alternative="greater")}')
        print(f' IoU: {ttest_rel(float_mean_circle_iou, quan_mean_circle_iou, alternative="greater")}')
        print('')


if __name__ == '__main__':
    train, val, test = load_data()
    del train
    del val
    test = test.batch(len(test)).get_single_element()[1].numpy()

    for model_config in MODEL_CONFIGURATIONS:
        calculate_metrics(model_config, test)
    create_plots(test)
    create_diff_plots(test)
    create_latex_table(test)
    quantized_t_tests(test)


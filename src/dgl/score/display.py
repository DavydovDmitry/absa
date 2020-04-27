import os
from typing import List

import matplotlib.pyplot as plt

from src import images_path

SCORE_NAME = 'accuracy'
PARAMETER_DECIMAL_LEN = 5
SCORE_DECIMAL_LEN = 3


def display_score(parameter_values: List,
                  train_acc: List[float],
                  val_acc: List[float],
                  parameter_name='Epochs',
                  score_name=SCORE_NAME) -> float:

    max_param, max_acc = [(parameter_values[index], val) for index, val in enumerate(val_acc)
                          if val == max(val_acc)][0]

    plt.figure(figsize=(8, 8))
    plt.grid(True, alpha=0.3)
    plt.xlim(left=min(parameter_values), right=max(parameter_values))
    plt.plot([min(parameter_values), max(parameter_values)], [max_acc, max_acc], ':r')

    # train
    plt.plot(parameter_values, train_acc, '-c')
    plt.plot(parameter_values, train_acc, 'ob', alpha=0.5, markersize=5)

    # val
    plt.plot(parameter_values, val_acc, '-c')
    plt.plot(parameter_values, val_acc, 'ob', alpha=0.5, markersize=5)

    # Decoration
    plt.title(f'Dependence of {score_name} from {parameter_name}')
    plt.xlabel(f'{parameter_name}')
    plt.ylabel(score_name.capitalize())
    if isinstance(max_param, (int, )):
        plt.legend([
            f'Maximal {score_name}={max_acc:.{SCORE_DECIMAL_LEN}} when {parameter_name}={max_param}'
        ])
    else:
        plt.legend([
            f'Maximal {score_name}={max_acc:.{SCORE_DECIMAL_LEN}}' +
            f' when {parameter_name}={max_param:.{PARAMETER_DECIMAL_LEN}f}'
        ])

    filename = 'polarity_classifier.jpg'
    plt.savefig(os.path.join(images_path, filename))
    return max_param

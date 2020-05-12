import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src import images_path

SCORE_NAME = 'accuracy'
PARAMETER_DECIMAL_LEN = 5
SCORE_DECIMAL_LEN = 3


def display_score(parameter_values: List,
                  train_values: List[float],
                  val_values: List[float],
                  parameter_name='epoch',
                  score_name=SCORE_NAME) -> float:

    max_param, max_acc = [(parameter_values[index], val)
                          for index, val in enumerate(val_values) if val == max(val_values)][0]

    plt.figure(figsize=(8, 8))
    plt.grid(True, alpha=0.3)
    plt.xlim(left=min(parameter_values), right=max(parameter_values))
    plt.plot([min(parameter_values), max(parameter_values)], [max_acc, max_acc], ':r')

    # train
    plt.plot(parameter_values, train_values, color='azure')
    plt.plot(parameter_values, train_values, color='blue', marker='o', alpha=0.5, markersize=5)
    # validation
    plt.plot(parameter_values, val_values, color='orange')
    plt.plot(parameter_values,
             val_values,
             color='orangered',
             marker='o',
             alpha=0.5,
             markersize=5)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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

    filename = 'polarity_classifier_' + parameter_name + '.jpg'
    plt.savefig(os.path.join(images_path, filename))
    return max_param

# Project Name: C:\Users\Lenovo\Documents\code\py\university-2019

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_graph_loss(file_name, model_name):
    values = pd.read_table(file_name, sep=',')
    data = pd.DataFrame()
    data['epoch'] = list(values['epoch'].get_values() + 1) + \
                    list(values['epoch'].get_values() + 1)
    data['loss name'] = ['training'] * len(values) + \
                        ['validation'] * len(values)
    data['loss'] = list(values['loss'].get_values()) + \
                   list(values['val_loss'].get_values())
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    sns.lineplot(x='epoch', y='loss', hue='loss name', style='loss name',
                 dashes=False, data=data, palette='Set2')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend().texts[0].set_text('')
    plt.title(model_name)
    plt.show()


def main():
    print('casual.py')
    print('')
    print('')
    print()


if __name__ == '__main__':
    main()

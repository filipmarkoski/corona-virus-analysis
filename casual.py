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


def couple(tu):
    pass


def fill(M, rna):
    """
    Fill the matrix as per the Nussinov algorithm
    """
    minimal_loop_length = 0

    for k in range(1, len(rna)):
        for i in range(len(rna) - k):
            j = i + k

            if j - i >= minimal_loop_length:
                down = M[i + 1][j]  # 1st rule
                left = M[i][j - 1]  # 2nd rule
                diag_left_down = M[i + 1][j - 1] + couple((rna[i], rna[j]))  # 3rd rule

                rc = max([M[i][t] + M[t + 1][j] for t in range(i, j)])  # 4th rule

                M[i][j] = max(down, left, diag_left_down, rc)  # max of all

            else:
                M[i][j] = 0

    return M


def initialize_nussinov_matrix(M, rna):
    """
    Fill the matrix as per the Nussinov algorithm
    """
    L = len(rna)

    for n in range(2, L):
        for j in range(n, L):
            i = j - n + 1

            down = M[i + 1][j]  # 1st rule
            left = M[i][j - 1]  # 2nd rule
            diag_left_down = M[i + 1][j - 1] + complementary(rna[i], rna[j])  # 3rd rule
            rc = max([M[i][k] + M[k + 1][j] for k in range(i, j)])  # 4th rule

            M[i][j] = max(down, left, diag_left_down, rc)  # max of all

    return M


def complementary(a, b):
    pass


def traceback(M, rna, i, L, fold=None):
    """
    Traceback through complete Nussinov matrix to find optimial RNA secondary structure solution through max base-pairs
    """
    if fold is None:
        fold = []

    j = L

    if i < j:
        if M[i][j] == M[i + 1][j]:  # 1st rule
            traceback(M, rna, i + 1, j, fold=fold)
        elif M[i][j] == M[i][j - 1]:  # 2nd rule
            traceback(M, rna, i, j - 1, fold=fold)
        elif M[i][j] == M[i + 1][j - 1] + complementary(rna[i], rna[j]):  # 3rd rule
            fold.append((i, j))
            traceback(M, rna, i + 1, j - 1, fold=fold)
        else:
            for k in range(i + 1, j - 1):
                if M[i][j] == M[i, k] + M[k + 1][j]:  # 4th rule
                    traceback(M, rna, i, k, fold=fold)
                    traceback(M, rna, k + 1, j, fold=fold)
                    break

    return fold


def main():
    print('casual.py')
    print('')
    print('')
    print()


if __name__ == '__main__':
    main()

import torch
import numpy as np
import matplotlib.pyplot as plt

def save_word_dist_plot(beta, vocab, save_name, top_n=None):
    if isinstance(beta, torch.Tensor):
        beta = beta.cpu().detach().numpy()
    sorted_indices = beta.argsort()[::-1]

    if top_n == None:
        top_n = len(sorted_indices)

    beta = beta[sorted_indices][:top_n]
    vocab = vocab[sorted_indices][:top_n]
    plt.bar(vocab, beta, color='blue')
    plt.xticks([])
    plt.savefig(save_name)
    plt.clf()

def save_histogram(data, save_name):
    plt.hist(data, bins=100)
    plt.savefig(save_name)
    plt.clf()
import matplotlib.pyplot as plt


def plot_training(train_losses, train_roc_aucs, val_losses, val_roc_aucs, model_name="", return_fig=True):
    """
    Plot losses and ROC AUC over the training process.

    train_losses (list): List of training losses over training.
    train_roc_aucs (list): List of training ROC AUCs over training.
    val_losses (list): List of validation losses over training.
    val_roc_aucs (list): List of validation ROC AUCs over training.
    model_name (str): Name of model as a string.
    return_fig (Boolean): Whether to return figure or not.
    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    ax1.plot(val_losses, label='Validation loss')
    ax1.plot(train_losses, label="Training loss")
    ax1.set_title('Loss over training for {}'.format(model_name), fontsize=20)
    ax1.set_xlabel("epoch", fontsize=18)
    ax1.set_ylabel("loss", fontsize=18)
    ax1.legend()

    ax2.plot(val_roc_aucs, label='Validation ROC AUC')
    ax2.plot(train_roc_aucs, label='Training ROC AUC')
    ax2.set_title('ROC AUC over training for {}'.format(model_name), fontsize=20)
    ax2.set_xlabel("epoch", fontsize=18)
    ax2.set_ylabel("ROC AUC", fontsize=18)
    ax2.legend()

    fig.tight_layout()
    if return_fig:
        return fig

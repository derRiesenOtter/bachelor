import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def run_train_eval(
    model_name,
    model,
    device,
    epochs,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    val_df,
):
    model = model.to(device)

    epoch_number = 0
    best_vloss = 1000000
    patience = 10
    patience_counter = 0

    train_loss_list = []
    val_loss_list = []
    all_preds = []
    all_probs = []
    all_labels = []

    for epoch in range(epochs):
        print("EPOCH {}:".format(epoch_number + 1))
        model.train(True)
        avg_loss = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        train_loss_list.append(avg_loss)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        all_preds_tmp = []
        all_probs_tmp = []
        all_labels_tmp = []
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                _, preds = torch.max(voutputs, 1)
                probs = torch.softmax(voutputs, dim=1)
                all_preds_tmp.extend(preds.cpu().numpy())
                all_probs_tmp.extend(probs.cpu().numpy())
                all_labels_tmp.extend(vlabels.cpu().numpy())

        avg_vloss = running_vloss / len(val_loader)
        val_loss_list.append(avg_vloss)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            all_preds = all_preds_tmp
            all_probs = all_probs_tmp
            all_labels = all_labels_tmp
            patience_counter = 0
            best_vloss = avg_vloss
            model_path = f"./data/processed_data/model_{model_name}"
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

        epoch_number += 1

    train_loss_list_cpu = [
        loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss
        for loss in train_loss_list
    ]
    val_loss_list_cpu = [
        loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss
        for loss in val_loss_list
    ]

    # create a loss over time plot
    plt.scatter(
        np.arange(1, 1 + len(train_loss_list_cpu)),
        train_loss_list_cpu,
        color="r",
        label="train_loss",
    )
    plt.scatter(
        np.arange(1, 1 + len(val_loss_list_cpu)),
        val_loss_list_cpu,
        color="b",
        label="Validation Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_loss")
    plt.close()

    n_classes = model.fc1.out_features
    y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    y_score = np.array(all_probs)

    create_rocauc(n_classes, y_true_bin, y_score, model_name)

    print(classification_report(all_labels, all_preds, digits=3))

    # create a confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"./results/plots/{model_name}_cm")
    plt.close()

    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        pr_auc = average_precision_score(y_true_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"Class {i} (AP = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_prauc")
    plt.close()


def create_rocauc(n_classes, y_true_bin, y_score, model_name):
    plt.figure()
    for i in range(1, n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    y_true_combined = np.any(y_true_bin[:, 1:], axis=1).astype(int)
    y_score_combined = np.max(y_score[:, 1:], axis=1)
    fpr, tpr, _ = roc_curve(y_true_combined, y_score_combined)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        label=f"Combined Classes 1-{n_classes-1} (AUC = {roc_auc:.2f})",
        linestyle=":",
        linewidth=3,
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.plot([0, 1], [0, 1], "k--")
    plt.savefig(f"./results/plots/{model_name}_rocauc")
    plt.close()


def train_one_epoch(model, train_loader, device, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:
            last_loss = running_loss / 20
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0

    return last_loss

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


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
    patience = 3
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
                probs = torch.sigmoid(voutputs)
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

    # create a confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"./results/plots/{model_name}_cm")
    plt.close()

    all_probs = np.array(all_probs)[:, 1]

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_rocauc")
    plt.close()

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_prauc")
    plt.close()

    # create the plots for proteins containing idrs and no idrs separetely
    if "pspire" in model_name:
        idr_probs = np.array(all_probs)[
            (val_df["idr_protein"] == 1) | (val_df["ps_label"] == 0)
        ]
        idr_labels = np.array(all_labels)[
            (val_df["idr_protein"] == 1) | (val_df["ps_label"] == 0)
        ]
        idr_preds = np.array(all_preds)[
            (val_df["idr_protein"] == 1) | (val_df["ps_label"] == 0)
        ]
    else:
        idr_probs = np.array(all_probs)[val_df["idr_protein"] == 1]
        idr_labels = np.array(all_labels)[val_df["idr_protein"] == 1]
        idr_preds = np.array(all_preds)[val_df["idr_protein"] == 1]

    nidr_probs = np.array(all_probs)[val_df["idr_protein"] == 0]
    nidr_labels = np.array(all_labels)[val_df["idr_protein"] == 0]
    nidr_preds = np.array(all_preds)[val_df["idr_protein"] == 0]

    # create a confusion_matrix
    cm = confusion_matrix(idr_labels, idr_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"./results/plots/{model_name}_cm_idr")
    plt.close()

    fpr, tpr, _ = roc_curve(idr_labels, idr_probs)
    roc_auc = roc_auc_score(idr_labels, idr_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_rocauc_idr")
    plt.close()

    precision, recall, _ = precision_recall_curve(idr_labels, idr_probs)
    pr_auc = average_precision_score(idr_labels, idr_probs)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_prauc_idr")
    plt.close()

    # create a confusion_matrix
    cm = confusion_matrix(nidr_labels, nidr_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"./results/plots/{model_name}_cm_nidr")
    plt.close()

    fpr, tpr, _ = roc_curve(nidr_labels, nidr_probs)
    roc_auc = roc_auc_score(nidr_labels, nidr_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_rocauc_nidr")
    plt.close()

    precision, recall, _ = precision_recall_curve(nidr_labels, nidr_probs)
    pr_auc = average_precision_score(nidr_labels, nidr_probs)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"./results/plots/{model_name}_prauc_nidr")
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

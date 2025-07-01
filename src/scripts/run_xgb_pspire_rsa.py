import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.modules.bd_tools import get_scalar_values_rsa
from src.modules.Words.Mappings import (
    APNAMapping,
    IDRMapping,
    MM5Mapping,
    PIPIFMapping,
    PIPIGMapping,
    RG2Mapping,
    RGMapping,
)

mappings = [
    APNAMapping,
    RGMapping,
    RG2Mapping,
    PIPIFMapping,
    PIPIGMapping,
    MM5Mapping,
    IDRMapping,
]


def main():
    model_name = Path(__file__).stem

    # Opening the data containing the mapped sequences
    with open("./data/intermediate_data/pspire_bd.pkl", "rb") as f:
        df = pickle.load(f)

        [col for col in df.columns]

    mapping_names = [mapping.__name__ for mapping in mappings]
    block_seqs = [block_seq for block_seq in df.columns if "vec" in block_seq]

    counter = 0
    map_counter = 0
    df_list = []
    for block_seq in block_seqs:
        counter += 1
        block_seq_list = []
        for _, row in df.iterrows():
            dic = get_scalar_values_rsa(
                row["UniprotEntry"],
                row[block_seq],
                mappings[map_counter].Forward,
                "./data/intermediate_data/pspire_rsa.pkl",
            )
            dic = {f"{key}_{block_seq}": value for key, value in dic.items()}
            dic["id"] = row["UniprotEntry"]
            dic["Datasets"] = row["Datasets"]
            dic["ps_label"] = row["ps_label"]
            dic["idr_protein"] = row["idr_protein"]
            block_seq_list.append(dic)
        df_list.append(pd.DataFrame(block_seq_list))
        if counter % 2 == 0:
            map_counter += 1

    df = df_list[0]
    for df_to_merge in df_list[1:]:
        df = pd.merge(df, df_to_merge, on=["id", "Datasets", "ps_label", "idr_protein"])

    train_df = df.loc[df["Datasets"] == "Training"]
    val_df = df.loc[df["Datasets"] == "Testing"]
    y_train = train_df["ps_label"]
    y_test = val_df["ps_label"]
    X_train = train_df.drop(columns=["id", "Datasets", "ps_label", "idr_protein"])
    X_test = val_df.drop(columns=["id", "Datasets", "ps_label", "idr_protein"])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    # random_state=13)

    # Use "hist" for constructing the trees, with early stopping enabled.
    clf = xgb.XGBClassifier(
        tree_method="hist",
        learning_rate=0.05,
        n_estimators=1000,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_lambda=1,
        reg_alpha=0.5,
        early_stopping_rounds=100,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    # Save model into JSON format.
    # Predict probabilities and classes
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    y_proba_idr = y_proba[(val_df["idr_protein"] == 1) | (val_df["ps_label"] == 0)]
    y_test_idr = y_test[(val_df["idr_protein"] == 1) | (val_df["ps_label"] == 0)]

    y_proba_nidr = y_proba[val_df["idr_protein"] == 0]
    y_test_nidr = y_test[val_df["idr_protein"] == 0]
    # -------------------- ROC Curve --------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"./results/plots/{model_name}_rocauc")
    plt.close()

    # -------------------- PR-AUC Curve --------------------
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"./results/plots/{model_name}_prauc")
    plt.close()

    # -------------------- Confusion Matrix --------------------
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"./results/plots/{model_name}_cm")
    plt.close()

    xgb.plot_importance(clf, max_num_features=20, importance_type="gain")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"./results/plots/{model_name}_importance")
    plt.close()

    clf.save_model(f"./data/processed_data/{model_name}.json")

    # -------------------- ROC Curve idr--------------------
    fpr, tpr, _ = roc_curve(y_test_idr, y_proba_idr)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"./results/plots/{model_name}_rocauc_idr")
    plt.close()

    # -------------------- PR-AUC Curve idr--------------------
    precision, recall, _ = precision_recall_curve(y_test_idr, y_proba_idr)
    pr_auc = average_precision_score(y_test_idr, y_proba_idr)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"./results/plots/{model_name}_prauc_idr")
    plt.close()

    # -------------------- ROC Curve nidr--------------------
    fpr, tpr, _ = roc_curve(y_test_nidr, y_proba_nidr)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"./results/plots/{model_name}_rocauc_nidr")
    plt.close()

    # -------------------- PR-AUC Curve nidr--------------------
    precision, recall, _ = precision_recall_curve(y_test_nidr, y_proba_nidr)
    pr_auc = average_precision_score(y_test_nidr, y_proba_nidr)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"./results/plots/{model_name}_prauc_nidr")
    plt.close()


if __name__ == "__main__":
    main()

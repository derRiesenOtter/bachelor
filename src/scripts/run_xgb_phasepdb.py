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

from src.modules.bd_tools import get_scalar_values
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
    with open("./data/intermediate_data/phasepdb_bd.pkl", "rb") as f:
        df = pickle.load(f)

    mapping_names = [mapping.__name__ for mapping in mappings]
    block_seqs = [block_seq for block_seq in df.columns if "vec" in block_seq]

    counter = 0
    map_counter = 0
    df_list = []
    for block_seq in block_seqs:
        counter += 1
        block_seq_list = []
        for _, row in df.iterrows():
            dic = get_scalar_values(row[block_seq], mappings[map_counter].Forward)
            dic = {f"{key}_{block_seq}": value for key, value in dic.items()}
            dic["id"] = row["UniprotEntry"]
            dic["type"] = row["type"]
            dic["ps_label"] = row["ps_label"]
            block_seq_list.append(dic)
        df_list.append(pd.DataFrame(block_seq_list))
        if counter % 2 == 0:
            map_counter += 1

    df = df_list[0]
    for df_to_merge in df_list[1:]:
        df = pd.merge(df, df_to_merge, on=["id", "type", "ps_label"])

    train_df = df.loc[df["type"] == "train"]
    val_df = df.loc[df["type"] == "test"]
    y_train = train_df["ps_label"]
    y_test = val_df["ps_label"]
    X_train = train_df.drop(columns=["id", "type", "ps_label"])
    X_test = val_df.drop(columns=["id", "type", "ps_label"])

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


if __name__ == "__main__":
    main()

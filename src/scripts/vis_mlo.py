import matplotlib.pyplot as plt
import pandas as pd

rocauc_idr = pd.DataFrame(
    {
        "Data_Set": [],
        "ROC-AUC_idr": [],
        "PR-AUC_idr": [],
        "ROC-AUC_nidr": [],
        "PR-AUC_nidr": [],
    }
)

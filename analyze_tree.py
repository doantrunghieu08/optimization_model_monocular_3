import ast
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

# ponytail: depth-3 regressor directly on report CSV; upgrade to random forest / cross-val only if tree overfits
def analyze_tree(csv_path="output/brute_force_report.csv", max_depth=3):
    df = pd.read_csv(csv_path)
    df = df[df["Set"] != "End"].copy()

    for col in ["% delta_pa-mpjpe", "% delta_mpjpe", "% delta_accel", "pa-mpjpe.all_baseline", "mpjpe.all_baseline", "accel.baseline", "Alpha", "Beta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["b_master"] = df["belief Master"].apply(lambda x: np.mean(ast.literal_eval(x)) if pd.notna(x) else 0)
    df["b_slave"] = df["belief Slave"].apply(lambda x: np.mean(ast.literal_eval(x)) if pd.notna(x) else 0)
    df["b_diff"] = df["b_slave"] - df["b_master"]

    features = ["pa-mpjpe.all_baseline", "mpjpe.all_baseline", "accel.baseline", "Alpha", "Beta", "b_master", "b_slave", "b_diff"]
    X = df[features].fillna(0)
    y = df["% delta_pa-mpjpe"].fillna(0)

    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X, y)

    print("=== DECISION TREE RULES (Target: % delta_pa-mpjpe) ===")
    print(export_text(tree, feature_names=features))
    print("=== FEATURE IMPORTANCES ===")
    for name, imp in sorted(zip(features, tree.feature_importances_), key=lambda x: -x[1]):
        if imp > 0:
            print(f"  {name:25s}: {imp:.3f}")
    return tree

if __name__ == "__main__":
    analyze_tree()


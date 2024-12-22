import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_feature_importance(
    df: pd.DataFrame,
    feature_name_col: str = "feature_name",
    feature_importance_col: str = "feature_importance",
    fold_col: str = "fold",
    top_k: int = None,
) -> plt.Figure:
    # Determine if it's a single fold or multiple folds
    if fold_col not in df.columns:
        is_single_fold = True
    else:
        is_single_fold = df[fold_col].nunique() == 1

    # Filter the top_k features if specified
    if top_k is not None:
        # Compute mean importance to determine the top_k features
        df_mean = df.groupby(feature_name_col, as_index=False)[feature_importance_col].mean()
        top_features = df_mean.nlargest(top_k, feature_importance_col)[feature_name_col].tolist()
        df = df[df[feature_name_col].isin(top_features)]

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))
    if is_single_fold:
        # Barplot for a single fold
        df_mean = df.groupby(feature_name_col, as_index=False)[feature_importance_col].mean()
        df_mean = df_mean.sort_values(by=feature_importance_col, ascending=False)

        sns.barplot(
            data=df_mean,
            x=feature_importance_col,
            y=feature_name_col,
            orient="h",
            ax=ax,
        )
        ax.set_title("Feature Importance (Single Fold)", fontsize=16)
    else:
        # Boxplot for multiple folds
        df_sorted = df.groupby(feature_name_col, as_index=False)[feature_importance_col].mean()
        feature_order = df_sorted.sort_values(by=feature_importance_col, ascending=False)[feature_name_col]

        sns.boxplot(
            data=df,
            x=feature_importance_col,
            y=feature_name_col,
            order=feature_order,
            ax=ax,
        )
        ax.set_title("Feature Importance Across Folds", fontsize=16)

    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig

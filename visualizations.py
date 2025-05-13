# === visualizations.py ===
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution des rendements
def plot_yield_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df["Predicted_Yield"], bins=20, kde=True, color="green", ax=ax)
    ax.set_title("Distribution of Predicted Yields")
    ax.set_xlabel("Yield (quintals/ha)")
    ax.set_ylabel("Frequency")
    return fig

# Camembert des fréquences de rendement
def plot_yield_pie(df):
    if "Predicted_Yield" in df.columns:
        bins = [0, 10, 20, 30, 50, 100]
        labels = ["<10", "10–20", "20–30", "30–50", ">50"]
        df["Yield_Bin"] = pd.cut(df["Predicted_Yield"], bins=bins, labels=labels, right=False)
        counts = df["Yield_Bin"].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Répartition des rendements prédits")
        return fig
    return None

# Courbe de tendance temporelle
def plot_yield_over_time(df):
    if "timestamp" in df.columns and "predicted_yield" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        fig, ax = plt.subplots()
        sns.lineplot(x="timestamp", y="predicted_yield", data=df, ax=ax, marker="o")
        ax.set_title("Évolution du rendement prédits dans le temps")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Rendement (qx/ha)")
        return fig
    return None

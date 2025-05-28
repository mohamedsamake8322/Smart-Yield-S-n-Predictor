# === visualizations.py ===
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

print("🚀 Script visualizations.py started...")

# ✅ Yield distribution histogram
def plot_yield_distribution(df):
    if "PredictedYield" not in df.columns:
        raise ValueError("❌ Column 'PredictedYield' is missing in DataFrame")

    fig, ax = plt.subplots()
    sns.histplot(df["PredictedYield"], bins=20, kde=True, color="green", ax=ax)
    ax.set_title("📊 Distribution of Predicted Yields")
    ax.set_xlabel("Yield (tons/ha)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

# ✅ Yield frequency pie chart
def plot_yield_pie(df):
    if "PredictedYield" not in df.columns:
        raise ValueError("❌ Column 'PredictedYield' is missing in DataFrame")

    bins = [0, 10, 20, 30, 50, 100]
    labels = ["<10", "10–20", "20–30", "30–50", ">50"]
    df["yield_bin"] = pd.cut(df["PredictedYield"], bins=bins, labels=labels, right=False)

    counts = df["yield_bin"].value_counts().sort_index()
    colors = sns.color_palette("pastel")  # ✅ Ajout de couleurs pour améliorer la lisibilité

    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title("🎂 Predicted Yield Distribution (Pie Chart)")
    fig.tight_layout()
    return fig

# ✅ Yield trend over time
def plot_yield_over_time(df):
    if "timestamp" not in df.columns or "PredictedYield" not in df.columns:
        raise ValueError("❌ Columns 'timestamp' and 'PredictedYield' are required in DataFrame")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")  # ✅ Gestion avancée des dates
    df = df.sort_values("timestamp")

    fig, ax = plt.subplots()
    sns.lineplot(x="timestamp", y="PredictedYield", data=df, ax=ax, marker="o")
    ax.set_title("📈 Predicted Yield Trend Over Time")
    ax.set_xlabel("📅 Date")
    ax.set_ylabel("🌾 Yield (tons/ha)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

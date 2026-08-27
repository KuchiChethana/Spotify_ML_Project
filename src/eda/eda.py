print("--------------------- SPOTIFY EDA & VISUALIZATION ---------------------")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

TRACKS_PATH = os.path.join(
    "src",
    "data",
    "dataset.csv"
)

HISTORY_PATH = os.path.join(
    "src",
    "data",
    "spotify_history.csv"
)

FIGURES_PATH = os.path.join(
    "reports",
    "figures"
)

# Create figures folder if it does not exist
os.makedirs(FIGURES_PATH, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

tracks_df = pd.read_csv(TRACKS_PATH)
history_df = pd.read_csv(HISTORY_PATH)

print("Spotify Tracks Dataset loaded successfully.")
print("Spotify Streaming History Dataset loaded successfully.")


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SPOTIFY TRACKS DATASET - BASIC INFORMATION")
print("=" * 70)

print("Rows    :", tracks_df.shape[0])
print("Columns :", tracks_df.shape[1])

print("\nMissing Values:")
print(tracks_df.isnull().sum())

print("\nDuplicate Records:")
print(tracks_df.duplicated().sum())


print("\n" + "=" * 70)
print("SPOTIFY STREAMING HISTORY DATASET - BASIC INFORMATION")
print("=" * 70)

print("Rows    :", history_df.shape[0])
print("Columns :", history_df.shape[1])

print("\nMissing Values:")
print(history_df.isnull().sum())

print("\nDuplicate Records:")
print(history_df.duplicated().sum())


# ============================================================
# 1. POPULARITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    tracks_df["popularity"],
    bins=20,
    kde=True
)

plt.title("Distribution of Spotify Track Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Tracks")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "01_popularity_distribution.png"),
    dpi=300
)

plt.close()


# ============================================================
# 2. TOP 10 MUSIC GENRES
# ============================================================

genre_counts = tracks_df["track_genre"].value_counts().head(10)

plt.figure(figsize=(10, 6))

genre_counts.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Track Genres")
plt.xlabel("Number of Tracks")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "02_top_10_genres.png"),
    dpi=300
)

plt.close()


# ============================================================
# 3. POPULARITY BY EXPLICIT CONTENT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=tracks_df,
    x="explicit",
    y="popularity"
)

plt.title("Popularity by Explicit Content")
plt.xlabel("Explicit")
plt.ylabel("Popularity")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "03_popularity_by_explicit.png"),
    dpi=300
)

plt.close()


# ============================================================
# 4. DANCEABILITY VS POPULARITY
# ============================================================

plt.figure(figsize=(10, 6))

sample_tracks = tracks_df.sample(
    min(10000, len(tracks_df)),
    random_state=42
)

sns.scatterplot(
    data=sample_tracks,
    x="danceability",
    y="popularity",
    alpha=0.4
)

plt.title("Danceability vs Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "04_danceability_vs_popularity.png"),
    dpi=300
)

plt.close()


# ============================================================
# 5. ENERGY VS POPULARITY
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_tracks,
    x="energy",
    y="popularity",
    alpha=0.4
)

plt.title("Energy vs Popularity")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "05_energy_vs_popularity.png"),
    dpi=300
)

plt.close()


# ============================================================
# 6. AUDIO FEATURES CORRELATION HEATMAP
# ============================================================

audio_features = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

correlation_matrix = tracks_df[audio_features].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Spotify Audio Features")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "06_correlation_heatmap.png"),
    dpi=300
)

plt.close()


# ============================================================
# 7. TOP 10 ARTISTS BY NUMBER OF TRACKS
# ============================================================

artist_counts = tracks_df["artists"].value_counts().head(10)

plt.figure(figsize=(10, 6))

artist_counts.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Artists by Number of Tracks")
plt.xlabel("Number of Tracks")
plt.ylabel("Artist")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "07_top_artists.png"),
    dpi=300
)

plt.close()


# ============================================================
# 8. AUDIO FEATURE DISTRIBUTIONS
# ============================================================

features_to_plot = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]

for feature in features_to_plot:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        tracks_df[feature],
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {feature.capitalize()}")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Number of Tracks")
    plt.tight_layout()

    filename = f"08_distribution_{feature}.png"

    plt.savefig(
        os.path.join(FIGURES_PATH, filename),
        dpi=300
    )

    plt.close()


# ============================================================
# STREAMING HISTORY EDA
# ============================================================

print("\n" + "=" * 70)
print("STREAMING HISTORY ANALYSIS")
print("=" * 70)


# ============================================================
# 9. SKIPPED TRACK DISTRIBUTION
# ============================================================

if "skipped" in history_df.columns:

    skipped_counts = history_df["skipped"].value_counts()

    plt.figure(figsize=(8, 6))

    skipped_counts.plot(
        kind="bar"
    )

    plt.title("Skipped vs Non-Skipped Tracks")
    plt.xlabel("Skipped")
    plt.ylabel("Number of Streams")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_PATH, "09_skipped_distribution.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# 10. SHUFFLE DISTRIBUTION
# ============================================================

if "shuffle" in history_df.columns:

    shuffle_counts = history_df["shuffle"].value_counts()

    plt.figure(figsize=(8, 6))

    shuffle_counts.plot(
        kind="bar"
    )

    plt.title("Shuffle Mode Usage")
    plt.xlabel("Shuffle")
    plt.ylabel("Number of Streams")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_PATH, "10_shuffle_distribution.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# 11. TOP 10 ARTISTS IN STREAMING HISTORY
# ============================================================

if "artist_name" in history_df.columns:

    history_artists = (
        history_df["artist_name"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    history_artists.sort_values().plot(
        kind="barh"
    )

    plt.title("Top 10 Most Streamed Artists")
    plt.xlabel("Number of Streams")
    plt.ylabel("Artist")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_PATH, "11_top_streamed_artists.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# 12. TOP 10 MOST PLAYED TRACKS
# ============================================================

if "track_name" in history_df.columns:

    top_tracks = (
        history_df["track_name"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    top_tracks.sort_values().plot(
        kind="barh"
    )

    plt.title("Top 10 Most Played Tracks")
    plt.xlabel("Number of Streams")
    plt.ylabel("Track")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_PATH, "12_top_played_tracks.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# 13. TOTAL LISTENING TIME
# ============================================================

if "ms_played" in history_df.columns:

    total_minutes = (
        history_df["ms_played"].sum() / 1000 / 60
    )

    total_hours = total_minutes / 60

    print("\nTotal Listening Time:")
    print(f"{total_minutes:.2f} minutes")
    print(f"{total_hours:.2f} hours")


# ============================================================
# 14. LISTENING TIME DISTRIBUTION
# ============================================================

if "ms_played" in history_df.columns:

    listening_minutes = (
        history_df["ms_played"] / 1000 / 60
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        listening_minutes,
        bins=30,
        kde=True
    )

    plt.title("Distribution of Listening Time per Track")
    plt.xlabel("Minutes Played")
    plt.ylabel("Number of Streams")

    plt.xlim(
        0,
        listening_minutes.quantile(0.99)
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURES_PATH, "13_listening_time_distribution.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGraphs have been saved to:")
print(FIGURES_PATH)

print("\nGenerated visualization files:")

for file in sorted(os.listdir(FIGURES_PATH)):

    if file.endswith(".png"):
        print("-", file)

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)
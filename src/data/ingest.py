import os
import pandas as pd


# ============================================================
# FUNCTION TO LOAD AND VALIDATE SPOTIFY TRACKS DATASET
# ============================================================

def load_tracks_data(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Tracks dataset not found at: {file_path}"
        )

    print("\nLoading Spotify Tracks Dataset...")
    df = pd.read_csv(file_path)

    required_columns = [
        "track_id",
        "artists",
        "album_name",
        "track_name",
        "popularity",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
        "track_genre"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Tracks Dataset - Missing columns: {missing_columns}"
        )

    print("Tracks Dataset validation successful.")

    return df


# ============================================================
# FUNCTION TO LOAD AND VALIDATE STREAMING HISTORY DATASET
# ============================================================

def load_history_data(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Streaming history dataset not found at: {file_path}"
        )

    print("\nLoading Spotify Streaming History Dataset...")
    df = pd.read_csv(file_path)

    required_columns = [
        "spotify_track_uri",
        "ts",
        "platform",
        "ms_played",
        "track_name",
        "artist_name",
        "album_name",
        "reason_start",
        "reason_end",
        "shuffle",
        "skipped"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Streaming History Dataset - Missing columns: {missing_columns}"
        )

    print("Streaming History Dataset validation successful.")

    return df


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

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

    try:

        # Load first dataset
        tracks_df = load_tracks_data(TRACKS_PATH)

        print("\n" + "=" * 60)
        print("--- SPOTIFY TRACKS DATASET ---")
        print("=" * 60)

        print(f"Rows    : {tracks_df.shape[0]}")
        print(f"Columns : {tracks_df.shape[1]}")

        print("\nColumn Names:")
        print(tracks_df.columns.tolist())

        print("\nFirst 5 Records:")
        print(tracks_df.head())

        # Load second dataset
        history_df = load_history_data(HISTORY_PATH)

        print("\n" + "=" * 60)
        print("--- SPOTIFY STREAMING HISTORY DATASET ---")
        print("=" * 60)

        print(f"Rows    : {history_df.shape[0]}")
        print(f"Columns : {history_df.shape[1]}")

        print("\nColumn Names:")
        print(history_df.columns.tolist())

        print("\nFirst 5 Records:")
        print(history_df.head())

        # Final message
        print("\n" + "=" * 60)
        print("DATA INGESTION AND VALIDATION COMPLETED SUCCESSFULLY.")
        print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print("DATA INGESTION FAILED")
        print("=" * 60)

        print(f"Error: {e}")
"""
Statcast pitch data fetcher with progressive filtering.

Usage (interactive):
    python statcast_fetcher.py

Usage (as module):
    from statcast_fetcher import StatcastFetcher
    fetcher = StatcastFetcher()
    df = fetcher.search_pitcher("Ohtani", 2024)
    games = fetcher.list_game_dates(df)
    df_game = fetcher.filter_by_date(df, "2024-07-04")
    pitch = fetcher.select_pitch(df_game, index=3)
"""

import pandas as pd
from pybaseball import statcast, playerid_lookup, statcast_pitcher


class StatcastFetcher:
    """Progressively filter Statcast data to identify a single pitch."""

    # Columns relevant to trajectory simulation
    SIM_COLUMNS = [
        # Core identification
        "pitch_type", "pitch_name", "game_date", "pitcher", "batter",
        "player_name", "game_pk", "game_type",
        # Release
        "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
        "release_spin_rate", "spin_axis", "release_extension",
        "arm_angle",
        # Velocity & acceleration at y=50ft
        "vx0", "vy0", "vz0",
        "ax", "ay", "az",
        # Movement (pfx = spin-induced, feet)
        "pfx_x", "pfx_z",
        # Break (API break = total including gravity)
        "api_break_x_arm", "api_break_x_batter_in", "api_break_z_with_gravity",
        # Plate crossing (feet)
        "plate_x", "plate_z",
        # Speed
        "effective_speed",
        # Pitch event / result
        "description", "events", "des", "type",
        "zone",
        "inning", "inning_topbot", "at_bat_number", "pitch_number",
        "balls", "strikes", "outs_when_up",
        "p_throws", "stand",
        "sz_top", "sz_bot",
        # Batted ball
        "launch_speed", "launch_angle", "hit_distance_sc",
        "bb_type", "hc_x", "hc_y",
        "bat_speed", "swing_length",
        # Expected stats (per-pitch)
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "estimated_slg_using_speedangle",
        "babip_value", "iso_value",
        "woba_value", "woba_denom",
        # Score context
        "home_team", "away_team",
        "on_1b", "on_2b", "on_3b",
    ]

    def search_pitcher(self, last_name: str, year: int,
                       first_name: str = None) -> pd.DataFrame:
        """
        Step 1: Search by pitcher name and year.

        Parameters
        ----------
        last_name : str
            Pitcher last name (e.g. "Ohtani")
        year : int
            Season year (e.g. 2024)
        first_name : str, optional
            First name to disambiguate

        Returns
        -------
        pd.DataFrame
            All pitches by that pitcher in the given year
        """
        print(f"Looking up player: {last_name} {first_name or ''}...")
        lookup = playerid_lookup(last_name, first_name) if first_name else playerid_lookup(last_name)

        if lookup.empty:
            print("Player not found.")
            return pd.DataFrame()

        print("\nMatched players:")
        for i, row in lookup.iterrows():
            print(f"  [{i}] {row['name_first']} {row['name_last']} "
                  f"(MLBAM ID: {row['key_mlbam']}, years: {row.get('mlb_played_first', '?')}-{row.get('mlb_played_last', '?')})")

        if len(lookup) == 1:
            player_id = int(lookup.iloc[0]["key_mlbam"])
        else:
            idx = int(input(f"Select player index [0-{len(lookup)-1}]: "))
            player_id = int(lookup.iloc[idx]["key_mlbam"])

        print(f"\nFetching Statcast data for MLBAM ID {player_id}, year {year}...")
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        df = statcast_pitcher(start, end, player_id)

        if df.empty:
            print("No data found.")
            return df

        # Keep useful columns (drop missing ones gracefully)
        cols = [c for c in self.SIM_COLUMNS if c in df.columns]
        df = df[cols].copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["game_date", "at_bat_number", "pitch_number"]).reset_index(drop=True)

        print(f"Found {len(df)} pitches across {df['game_date'].nunique()} games.")
        return df

    def list_game_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Show available game dates with pitch counts.

        Returns summary DataFrame.
        """
        if df.empty:
            print("No data.")
            return pd.DataFrame()

        summary = (
            df.groupby("game_date")
            .agg(pitches=("pitch_type", "count"),
                 types=("pitch_type", lambda x: ", ".join(sorted(x.dropna().unique()))))
            .reset_index()
            .sort_values("game_date")
        )
        print("\nAvailable game dates:")
        for i, row in summary.iterrows():
            print(f"  {row['game_date'].strftime('%Y-%m-%d')}  "
                  f"{row['pitches']:>3} pitches  ({row['types']})")
        return summary

    def filter_by_date(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """
        Step 3: Filter to a specific game date.

        Parameters
        ----------
        date_str : str
            Date string (e.g. "2024-07-04")
        """
        date = pd.to_datetime(date_str)
        filtered = df[df["game_date"] == date].reset_index(drop=True)
        print(f"\n{len(filtered)} pitches on {date_str}")

        if not filtered.empty:
            print("\nPitch list:")
            for i, row in filtered.iterrows():
                speed = row.get("release_speed", "?")
                ptype = row.get("pitch_type", "?")
                spin = row.get("release_spin_rate", "?")
                inning = row.get("inning", "?")
                ab = row.get("at_bat_number", "?")
                pnum = row.get("pitch_number", "?")
                desc = row.get("description", "")
                print(f"  [{i:>3}] {ptype:>3}  {speed:>5} mph  "
                      f"spin={spin:>6}  inn={inning} AB={ab} P#{pnum}  {desc}")
        return filtered

    def select_pitch(self, df: pd.DataFrame, index: int) -> dict:
        """
        Step 4: Select one pitch and return parameters for the simulator.

        Returns
        -------
        dict
            Pitch data with Statcast raw values
        """
        if index < 0 or index >= len(df):
            print(f"Index {index} out of range [0, {len(df)-1}].")
            return {}

        row = df.iloc[index]
        pitch_data = row.to_dict()

        print(f"\nSelected pitch #{index}:")
        for k, v in pitch_data.items():
            print(f"  {k}: {v}")

        return pitch_data


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def interactive():
    fetcher = StatcastFetcher()

    # Step 1: Pitcher + Year
    last_name = input("Pitcher last name: ").strip()
    first_name = input("First name (Enter to skip): ").strip() or None
    year = int(input("Year (e.g. 2024): ").strip())

    df = fetcher.search_pitcher(last_name, year, first_name)
    if df.empty:
        return

    # Step 2: Game dates
    fetcher.list_game_dates(df)
    date_str = input("\nSelect date (YYYY-MM-DD): ").strip()

    # Step 3: Filter by date
    df_game = fetcher.filter_by_date(df, date_str)
    if df_game.empty:
        return

    # Step 4: Select pitch
    idx = int(input("\nSelect pitch index: ").strip())
    pitch = fetcher.select_pitch(df_game, idx)

    print("\n--- Ready to pass to simulator ---")
    print(f"  release_speed: {pitch.get('release_speed')} mph")
    print(f"  release_pos:   ({pitch.get('release_pos_x')}, {pitch.get('release_pos_y')}, {pitch.get('release_pos_z')})")
    print(f"  velocity:      ({pitch.get('vx0')}, {pitch.get('vy0')}, {pitch.get('vz0')})")
    print(f"  acceleration:  ({pitch.get('ax')}, {pitch.get('ay')}, {pitch.get('az')})")
    print(f"  spin_rate:     {pitch.get('release_spin_rate')} rpm")
    print(f"  spin_axis:     {pitch.get('spin_axis')} deg")


if __name__ == "__main__":
    interactive()

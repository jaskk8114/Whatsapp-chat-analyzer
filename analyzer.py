"""
WhatsApp Chat Analyzer — a data curation pipeline.

Takes a raw WhatsApp chat export (.txt) — which is messy, unstructured text —
and turns it into a clean, analysis-ready dataset, then produces insights.

Pipeline stages:
  1. LOAD      raw text lines from the export file
  2. PARSE     regex-match message lines, merge multi-line messages
  3. CURATE    separate system messages / media / deleted messages from real text
  4. VALIDATE  produce a data quality report (raw lines in -> clean rows out)
  5. ANALYZE   per-user stats, activity patterns, reply times, emoji usage
  6. VISUALIZE save charts to the output folder

Usage:
    python analyzer.py sample_chat.txt
    python analyzer.py my_real_chat.txt
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files, no display window needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_raw_lines(filepath: str) -> list[str]:
    """Read the export file and return raw lines (curation starts with raw input)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().splitlines()


# ---------------------------------------------------------------------------
# 2. PARSE
# ---------------------------------------------------------------------------

# WhatsApp export formats differ by phone/region. We try each known pattern.
# Groups captured: date, time, rest-of-line
MESSAGE_PATTERNS = [
    # Android (India default): 12/01/26, 9:05 pm - Name: message
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[apAP][mM]) - (.*)$"),
    # Android 24-hour: 12/01/2026, 21:05 - Name: message
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*)$"),
    # iOS: [12/01/26, 9:05:33 PM] Name: message
    re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}:\d{2}\s?[apAP][mM])\] (.*)$"),
]


def parse_chat(raw_lines: list[str]) -> tuple[list[dict], dict]:
    """
    Convert raw lines into structured records.

    Handles the two classic messes in WhatsApp exports:
      - multi-line messages (continuation lines have NO timestamp -> append
        them to the previous message)
      - lines that match a timestamp but have no "Name:" part -> system events
        (group created, subject changed, encryption notice, etc.)

    Returns (records, parse_stats).
    """
    records = []
    stats = {"raw_lines": len(raw_lines), "continuation_lines_merged": 0,
             "unparseable_lines": 0}

    for line in raw_lines:
        if not line.strip():
            continue

        matched = None
        for pattern in MESSAGE_PATTERNS:
            m = pattern.match(line)
            if m:
                matched = m
                break

        if matched:
            date_str, time_str, rest = matched.groups()
            # "Name: message" -> user message; otherwise it's a system event
            if ": " in rest:
                sender, message = rest.split(": ", 1)
                records.append({"date": date_str, "time": time_str,
                                "sender": sender, "message": message,
                                "type": "user"})
            else:
                records.append({"date": date_str, "time": time_str,
                                "sender": None, "message": rest,
                                "type": "system"})
        else:
            # No timestamp -> continuation of the previous message
            if records and records[-1]["type"] == "user":
                records[-1]["message"] += " " + line.strip()
                stats["continuation_lines_merged"] += 1
            else:
                stats["unparseable_lines"] += 1

    return records, stats


# ---------------------------------------------------------------------------
# 3. CURATE
# ---------------------------------------------------------------------------

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2764\ufe0f]+"
)


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Structured records -> typed, flagged, analysis-ready DataFrame."""
    df = pd.DataFrame(records)

    # Parse datetime (dayfirst=True because Indian exports are DD/MM/YY)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    dayfirst=True, format="mixed")

    # Flag noise categories instead of silently deleting them — a curation
    # principle: keep the raw signal, label it, filter at analysis time.
    df["is_media"] = df["message"].str.contains("<Media omitted>", na=False)
    df["is_deleted"] = df["message"].str.contains("This message was deleted",
                                                  na=False)
    df["is_system"] = df["type"] == "system"

    # Derived fields for analysis
    df["hour"] = df["datetime"].dt.hour
    df["day_name"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    df["word_count"] = df["message"].str.split().str.len()
    df["emojis"] = df["message"].apply(lambda m: EMOJI_RE.findall(str(m)))

    return df


def clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """The 'clean' analytical view: real text messages from real people."""
    return df[(df["type"] == "user") & ~df["is_media"] & ~df["is_deleted"]]


# ---------------------------------------------------------------------------
# 4. VALIDATE — data quality report
# ---------------------------------------------------------------------------

def quality_report(df: pd.DataFrame, parse_stats: dict) -> str:
    clean = clean_view(df)
    lines = [
        "=" * 52,
        "DATA QUALITY REPORT — raw export -> clean dataset",
        "=" * 52,
        f"Raw lines in export file      : {parse_stats['raw_lines']}",
        f"Multi-line continuations merged: {parse_stats['continuation_lines_merged']}",
        f"Unparseable lines dropped     : {parse_stats['unparseable_lines']}",
        f"Total parsed records          : {len(df)}",
        f"  - system events filtered    : {int(df['is_system'].sum())}",
        f"  - media placeholders        : {int(df['is_media'].sum())}",
        f"  - deleted messages          : {int(df['is_deleted'].sum())}",
        f"CLEAN text messages (final)   : {len(clean)}",
        f"Date range                    : {df['datetime'].min()} -> {df['datetime'].max()}",
        f"Participants                  : {clean['sender'].nunique()}",
        "=" * 52,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. ANALYZE
# ---------------------------------------------------------------------------

def compute_reply_times(clean: pd.DataFrame,
                        session_gap_minutes: int = 120) -> pd.Series:
    """
    Median reply time per user (minutes).

    A 'reply' = a message whose previous message is from a DIFFERENT sender
    and arrived within `session_gap_minutes` (gaps longer than that are a new
    conversation, not a slow reply — this threshold is a curation decision).
    """
    c = clean.sort_values("datetime")
    gap = c["datetime"].diff().dt.total_seconds() / 60
    different_sender = c["sender"] != c["sender"].shift()
    replies = c[different_sender & (gap <= session_gap_minutes)].copy()
    replies["reply_minutes"] = gap[replies.index]
    return replies.groupby("sender")["reply_minutes"].median().round(1)


def analyze(df: pd.DataFrame) -> dict:
    clean = clean_view(df)
    top_emojis = Counter(e for lst in clean["emojis"] for e in lst).most_common(5)
    return {
        "messages_per_user": clean["sender"].value_counts(),
        "words_per_user": clean.groupby("sender")["word_count"].sum().sort_values(ascending=False),
        "media_per_user": df[df["is_media"]]["sender"].value_counts(),
        "busiest_hours": clean["hour"].value_counts().sort_index(),
        "busiest_days": clean["day_name"].value_counts(),
        "monthly_trend": clean.groupby("month").size(),
        "median_reply_minutes": compute_reply_times(clean),
        "top_emojis": top_emojis,
    }


# ---------------------------------------------------------------------------
# 6. VISUALIZE
# ---------------------------------------------------------------------------

def save_charts(results: dict, outdir: Path) -> list[Path]:
    outdir.mkdir(exist_ok=True)
    saved = []

    fig, ax = plt.subplots(figsize=(7, 4))
    results["messages_per_user"].plot(kind="barh", ax=ax, color="#2a9d8f")
    ax.set_title("Messages per participant")
    ax.set_xlabel("Clean text messages")
    fig.tight_layout()
    p = outdir / "messages_per_user.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    results["busiest_hours"].plot(kind="bar", ax=ax, color="#e76f51")
    ax.set_title("Activity by hour of day")
    ax.set_xlabel("Hour (0-23)"); ax.set_ylabel("Messages")
    fig.tight_layout()
    p = outdir / "activity_by_hour.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    results["monthly_trend"].plot(kind="line", marker="o", ax=ax, color="#264653")
    ax.set_title("Messages per month")
    ax.set_ylabel("Messages")
    fig.tight_layout()
    p = outdir / "monthly_trend.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    chat_file = sys.argv[1] if len(sys.argv) > 1 else "sample_chat.txt"
    outdir = Path("output")

    raw = load_raw_lines(chat_file)
    records, parse_stats = parse_chat(raw)
    df = build_dataframe(records)

    report = quality_report(df, parse_stats)
    print(report)
    outdir.mkdir(exist_ok=True)
    (outdir / "quality_report.txt").write_text(report, encoding="utf-8")

    results = analyze(df)
    print("\nMessages per user:\n", results["messages_per_user"].to_string())
    print("\nMedian reply time (min):\n", results["median_reply_minutes"].to_string())
    print("\nTop emojis:", results["top_emojis"])

    clean_view(df).drop(columns=["emojis"]).to_csv(outdir / "clean_messages.csv",
                                                   index=False)
    charts = save_charts(results, outdir)
    print(f"\nSaved clean dataset + quality report + {len(charts)} charts to ./{outdir}/")


if __name__ == "__main__":
    main()

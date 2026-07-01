# WhatsApp Chat Analyzer — Data Curation Pipeline

Turns a raw WhatsApp chat export (unstructured, messy text) into a clean,
analysis-ready dataset, with a data quality report and usage insights.

**One-line pitch:** *"I built a pipeline that parses raw WhatsApp exports —
multi-line messages, system events, media placeholders and all — into a
structured dataset, and generates a quality report plus activity analytics."*

## Run it

```bash
pip install pandas matplotlib
python analyzer.py sample_chat.txt      # demo on included sample
python analyzer.py my_chat.txt          # your real chat
```

To export a real chat: open the chat in WhatsApp → ⋮ (three dots) → More →
**Export chat** → **Without media** → send the .txt to yourself.

## Why this is a data curation project

Raw WhatsApp exports are genuinely dirty data:

| Mess in the raw file | How the pipeline handles it |
|---|---|
| Multi-line messages (no timestamp on continuation lines) | Merged into the parent message during parsing |
| System events (group created, subject changed, encryption notice) | Detected (no `Name:` part) and flagged, not deleted |
| `<Media omitted>` placeholders | Flagged `is_media`, excluded from text analytics, counted separately |
| "This message was deleted" | Flagged `is_deleted` |
| Multiple export formats (Android 12h/24h, iOS) | Regex pattern list tries each known format |
| Dates in DD/MM/YY (Indian locale) | Parsed with `dayfirst=True` |

**Key curation principle used:** noise is *flagged and filtered at analysis
time*, never silently deleted — the raw signal is preserved.

## Pipeline (analyzer.py)

`load_raw_lines → parse_chat → build_dataframe → quality_report → analyze → save_charts`

## Outputs (in ./output/)

- `quality_report.txt` — raw lines in → clean rows out, with every drop accounted for
- `clean_messages.csv` — the curated dataset
- 3 charts: messages per user, activity by hour, monthly trend

## Analytics included

Messages & words per user, media senders, busiest hours/days, monthly trend,
top emojis, and **median reply time per user** (a reply = next message from a
different sender within 120 min — gaps longer than that are treated as a new
conversation, not a slow reply).

## Decisions I made (and can defend)

1. **Flag, don't delete** — media/system/deleted rows stay in the DataFrame
   with boolean flags; the clean view filters them. Auditable and reversible.
2. **120-minute session gap** for reply-time analysis — without it, overnight
   silences would wreck the median.
3. **Regex pattern list** instead of one giant regex — each export format gets
   its own readable pattern; easy to add new formats.
4. **Median (not mean) reply time** — reply times are heavily right-skewed;
   one 3-hour gap would distort a mean.

## Possible improvements

Sentiment analysis per user, a Streamlit dashboard, conversation-starter
detection, support for exported group-call logs.

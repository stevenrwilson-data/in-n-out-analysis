# In-N-Out: cleaning, observation exposure, and distribution analysis.
# Raw JSON is read-only; applied corrections and reliability flags are exported.
# Dependencies: python -m pip install numpy pandas matplotlib scipy
# Run: python burgers_clean.py

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats

# SECTION INDEX — search for a numbered heading below
#   0            OUTPUT DIRECTORIES AND GLOBAL PLOT SETTINGS
#   0.1          Create image and CSV output directories
#   0.2          GLOBAL PORTFOLIO PLOT SETTINGS
#   1            LOAD RAW COMBINED JSON
#   1.1          This is the combined source file containing all observation
#   2            FLATTEN THE MIXED HISTORICAL SCHEMAS
#   2.1          Earlier text-converted format
#   2.2          Later structured JSON export format
#   2.3          Preserve source metadata on every event
#   3            STANDARDIZE TIMESTAMPS
#   4            BASIC VALIDATION
#   5            EXCLUDE THE KNOWN DEVELOPMENT / TEST EXPORT
#   6            APPLY KNOWN COUNTER-ORDER CORRECTIONS
#   6.1          Work on a dedicated copy of counter-order events.
#   6.2          Helper: normalize the "numbers" field
#   6.3          Correct sequence 1964: replace order 12 with order 11
#   6.4          Exclude sequence 1971: compensating order 11 entry
#   6.3.1        Correct the independently verified September 3 mistyped order
#   6.5          Correct sequence 1810: replace order 35 with order 55
#   6.6          Explicit "not repeat" corrections
#   7            RECALCULATE COUNTER REPEATS USING THE 30-MINUTE RULE
#   7.1          Manually excluded compensating entries contribute zero.
#   7.2          If no usable order number exists, count the event as one.
#   7.3          Some events have explicit notes saying the logger's
#   7.4          Update the most recent observed time regardless of
#   8            COUNTER THROUGHPUT SUMMARY
#   9            DRIVE-THRU ORDER CORRECTIONS
#   9.1          Prepare the analytical drive-thru copy
#   9.2          Exclude bookkeeping completions: cars 34 and 35
#   9.3          Exclude the confirmed accidental duplicate: car 1318
#   9.3.1        Keep zero-exposure cars in the audit, outside rate/gap samples
#   9.4          Select observed completions for downstream analysis
#   9.5          Print the exclusion audit
#   10           DRIVE-THRU THROUGHPUT SUMMARY
#   10.1         Shared note-based interval reliability rules
#   11           INTER-TRANSACTION TIMES
#   11.1         DRIVE-THRU
#   11.2         Keep only gaps belonging to the same continuous period of
#   11.3         COUNTER
#   12           FIRST THROUGHPUT RESULTS
#   13           BUILD TRUSTWORTHY OBSERVATION WINDOWS
#   13.1         Timer identity
#   13.2         Remove unusable negative-duration timer-end records
#   13.3         Remove exact duplicate timer records
#   14           BUILD CONTINUOUS EVENT SEGMENTS
#   14.1         First row always starts a new segment.
#   14.2         Drive-thru segments
#   14.3         Counter segments
#   15           MATCH SEGMENTS TO NEARBY TIMER BOUNDARIES
#   15.1         Candidate timer starts before the first event
#   15.2         Only trust it if it is within 8 minutes.
#   15.3         Candidate timer ends after the last event
#   15.4         Only trust it if it is within 10 minutes.
#   15.4.1       Remove documented observer breaks from measured exposure
#   15.5         Combine for easier inspection.
#   16           CHANNEL-HOUR SUMMARY
#   17           PREPARE DATA FOR OPERATING-DAY PLOTS
#   17.1         Drive-thru operating-day data
#   17.1.1       Keep only observations between 10 AM and 2 AM.
#   17.1.2       Shift midnight through 2 AM to the end of the operating day.
#   17.1.3       Final hard cutoff at 2:00 AM.
#   17.2         Counter operating-day data
#   17.3         Shared operating-day axis labels
#   18           TRANSACTION-GAP PLOTS
#   18.1         DRIVE-THRU GAP VS TIME OF DAY
#   18.1.1       Save drive_gap_by_time.png to images/
#   18.2         COUNTER GAP VS TIME OF DAY
#   18.2.1       Save counter_gap_by_time.png to images/
#   18.3         SHARED HOURLY GAP SUMMARY AND PLOT
#   18.3.1       Drive-thru hourly completion gaps
#   18.4         Counter hourly announcement gaps
#   18.5         Save sample counts and quartiles with their hourly summaries
#   19           COMPLETED ORDERS BY OPERATING HOUR
#   19.1         DRIVE-THRU COMPLETED ORDERS BY HOUR
#   19.1.1       Keep only the operating-day window.
#   19.1.2       Shift after-midnight observations to the end of the same
#   19.2         COUNTER COMPLETED ORDERS BY HOUR
#   19.2.1       Sum corrected_count, not event rows.
#   19.3         PRINT HOURLY COUNTS
#   20           HOURLY OBSERVATION EXPOSURE AND THROUGHPUT
#   20.1         SPLIT OBSERVATION WINDOWS INTO CLOCK-HOUR EXPOSURE
#   20.1.1       Split windows at hour boundaries and keep operating-day bins
#   20.2         TOTAL OBSERVED MINUTES BY HOUR AND MODE
#   20.2.1       Guarantee both columns exist.
#   20.3         COMPLETED ORDERS BY THE SAME HOUR BINS
#   20.4         CALCULATE EXPOSURE-ADJUSTED HOURLY RATES
#   20.4.1       Hours with no observation time should remain missing,
#   20.4.2       Add independent-day coverage and export the hourly table
#   20.5         PRINT HOURLY EXPOSURE AND THROUGHPUT
#   20.6         PLOT EXPOSURE-ADJUSTED THROUGHPUT
#   20.6.1       Save throughput_by_hour.png to images/
#   20.7         PLOT OBSERVATION MINUTES BY HOUR
#   20.7.1       Save observation_minutes_by_hour.png to images/
#   21           SIMULTANEOUS DRIVE-THRU VS COUNTER THROUGHPUT
#   21.1         MERGE OVERLAPPING WINDOWS WITHIN EACH CHANNEL
#   21.1.1       Merge overlapping or touching windows.
#   21.2         FIND EXACT CLOCK-TIME OVERLAPS
#   21.2.1       Positive duration means both channels were actively
#   21.3         COUNT COMPLETED ORDERS INSIDE EACH OVERLAP WINDOW
#   21.3.1       Drive-thru completed orders
#   21.3.2       Counter completed orders
#   21.3.3       Exposure-adjusted rates
#   21.4         POOLED OVERLAP-ONLY RESULTS
#   21.5         INDIVIDUAL OVERLAP WINDOWS
#   21.6         PAIRED THROUGHPUT BY SIMULTANEOUS OBSERVATION START
#   22           COMPLETION-GAP DISTRIBUTION CHECK
#   22.1         Prepare both channels' valid completion gaps
#   22.2         Calculate descriptive distribution statistics
#   22.3         Print observed statistics and exponential references
#   22.4         Plot observed and reference survival curves
#   22.4.1       Calculate empirical survival with ties handled
#   22.4.2       Calculate the same-mean exponential reference
#   22.4.3       Format the channel panel
#   22.5         Save the completion-gap distribution comparison
#   23           MOVEMENT PAUSES AND COMPLETION-GAP DISTRIBUTIONS
#   23.1         Assign markers to existing drive-thru observation blocks
#   23.1.1       Remove retracted movement-marker entries
#   23.2         Pair markers within each block and flag incomplete episodes
#   23.3         Match paired pauses to consecutive completed cars
#   23.3.1       Exclude gaps explicitly identified as unreliable
#   23.4         Flag explicitly late up taps and print diagnostics
#   23.4.1       Flag candidate matches around explicit late-up notes
#   23.5         Compare survival curves and exponential Q-Q distributions
#   23.6         Examine pause duration versus time after advancing
#   24           EXPONENTIAL VS GAMMA DISTRIBUTION FITS
#   24.1         Prepare the two distribution samples
#   24.2         Fit positive-parameter models by maximum likelihood
#   24.2.1       Use derivative-free optimization to avoid numerical
#   24.3         Fit and draw survival and Q-Q comparisons for each sample
#   24.3.1       Condition fitted curves on the completion cutoff
#   24.4         Print within-sample model comparisons
#   24.5         Save the distribution-model comparison
#   25           CONSECUTIVE COMPLETION GAPS
#   25.1         Preserve event order and split at unreliable intervals
#   25.2         Build actual neighboring pairs at lags 1 through 10
#   25.3         Resample operating days using sufficient statistics
#   25.4         Calculate pooled and within-block correlations
#   25.5         Plot adjacent gaps and correlations across lags
#   25.6         Save the numeric diagnostic table
#   26           DO FAST COMPLETIONS PRECEDE ADVANCE PAUSES?
#   26.1         Identify markers in each actual completion interval
#   26.2         Use the preceding three gaps, excluding earlier pauses
#   26.3         Compare speed relative to the same observation block
#   26.4         Divide eligible sequences into four speed groups
#   26.5         Print results and save the event-level audit
#   26.6         Plot preceding-gap distributions and pause fractions
#   27           COUNTER ANNOUNCEMENT BATCHES AND SHORT BURSTS
#   27.1         Prepare corrected positive-count announcement events
#   27.2         Group announcements within a bounded total time span
#   27.3         Check sensitivity to 5-, 10-, and 15-second spans
#   27.4         Print distributions and save audit tables
#   27.5         Plot announcement sizes and grouped burst sizes
#   27.6         Compare preceding quiet-gap distributions by burst size
#   28           REVIEW LINE-LENGTH AND REGISTER OBSERVATIONS
#   28.1         Collect candidate notes without changing their wording
#   28.2         Suggest numbers and flag context that needs judgment
#   28.3         Locate notes within counter observation windows
#   28.4         Export the review sheet and print coverage
#   29           COUNTER BURSTS AFTER LINE-LENGTH OBSERVATIONS
#   29.1         Include the reviewed numeric and explicit empty-line notes
#   29.2         Validate notes and restrict the main comparison
#   29.3         Match each burst to one recent eligible snapshot per lag band
#   29.4         Summarize burst distributions with whole-day uncertainty
#   29.5         Print results and save reproducible audit tables
#   29.6         Plot complete burst-size distributions by line condition
#   30           COUNTER OUTPUT AFTER LINE SNAPSHOTS AND RECENT OUTPUT
#   30.1         Measure observed output before each eligible line snapshot
#   30.2         Define joint line-size and recent-output conditions
#   30.3         Partition overlapping follow-up intervals without double counting
#   30.4         Calculate rates and whole-day bootstrap intervals
#   30.5         Print coverage, results and limitations; save audits
#   30.6         Plot output rates and changes across delay bands
#   31           COUNTER ANNOUNCEMENT SPACING VS RANDOM TIMING
#   31.1         Collect actual adjacent gaps inside observed counter blocks
#   31.2         Simulate timing while preserving counts and boundaries
#   31.3         Print matched-denominator comparisons and save results
#   31.4         Plot observed fractions against the random reference
#   32           FINAL AUDIT AND RECONCILIATION
#   32.1         Export applied event corrections and unresolved decisions
#   32.1.1       Save unreliable intervals separately from completion exclusions
#   32.2         Check positive exposure and event coverage

# ============================================================
# 0. OUTPUT DIRECTORIES AND GLOBAL PLOT SETTINGS
# ============================================================

# ------------------------------------------------------------
# 0.1 Create image and CSV output directories
# ------------------------------------------------------------

IMAGE_DIR = Path(__file__).resolve().parent / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = Path(__file__).resolve().parent / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 0.2 GLOBAL PORTFOLIO PLOT SETTINGS
# ============================================================

brand_colors = [
    "#08415C",  # blue
    "#8E443D",  # brick
    "#511730",  # plum
    "#E0D68A",  # sand
    "#F7A399",  # salmon
    "#041F2C",  # field
]

plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=brand_colors),
    "font.family": "Montserrat",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.facecolor": "#F4F0E6",
    "axes.facecolor": "#F4F0E6",
    "axes.labelcolor": "#041F2C",
    "xtick.color": "#041F2C",
    "ytick.color": "#041F2C",
    "text.color": "#041F2C",
    "grid.color": "#F7A399",
    "grid.linewidth": 0.5,
})

# ============================================================
# 1. LOAD RAW COMBINED JSON
# ============================================================

# 1.1 This is the combined source file containing all observation
# sessions from August 17 through September 6, 2026.
#
# IMPORTANT:
# The raw file is never overwritten. Everything produced later
# in this script will be saved under a new filename.

RAW_FILE = Path(__file__).resolve().parent / "innout_log_2026.json"

raw_df = pd.read_json(RAW_FILE)

# 2. FLATTEN THE MIXED HISTORICAL SCHEMAS
# The combined file contains two slightly different historical
# storage formats because the Queue Log app evolved while data

flattened_events = []

for source_row, row in raw_df.iterrows():
    session = row["sessions"]

    # --------------------------------------------------------
    # 2.1 Earlier text-converted format
    # --------------------------------------------------------
    if "events" in session:
        events = session["events"]

    # --------------------------------------------------------
    # 2.2 Later structured JSON export format
    # --------------------------------------------------------
    elif "original" in session and "events" in session["original"]:
        events = session["original"]["events"]
    else:
        raise ValueError(
            f"Unknown session structure in source row {source_row}"
        )

    # --------------------------------------------------------
    # 2.3 Preserve source metadata on every event
    #
    # This makes it possible to trace any cleaned record back
    # to the original export that produced it.
    # --------------------------------------------------------
    for event in events:
        record = event.copy()
        record["source_row"] = source_row
        record["source_file"] = session.get("source_file")
        record["source_date"] = session.get("date")
        flattened_events.append(record)

events_df = pd.DataFrame(flattened_events)

# ============================================================
# 3. STANDARDIZE TIMESTAMPS
# ============================================================

events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], errors='coerce')

# 4. BASIC VALIDATION
# These are lightweight checks only.
# The giant working script contains the deeper diagnostics.

print("\n" + "=" * 50)
print("FLATTENED RAW EVENT DATA")
print("=" * 50)

print(f"Rows: {len(events_df):,}")
print(f"Columns: {len(events_df.columns)}")

print("\nEvent counts by kind:")
print(events_df["kind"].value_counts())

print("\nMissing timestamps:")
print(events_df["timestamp"].isna().sum())

print("\nDate range:")
print(events_df["timestamp"].min())
print(events_df["timestamp"].max())

# 5. EXCLUDE THE KNOWN DEVELOPMENT / TEST EXPORT
# source_row == 3 is the small August 19 development/test file.
# We DO NOT delete it from the raw data.

analysis_events = events_df.loc[events_df['source_row'] != 3].copy()

print("\n" + "=" * 50)
print("ANALYTICAL EVENT DATA")
print("=" * 50)

print(f"Rows after excluding test export: {len(analysis_events):,}")

print("\nEvent counts by kind:")
print(analysis_events["kind"].value_counts())

# 6. APPLY KNOWN COUNTER-ORDER CORRECTIONS
# Several contemporaneous notes recorded known entry mistakes.
# These corrections are applied explicitly here so the final

# 6.1 Work on a dedicated copy of counter-order events.
counter_orders = analysis_events.loc[analysis_events['kind'] == 'order'].copy()

# ------------------------------------------------------------
# 6.2 Helper: normalize the "numbers" field
#
# Most order events contain one or more order numbers.
# We convert them to strings consistently so repeat matching
# behaves predictably.
# ------------------------------------------------------------

def normalize_numbers(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    return [str(value)]

counter_orders['numbers_clean'] = counter_orders['numbers'].apply(normalize_numbers)

# ------------------------------------------------------------
# 6.3 Correct sequence 1964: replace order 12 with order 11
# sequence 1964 was logged as orders 10 and 12.
#
# Contemporary note says:
#     "12 was 11.. fat fingered it"
#
# Correct the event to 10 and 11.
# ------------------------------------------------------------

mask = (counter_orders['source_row'] == 4) & (counter_orders['seq'] == 1964)

counter_orders.loc[mask, "numbers_clean"] = pd.Series(
    [["10", "11"]] * mask.sum(),
    index=counter_orders.index[mask]
)

# ------------------------------------------------------------
# 6.4 Exclude sequence 1971: compensating order 11 entry
# sequence 1971 was a compensating entry of order 11 that was
# added only because the earlier event had been mistyped.
#
# Once sequence 1964 is corrected to include order 11, this
# compensating entry should not count as another transaction.
# ------------------------------------------------------------

# 6.3.1 Correct the independently verified September 3 mistyped order
mask = counter_orders['source_row'].eq(4) & counter_orders['seq'].eq(2320)
counter_orders.loc[mask, 'numbers_clean'] = pd.Series(
    [['11']] * mask.sum(), index=counter_orders.index[mask])

counter_orders["exclude_manual"] = False

mask = (counter_orders['source_row'] == 4) & (counter_orders['seq'] == 1971)

counter_orders.loc[mask, "exclude_manual"] = True

# ------------------------------------------------------------
# 6.5 Correct sequence 1810: replace order 35 with order 55
# sequence 1810 was recorded as order 35, but the note says:
#
#     "They said 35 was 55."
#
# Correct the observed order number to 55.
# ------------------------------------------------------------

mask = (counter_orders['source_row'] == 4) & (counter_orders['seq'] == 1810)

counter_orders.loc[mask, "numbers_clean"] = pd.Series(
    [["55"]] * mask.sum(),
    index=counter_orders.index[mask]
)

# ------------------------------------------------------------
# 6.6 Explicit "not repeat" corrections
#
# Contemporary notes state that these orders were new orders,
# even though the old logger classified them as repeats.
# ------------------------------------------------------------

force_new = {940: {'76'}, 1701: {'65'}}

# 7. RECALCULATE COUNTER REPEATS USING THE 30-MINUTE RULE
# Historical logger repeat flags are not authoritative.
# Final rule:

counter_orders = counter_orders.sort_values("timestamp").copy()

last_seen = {}

corrected_counts = []
repeat_counts = []

for _, row in counter_orders.iterrows():

    # 7.1 Manually excluded compensating entries contribute zero.
    if row["exclude_manual"]:
        corrected_counts.append(0)
        repeat_counts.append(0)
        continue
    timestamp = row["timestamp"]
    numbers = row["numbers_clean"]

    # --------------------------------------------------------
    # 7.2 If no usable order number exists, count the event as one.
    # This covers unannounced / X-type orders.
    # --------------------------------------------------------
    if len(numbers) == 0:
        corrected_counts.append(1)
        repeat_counts.append(0)
        continue
    new_orders = 0
    repeats = 0
    for number in numbers:
        # 7.3 Some events have explicit notes saying the logger's
        # repeat classification was wrong.
        forced_new_here = (
            row["source_row"] == 4
            and row["seq"] in force_new
            and number in force_new[row["seq"]]
        )
        if forced_new_here:
            is_repeat = False
        elif number not in last_seen:
            is_repeat = False
        else:
            elapsed = timestamp - last_seen[number]
            is_repeat = elapsed <= pd.Timedelta(minutes=30)
        if is_repeat:
            repeats += 1
        else:
            new_orders += 1
        # 7.4 Update the most recent observed time regardless of
        # whether this announcement was classified as repeat.
        last_seen[number] = timestamp
    corrected_counts.append(new_orders)
    repeat_counts.append(repeats)

counter_orders["corrected_count"] = corrected_counts
counter_orders["repeat_count"] = repeat_counts

# ============================================================
# 8. COUNTER THROUGHPUT SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("COUNTER ORDER CORRECTION")
print("=" * 50)

print('Counter order-event rows:', f'{len(counter_orders):,}')

print('Corrected counter orders:', f"{counter_orders['corrected_count'].sum():,}")

print('Repeat announcements excluded:', f"{counter_orders['repeat_count'].sum():,}")

print('Manual compensating entries excluded:', int(counter_orders['exclude_manual'].sum()))

# 9. DRIVE-THRU ORDER CORRECTIONS
# Exclude confirmed erroneous completion entries:
# - Cars 34 and 35: generated when lost tracking records were

# ------------------------------------------------------------
# 9.1 Prepare the analytical drive-thru copy
# ------------------------------------------------------------

drive_thru_orders = analysis_events.loc[analysis_events['kind'] == 'car'].copy()

drive_thru_orders["exclude_manual"] = False
drive_thru_orders["exclusion_reason"] = ""

# ------------------------------------------------------------
# 9.2 Exclude bookkeeping completions: cars 34 and 35
# ------------------------------------------------------------

tracking_closure_mask = (
    (drive_thru_orders["source_row"] == 4)
    & (drive_thru_orders["car_number"].isin([34, 35]))
)

drive_thru_orders.loc[tracking_closure_mask, 'exclude_manual'] = True

drive_thru_orders.loc[
    tracking_closure_mask,
    "exclusion_reason"
] = "Bookkeeping entry from closing a lost tracking record"

# ------------------------------------------------------------
# 9.3 Exclude the confirmed accidental duplicate: car 1318
# ------------------------------------------------------------

duplicate_mask = (drive_thru_orders['source_row'] == 4) & (drive_thru_orders['car_number'] == 1318)

drive_thru_orders.loc[duplicate_mask, 'exclude_manual'] = True

drive_thru_orders.loc[duplicate_mask, 'exclusion_reason'] = 'Confirmed accidental duplicate'

# ------------------------------------------------------------
# 9.3.1 Keep zero-exposure cars in the audit, outside rate/gap samples
# Cars 1011 and 1284 formed isolated windows with no measurable duration.
# This does not say the observed departures were invalid.
zero_exposure_mask = (
    drive_thru_orders['source_row'].eq(4)
    & drive_thru_orders['car_number'].isin([1011, 1284]))
drive_thru_orders.loc[zero_exposure_mask, 'exclude_manual'] = True
drive_thru_orders.loc[zero_exposure_mask, 'exclusion_reason'] = (
    'Observed car retained in audit; isolated zero-exposure window')

# 9.4 Select observed completions for downstream analysis
# ------------------------------------------------------------

drive_thru_final = drive_thru_orders.loc[~drive_thru_orders['exclude_manual']].copy()

# ------------------------------------------------------------
# 9.5 Print the exclusion audit
# ------------------------------------------------------------

print("\nDrive-thru manual exclusion audit:")

print(
    drive_thru_orders.loc[
        drive_thru_orders["exclude_manual"],
        ["source_row", "car_number", "timestamp", "exclusion_reason"]
    ].to_string(index=False)
)

# ============================================================
# 10. DRIVE-THRU THROUGHPUT SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("DRIVE-THRU ORDER CORRECTION")
print("=" * 50)

print('Logged drive-thru car events:', f'{len(drive_thru_orders):,}')

print('Manual exclusions:', f"{drive_thru_orders['exclude_manual'].sum():,}")

print('Final counted drive-thru orders:', f'{len(drive_thru_final):,}')

# ------------------------------------------------------------
# 10.1 Shared note-based interval reliability rules
# ------------------------------------------------------------
BREAK_START = pd.Timestamp('2026-08-30T13:18:42-07:00')
BREAK_END = pd.Timestamp('2026-08-30T13:20:34-07:00')

def unreliable_drive_gap(frame, previous='previous_completion'):
    """Identify the noted missed-car gap and the observer-break overlap."""
    missed = frame['source_row'].eq(4) & frame['car_number'].eq(183)
    interrupted = frame[previous].lt(BREAK_END) & frame['timestamp'].gt(BREAK_START)
    return missed | interrupted

# 11. INTER-TRANSACTION TIMES
# Before relying on observation-window durations, we can study
# the spacing between consecutive completed transactions.

# ------------------------------------------------------------
# 11.1 DRIVE-THRU
# ------------------------------------------------------------

drive_intervals = drive_thru_final[
    ["source_row", "seq", "timestamp", "car_number"]
].sort_values("timestamp").copy()

drive_intervals['gap_seconds'] = drive_intervals['timestamp'].diff().dt.total_seconds()

# 11.2 Keep only gaps belonging to the same continuous period of
# observation.
drive_intervals["valid_interval"] = (
    drive_intervals["gap_seconds"] > 0
) & (
    drive_intervals["gap_seconds"] <= 10 * 60
)

drive_intervals['previous_completion'] = drive_intervals['timestamp'].shift()
drive_intervals['valid_interval'] &= ~unreliable_drive_gap(drive_intervals)

drive_valid_gaps = drive_intervals.loc[drive_intervals['valid_interval'], 'gap_seconds']

# ------------------------------------------------------------
# 11.3 COUNTER
# ------------------------------------------------------------

counter_intervals = counter_orders.loc[
    counter_orders["corrected_count"] > 0,
    ["timestamp", "corrected_count"]
].sort_values("timestamp").copy()

counter_intervals['gap_seconds'] = counter_intervals['timestamp'].diff().dt.total_seconds()

counter_intervals["valid_interval"] = (
    counter_intervals["gap_seconds"] > 0
) & (
    counter_intervals["gap_seconds"] <= 20 * 60
)

counter_valid_gaps = counter_intervals.loc[counter_intervals['valid_interval'], 'gap_seconds']

# ============================================================
# 12. FIRST THROUGHPUT RESULTS
# ============================================================

print("\n" + "=" * 50)
print("INTER-TRANSACTION TIMES")
print("=" * 50)

print("\nDRIVE-THRU")
print(f"Valid intervals: {len(drive_valid_gaps):,}")
print(f"Mean seconds between cars: {drive_valid_gaps.mean():.1f}")
print(f"Median seconds between cars: {drive_valid_gaps.median():.1f}")

print("\nCOUNTER")
print(f"Valid intervals: {len(counter_valid_gaps):,}")
print(f"Mean seconds between order events: {counter_valid_gaps.mean():.1f}")
print(f"Median seconds between order events: {counter_valid_gaps.median():.1f}")

# ============================================================
# 13. BUILD TRUSTWORTHY OBSERVATION WINDOWS
# ============================================================
#
# Goal:
# Create separate observation windows for:
#
#   1. drive-thru
#   2. counter
#
# These windows are allowed to overlap.
#
# If both channels were being observed from 7:00-7:30,
# that contributes:
#
#   30 drive-thru channel-minutes
#   30 counter channel-minutes
#   30 combo-minutes
#   30 unique clock-minutes
#
# We do NOT force each minute into only one mode.
# ============================================================

# ------------------------------------------------------------
# 13.1 Timer identity
# ------------------------------------------------------------
#
# IMPORTANT:
# For timer events, "run_mode" is the authoritative channel.
#
# The regular "mode" field can represent whichever screen was
# active when the timer button was tapped, which is not always
# the timer's actual channel.
# ------------------------------------------------------------

timer_events = analysis_events.loc[
    analysis_events["kind"].isin(["timer_start", "timer_end"])
].copy()

timer_events['timer_mode'] = timer_events['run_mode'].fillna(timer_events['mode'])

# ------------------------------------------------------------
# 13.2 Remove unusable negative-duration timer-end records
# ------------------------------------------------------------

if "duration_sec" in timer_events.columns:
    bad_duration = (
        (timer_events["kind"] == "timer_end")
        & timer_events["duration_sec"].notna()
        & (timer_events["duration_sec"] < 0)
    )
    timer_events = timer_events.loc[~bad_duration].copy()

# ------------------------------------------------------------
# 13.3 Remove exact duplicate timer records
# ------------------------------------------------------------

timer_events = timer_events.drop_duplicates(
    subset=[
        "source_row",
        "run_id",
        "timestamp",
        "kind",
        "timer_mode"
    ]
).copy()

# 14. BUILD CONTINUOUS EVENT SEGMENTS
# Historical run_id values are useful, but some remained open
# across long inactive periods.

def build_event_segments(event_df, mode_name, max_gap_minutes):
    """
    Build continuous observation segments using only valid
    throughput events from one channel.
    """
    df = event_df.sort_values("timestamp").copy()
    df["previous_timestamp"] = df["timestamp"].shift()
    df['gap_minutes'] = (df['timestamp'] - df['previous_timestamp']).dt.total_seconds() / 60

    # 14.1 First row always starts a new segment.
    df['new_segment'] = df['previous_timestamp'].isna() | (df['gap_minutes'] > max_gap_minutes)
    df["segment_id"] = df["new_segment"].cumsum()
    segments = (
        df.groupby("segment_id")
        .agg(
            first_event=("timestamp", "min"),
            last_event=("timestamp", "max"),
            event_rows=("timestamp", "size")
        )
        .reset_index()
    )
    segments["mode"] = mode_name
    return df, segments

# 14.2 Drive-thru segments
drive_segment_events, drive_segments = build_event_segments(
    drive_thru_final,
    mode_name="drive_thru",
    max_gap_minutes=10
)

# 14.3 Counter segments
counter_segment_events, counter_segments = build_event_segments(
    counter_orders.loc[
        counter_orders["corrected_count"] > 0
    ].copy(),
    mode_name="counter",
    max_gap_minutes=20
)

# 15. MATCH SEGMENTS TO NEARBY TIMER BOUNDARIES
# Boundary rules established during cleaning:
# START:

def attach_timer_boundaries(segments, mode_name):
    """
    Attach the nearest reasonable timer start/end to each
    continuous event segment.
    """
    mode_timers = timer_events.loc[
        timer_events["timer_mode"] == mode_name
    ].sort_values("timestamp").copy()
    starts = mode_timers.loc[mode_timers['kind'] == 'timer_start', 'timestamp']
    ends = mode_timers.loc[mode_timers['kind'] == 'timer_end', 'timestamp']
    output_rows = []
    for _, row in segments.iterrows():
        first_event = row["first_event"]
        last_event = row["last_event"]
        # ----------------------------------------------------
        # 15.1 Candidate timer starts before the first event
        # ----------------------------------------------------
        possible_starts = starts.loc[starts <= first_event]
        timer_start = possible_starts.iloc[-1] if len(possible_starts) > 0 else pd.NaT
        # 15.2 Only trust it if it is within 8 minutes.
        if pd.notna(timer_start):
            start_gap = (first_event - timer_start).total_seconds() / 60
        else:
            start_gap = None
        if (
            pd.notna(timer_start)
            and start_gap <= 8
        ):
            effective_start = timer_start
            start_source = "timer"
        else:
            effective_start = first_event
            start_source = "first_event"
        # ----------------------------------------------------
        # 15.3 Candidate timer ends after the last event
        # ----------------------------------------------------
        possible_ends = ends.loc[ends >= last_event]
        timer_end = possible_ends.iloc[0] if len(possible_ends) > 0 else pd.NaT
        # 15.4 Only trust it if it is within 10 minutes.
        if pd.notna(timer_end):
            end_gap = (timer_end - last_event).total_seconds() / 60
        else:
            end_gap = None
        if (
            pd.notna(timer_end)
            and end_gap <= 10
        ):
            effective_end = timer_end
            end_source = "timer"
        else:
            effective_end = last_event
            end_source = "last_event"
        duration_minutes = (effective_end - effective_start).total_seconds() / 60
        output_rows.append(
            {
                "mode": mode_name,
                "segment_id": row["segment_id"],
                "first_event": first_event,
                "last_event": last_event,
                "effective_start": effective_start,
                "effective_end": effective_end,
                "duration_minutes": duration_minutes,
                "start_source": start_source,
                "end_source": end_source,
                "event_rows": row["event_rows"],
            }
        )
    return pd.DataFrame(output_rows)

drive_windows = attach_timer_boundaries(drive_segments, 'drive_thru')

counter_windows = attach_timer_boundaries(counter_segments, 'counter')

# 15.4.1 Remove documented observer breaks from measured exposure

def subtract_observation_breaks(windows, breaks):
    """Split observation windows around unobserved intervals; preserve metadata."""
    rows = []
    for _, window in windows.iterrows():
        pieces = [(window['effective_start'], window['effective_end'])]
        for start, end in breaks:
            remaining = []
            for left, right in pieces:
                if end <= left or start >= right:
                    remaining.append((left, right))
                else:
                    if left < start:
                        remaining.append((left, start))
                    if end < right:
                        remaining.append((end, right))
            pieces = remaining
        for left, right in pieces:
            if right > left:
                record = window.to_dict()
                record.update(effective_start=left, effective_end=right,
                              duration_minutes=(right-left).total_seconds()/60)
                rows.append(record)
    return pd.DataFrame(rows, columns=windows.columns)

before_break_minutes = drive_windows['duration_minutes'].sum()
drive_windows = subtract_observation_breaks(drive_windows, [(BREAK_START, BREAK_END)])
print(f'Observer-break minutes removed: '
      f'{before_break_minutes-drive_windows["duration_minutes"].sum():.3f}')

# 15.5 Combine for easier inspection.
observation_windows = pd.concat([drive_windows, counter_windows], ignore_index=True)

# ============================================================
# 16. CHANNEL-HOUR SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("OBSERVATION WINDOWS")
print("=" * 50)

print("\nWindows by mode:")
print(observation_windows.groupby('mode').size())

print("\nObserved hours by mode:")
print(observation_windows.groupby('mode')['duration_minutes'].sum().div(60).round(2))

# 17. PREPARE DATA FOR OPERATING-DAY PLOTS
#     10:00 AM -> 2:00 AM next day
#     12:00 AM -> 24
#      1:00 AM -> 25
#      2:00 AM -> 26
# The analytical operating day runs from:
# Clock times after midnight are shifted forward:

# ------------------------------------------------------------
# 17.1 Drive-thru operating-day data
# ------------------------------------------------------------

drive_plot = drive_intervals.loc[drive_intervals['valid_interval']].copy()

drive_plot["clock_hour"] = (
    drive_plot["timestamp"].dt.hour
    + drive_plot["timestamp"].dt.minute / 60
    + drive_plot["timestamp"].dt.second / 3600
)

# 17.1.1 Keep only observations between 10 AM and 2 AM.
drive_plot = drive_plot.loc[
    (drive_plot["clock_hour"] >= 10)
    | (drive_plot["clock_hour"] <= 2)
].copy()

# 17.1.2 Shift midnight through 2 AM to the end of the operating day.
drive_plot["operating_hour"] = drive_plot["clock_hour"]

drive_plot.loc[
    drive_plot["clock_hour"] < 10,
    "operating_hour"
] += 24

# 17.1.3 Final hard cutoff at 2:00 AM.
drive_plot = drive_plot.loc[drive_plot['operating_hour'] <= 26].copy()

# ------------------------------------------------------------
# 17.2 Counter operating-day data
# ------------------------------------------------------------

counter_plot = counter_intervals.loc[counter_intervals['valid_interval']].copy()

counter_plot["clock_hour"] = (
    counter_plot["timestamp"].dt.hour
    + counter_plot["timestamp"].dt.minute / 60
    + counter_plot["timestamp"].dt.second / 3600
)

counter_plot = counter_plot.loc[
    (counter_plot["clock_hour"] >= 10)
    | (counter_plot["clock_hour"] <= 2)
].copy()

counter_plot["operating_hour"] = counter_plot["clock_hour"]

counter_plot.loc[
    counter_plot["clock_hour"] < 10,
    "operating_hour"
] += 24

counter_plot = counter_plot.loc[counter_plot['operating_hour'] <= 26].copy()

# ------------------------------------------------------------
# 17.3 Shared operating-day axis labels
# ------------------------------------------------------------

operating_ticks = list(range(10, 27))

operating_labels = [
    "10 AM",
    "11 AM",
    "12 PM",
    "1 PM",
    "2 PM",
    "3 PM",
    "4 PM",
    "5 PM",
    "6 PM",
    "7 PM",
    "8 PM",
    "9 PM",
    "10 PM",
    "11 PM",
    "12 AM",
    "1 AM",
    "2 AM"
]

# 18. TRANSACTION-GAP PLOTS
# These are intentionally POINT plots rather than connected
# time-series lines.

# ============================================================
# 18.1 DRIVE-THRU GAP VS TIME OF DAY
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

ax.scatter(drive_plot['operating_hour'], drive_plot['gap_seconds'], s=24, alpha=0.5)

ax.set_title('Drive-Thru Completion Gaps by Time of Day', fontfamily='Archivo Black')

ax.set_xlabel("Operating-Day Time")
ax.set_ylabel("Seconds Since Previous Drive-Thru Completion")

ax.set_xlim(10, 25)

ax.set_xticks(operating_ticks[:-1])
ax.set_xticklabels(operating_labels[:-1], rotation=45, ha='right')

ax.grid(axis='y', alpha=0.6)

ax.grid(axis='x', visible=False)

fig.tight_layout()

# ------------------------------------------------------------
# 18.1.1 Save drive_gap_by_time.png to images/
# ------------------------------------------------------------

fig.savefig(IMAGE_DIR / 'drive_gap_by_time.png', dpi=200, bbox_inches='tight')

plt.close()

# ============================================================
# 18.2 COUNTER GAP VS TIME OF DAY
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

ax.scatter(counter_plot['operating_hour'], counter_plot['gap_seconds'], s=24, alpha=0.5)

ax.set_title('Counter Completion-Event Gaps by Time of Day', fontfamily='Archivo Black')

ax.set_xlabel("Operating-Day Time")
ax.set_ylabel("Seconds Since Previous Counter Completion Event")

ax.set_xlim(10, 25)

ax.set_xticks(operating_ticks[:-1])
ax.set_xticklabels(operating_labels[:-1], rotation=45, ha='right')

ax.grid(axis='y', alpha=0.6)

ax.grid(axis='x', visible=False)

fig.tight_layout()

# ------------------------------------------------------------
# 18.2.1 Save counter_gap_by_time.png to images/
# ------------------------------------------------------------

fig.savefig(IMAGE_DIR / 'counter_gap_by_time.png', dpi=200, bbox_inches='tight')

plt.close()

# ============================================================
# 18.3 SHARED HOURLY GAP SUMMARY AND PLOT
# ============================================================

def plot_hourly_gap_summary(frame, title, ylabel, filename):
    """Show hourly medians, interquartile ranges, and sample counts."""
    frame['operating_hour_bin'] = frame['operating_hour'].astype(int)
    summary = frame.groupby('operating_hour_bin')['gap_seconds'].agg(
        median='median', q25=lambda x:x.quantile(.25),
        q75=lambda x:x.quantile(.75), observations='size').reset_index()
    fig, ax = plt.subplots(figsize=(14,7))
    centers = summary['operating_hour_bin'] + .5
    ax.errorbar(centers, summary['median'],
                yerr=[summary['median']-summary['q25'],
                      summary['q75']-summary['median']],
                fmt='o', markersize=7, capsize=4, linewidth=1.5,
                label='Median; bars show middle 50% (IQR)')
    for x, row in zip(centers, summary.itertuples()):
        ax.annotate(f'n={row.observations}', (x,row.q75),
                    xytext=(0,6), textcoords='offset points', ha='center', fontsize=8)
    ax.set(title=title, xlabel='Operating-Day Time', ylabel=ylabel, xlim=(10,25))
    ax.title.set_fontfamily('Archivo Black')
    ax.set_xticks(operating_ticks[:-1], operating_labels[:-1], rotation=45, ha='right')
    ax.grid(axis='y', alpha=.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return summary

# 18.3.1 Drive-thru hourly completion gaps
drive_hourly_summary = plot_hourly_gap_summary(
    drive_plot, 'Typical Drive-Thru Completion Gap by Hour', 'Gap (Seconds)',
    'drive_gap_hourly_summary.png')

# 18.4 Counter hourly announcement gaps
counter_hourly_summary = plot_hourly_gap_summary(
    counter_plot, 'Typical Counter Announcement Gap by Hour', 'Gap (Seconds)',
    'counter_gap_hourly_summary.png')

# 18.5 Save sample counts and quartiles with their hourly summaries
# Counts now appear directly on the charts instead of a separate sample-size plot.
drive_hourly_summary.to_csv(CSV_DIR / 'drive_hourly_gap_summary.csv', index=False)
counter_hourly_summary.to_csv(CSV_DIR / 'counter_hourly_gap_summary.csv', index=False)

# 19. COMPLETED ORDERS BY OPERATING HOUR
#     10:00 AM -> 2:00 AM next day
#     12:00 AM -> 24
#      1:00 AM -> 25
#      2:00 AM -> 26
# This section counts actual completed transactions by hour
# across the operating-day window:

# ============================================================
# 19.1 DRIVE-THRU COMPLETED ORDERS BY HOUR
# ============================================================

drive_orders_hourly = drive_thru_final.copy()

drive_orders_hourly["clock_hour"] = (
    drive_orders_hourly["timestamp"].dt.hour
    + drive_orders_hourly["timestamp"].dt.minute / 60
    + drive_orders_hourly["timestamp"].dt.second / 3600
)

# 19.1.1 Keep only the operating-day window.
drive_orders_hourly = drive_orders_hourly.loc[
    (drive_orders_hourly["clock_hour"] >= 10)
    | (drive_orders_hourly["clock_hour"] <= 2)
].copy()

# 19.1.2 Shift after-midnight observations to the end of the same
# operating day.
drive_orders_hourly['operating_hour'] = drive_orders_hourly['clock_hour']

drive_orders_hourly.loc[
    drive_orders_hourly["clock_hour"] < 10,
    "operating_hour"
] += 24

drive_orders_hourly = drive_orders_hourly.loc[drive_orders_hourly['operating_hour'] <= 26].copy()

drive_orders_hourly['operating_hour_bin'] = drive_orders_hourly['operating_hour'].astype(int)

drive_order_counts = (
    drive_orders_hourly
    .groupby("operating_hour_bin")
    .size()
    .reindex(range(10, 27), fill_value=0)
)

# ============================================================
# 19.2 COUNTER COMPLETED ORDERS BY HOUR
# ============================================================

counter_orders_hourly = counter_orders.loc[counter_orders['corrected_count'] > 0].copy()

counter_orders_hourly["clock_hour"] = (
    counter_orders_hourly["timestamp"].dt.hour
    + counter_orders_hourly["timestamp"].dt.minute / 60
    + counter_orders_hourly["timestamp"].dt.second / 3600
)

counter_orders_hourly = counter_orders_hourly.loc[
    (counter_orders_hourly["clock_hour"] >= 10)
    | (counter_orders_hourly["clock_hour"] <= 2)
].copy()

counter_orders_hourly['operating_hour'] = counter_orders_hourly['clock_hour']

counter_orders_hourly.loc[
    counter_orders_hourly["clock_hour"] < 10,
    "operating_hour"
] += 24

counter_orders_hourly = counter_orders_hourly.loc[
    counter_orders_hourly["operating_hour"] <= 26
].copy()

counter_orders_hourly['operating_hour_bin'] = counter_orders_hourly['operating_hour'].astype(int)

# 19.2.1 Sum corrected_count, not event rows.
counter_order_counts = (
    counter_orders_hourly
    .groupby("operating_hour_bin")["corrected_count"]
    .sum()
    .reindex(range(10, 27), fill_value=0)
)

# ============================================================
# 19.3 PRINT HOURLY COUNTS
# ============================================================

print("\n" + "=" * 60)
print("COMPLETED ORDERS BY OPERATING HOUR")
print("=" * 60)

print(f"{'Hour':>6}  {'Drive-Thru':>12}  {'Counter':>10}")

print("-" * 34)

for hour in range(10, 27):
    label = operating_labels[hour - 10]
    drive_count = int(drive_order_counts.loc[hour])
    counter_count = int(counter_order_counts.loc[hour])
    print(f'{label:>6}  {drive_count:>12,}  {counter_count:>10,}')

print("-" * 34)

print(f"{'TOTAL':>6}  {int(drive_order_counts.sum()):>12,}  {int(counter_order_counts.sum()):>10,}")

# ============================================================
# 20. HOURLY OBSERVATION EXPOSURE AND THROUGHPUT
# ============================================================
#
# Raw transaction counts cannot be compared directly across
# hours because observation time varies substantially by hour.
#
# This section:
#
#   1. Splits each cleaned observation window across clock-hour
#      boundaries.
#
#   2. Calculates the actual number of observed minutes within
#      each operating-hour bin.
#
#   3. Calculates completed orders per observed hour:
#
#          orders
#          --------------------  x 60
#          observed minutes
#
# The operating day runs from:
#
#     10:00 AM -> 2:00 AM next day
#
# Hour bins therefore run:
#
#     10-11 AM
#     ...
#     11 PM-midnight
#     midnight-1 AM
#     1-2 AM
#
# ============================================================

# ============================================================
# 20.1 SPLIT OBSERVATION WINDOWS INTO CLOCK-HOUR EXPOSURE
# ============================================================

# 20.1.1 Split windows at hour boundaries and keep operating-day bins

def split_hourly_exposure(windows, opening_hour=10, closing_hour=2):
    """Return observed minutes and operating day for each channel/hour."""
    rows = []
    for window in windows.itertuples():
        hour_start = window.effective_start.floor('h')
        while hour_start < window.effective_end:
            hour_end = hour_start + pd.Timedelta(hours=1)
            left = max(window.effective_start, hour_start)
            right = min(window.effective_end, hour_end)
            hour = hour_start.hour
            operating_hour = hour if hour >= opening_hour else (
                hour+24 if hour < closing_hour else None)
            if right > left and operating_hour is not None:
                rows.append({'mode':window.mode, 'operating_hour':operating_hour,
                             'observed_minutes':(right-left).total_seconds()/60,
                             'operating_day':(hour_start-pd.Timedelta(hours=closing_hour)).date()})
            hour_start = hour_end
    return pd.DataFrame(rows)

hourly_exposure = split_hourly_exposure(observation_windows)

# ============================================================
# 20.2 TOTAL OBSERVED MINUTES BY HOUR AND MODE
# ============================================================

hourly_minutes = (
    hourly_exposure
    .groupby(
        ["operating_hour", "mode"]
    )["observed_minutes"]
    .sum()
    .unstack(fill_value=0)
    .reindex(
        range(10, 26),
        fill_value=0
    )
)

# 20.2.1 Guarantee both columns exist.
for mode_name in ["drive_thru", "counter"]:
    if mode_name not in hourly_minutes.columns:
        hourly_minutes[mode_name] = 0

# ============================================================
# 20.3 COMPLETED ORDERS BY THE SAME HOUR BINS
# ============================================================
#
# Drive-thru:
#     one valid car = one completed order
#
# Counter:
#     use corrected_count because one timestamp may contain
#     several completed orders.
# ============================================================

drive_hourly_orders = (
    drive_orders_hourly
    .loc[
        drive_orders_hourly["operating_hour_bin"] < 26
    ]
    .groupby(
        "operating_hour_bin"
    )
    .size()
    .reindex(
        range(10, 26),
        fill_value=0
    )
)

counter_hourly_orders = (
    counter_orders_hourly
    .loc[
        counter_orders_hourly["operating_hour_bin"] < 26
    ]
    .groupby(
        "operating_hour_bin"
    )["corrected_count"]
    .sum()
    .reindex(
        range(10, 26),
        fill_value=0
    )
)

# ============================================================
# 20.4 CALCULATE EXPOSURE-ADJUSTED HOURLY RATES
# ============================================================

hourly_throughput = pd.DataFrame(
    {
        "drive_orders": drive_hourly_orders,
        "counter_orders": counter_hourly_orders,
        "drive_minutes":
            hourly_minutes["drive_thru"],
        "counter_minutes":
            hourly_minutes["counter"],
    }
)

hourly_throughput["drive_orders_per_hour"] = (
    hourly_throughput["drive_orders"]
    / hourly_throughput["drive_minutes"]
    * 60
)

hourly_throughput["counter_orders_per_hour"] = (
    hourly_throughput["counter_orders"]
    / hourly_throughput["counter_minutes"]
    * 60
)

# 20.4.1 Hours with no observation time should remain missing,
# not become infinity.
hourly_throughput.loc[
    hourly_throughput["drive_minutes"] == 0,
    "drive_orders_per_hour"
] = float("nan")

hourly_throughput.loc[
    hourly_throughput["counter_minutes"] == 0,
    "counter_orders_per_hour"
] = float("nan")

# ============================================================
# 20.4.2 Add independent-day coverage and export the hourly table
hourly_days = hourly_exposure.groupby(['operating_hour', 'mode'])[
    'operating_day'].nunique().unstack(fill_value=0).reindex(range(10,26), fill_value=0)
for mode_name, prefix in [('drive_thru', 'drive'), ('counter', 'counter')]:
    hourly_throughput[f'{prefix}_days'] = hourly_days.get(mode_name, 0)
    hourly_throughput[f'{prefix}_preliminary'] = hourly_throughput[f'{prefix}_days'] < 3
hourly_throughput.to_csv(CSV_DIR / 'hourly_throughput.csv', index_label='operating_hour')
print('Distinct operating days per hour (fewer than 3 = preliminary):')
print(hourly_throughput[['drive_days', 'counter_days']].to_string())

# 20.5 PRINT HOURLY EXPOSURE AND THROUGHPUT
# ============================================================

rate_labels = operating_labels[:-1]

print("\n" + "=" * 86)
print("HOURLY EXPOSURE-ADJUSTED THROUGHPUT")
print("=" * 86)

print(f"{'Hour':>6}  {'DT min':>8}  {'DT ord/hr':>10}  {'Counter min':>11}  {'Counter ord/hr':>14}")

print("-" * 86)

for hour in range(10, 26):
    row = hourly_throughput.loc[hour]
    label = rate_labels[hour - 10]
    drive_rate = row["drive_orders_per_hour"]
    counter_rate = row["counter_orders_per_hour"]
    drive_rate_text = f'{drive_rate:.1f}' if pd.notna(drive_rate) else '--'
    counter_rate_text = f'{counter_rate:.1f}' if pd.notna(counter_rate) else '--'
    print(
        f"{label:>6}  "
        f"{row['drive_minutes']:>8.1f}  "
        f"{drive_rate_text:>10}  "
        f"{row['counter_minutes']:>11.1f}  "
        f"{counter_rate_text:>14}"
    )

# ============================================================
# 20.6 PLOT EXPOSURE-ADJUSTED THROUGHPUT
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

x_positions = list(range(len(hourly_throughput)))

bar_width = 0.38

ax.bar(
    [
        x - bar_width / 2
        for x in x_positions
    ],
    hourly_throughput[
        "drive_orders_per_hour"
    ],
    width=bar_width,
    label="Drive-thru"
)

ax.bar(
    [
        x + bar_width / 2
        for x in x_positions
    ],
    hourly_throughput[
        "counter_orders_per_hour"
    ],
    width=bar_width,
    label="Counter"
)

ax.set_title('Completed Orders per Observed Hour', fontfamily='Archivo Black')

ax.set_xlabel('Operating-Day Time')

ax.set_ylabel('Completed Orders per Observed Hour')

ax.set_xticks(x_positions)

ax.set_xticklabels(rate_labels, rotation=45, ha='right')

for prefix, offset in [('drive', -bar_width/2), ('counter', bar_width/2)]:
    for position, (_, row) in zip(x_positions, hourly_throughput.iterrows()):
        rate = row[f'{prefix}_orders_per_hour']
        days = int(row[f'{prefix}_days'])
        if pd.notna(rate):
            ax.annotate(f'{days}d' + ('*' if days < 3 else ''),
                        (position+offset, rate), xytext=(0,4),
                        textcoords='offset points', ha='center', fontsize=8)
ax.legend(frameon=False)
fig.text(.5,.01, '* Preliminary: fewer than 3 distinct operating days.',
         ha='center', fontsize=10)

ax.grid(axis='y', alpha=0.6)

ax.grid(axis='x', visible=False)

fig.tight_layout(rect=(0,.035,1,1))

# ------------------------------------------------------------
# 20.6.1 Save throughput_by_hour.png to images/
# ------------------------------------------------------------

fig.savefig(IMAGE_DIR / 'throughput_by_hour.png', dpi=200, bbox_inches='tight')

plt.close()

# ============================================================
# 20.7 PLOT OBSERVATION MINUTES BY HOUR
# ============================================================
#
# This companion plot makes sampling coverage explicit.
#
# It should be shown near the throughput-rate plot so readers
# can see which hourly estimates are based on substantial
# observation time and which are based on limited exposure.
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

ax.bar(
    [
        x - bar_width / 2
        for x in x_positions
    ],
    hourly_throughput[
        "drive_minutes"
    ],
    width=bar_width,
    label="Drive-thru"
)

ax.bar(
    [
        x + bar_width / 2
        for x in x_positions
    ],
    hourly_throughput[
        "counter_minutes"
    ],
    width=bar_width,
    label="Counter"
)

ax.set_title('Observed Minutes by Operating Hour', fontfamily='Archivo Black')

ax.set_xlabel('Operating-Day Time')

ax.set_ylabel('Observed Minutes')

ax.set_xticks(x_positions)

ax.set_xticklabels(rate_labels, rotation=45, ha='right')

ax.legend(frameon=False)

ax.grid(axis='y', alpha=0.6)

ax.grid(axis='x', visible=False)

fig.tight_layout()

# ------------------------------------------------------------
# 20.7.1 Save observation_minutes_by_hour.png to images/
# ------------------------------------------------------------

fig.savefig(IMAGE_DIR / 'observation_minutes_by_hour.png', dpi=200, bbox_inches='tight')

plt.close()

# 21. SIMULTANEOUS DRIVE-THRU VS COUNTER THROUGHPUT
# This analysis uses ONLY clock time during which both the
# drive-thru and counter were simultaneously being observed.

# ============================================================
# 21.1 MERGE OVERLAPPING WINDOWS WITHIN EACH CHANNEL
# ============================================================
#
# Before comparing the channels, merge any overlapping or
# touching windows within the same mode.
#
# This prevents the same clock minute from being counted twice.
# ============================================================

def merge_observation_windows(window_df):
    windows = window_df[['effective_start', 'effective_end']].sort_values('effective_start').copy()
    merged = []
    for _, row in windows.iterrows():
        start = row["effective_start"]
        end = row["effective_end"]
        if len(merged) == 0:
            merged.append([start, end])
            continue
        previous_start = merged[-1][0]
        previous_end = merged[-1][1]
        # 21.1.1 Merge overlapping or touching windows.
        if start <= previous_end:
            merged[-1][1] = max(previous_end, end)
        else:
            merged.append([start, end])
    return pd.DataFrame(merged, columns=['effective_start', 'effective_end'])

drive_merged_windows = merge_observation_windows(
    observation_windows.loc[
        observation_windows["mode"] == "drive_thru"
    ]
)

counter_merged_windows = merge_observation_windows(
    observation_windows.loc[
        observation_windows["mode"] == "counter"
    ]
)

# ============================================================
# 21.2 FIND EXACT CLOCK-TIME OVERLAPS
# ============================================================

overlap_rows = []

for _, drive_window in drive_merged_windows.iterrows():
    for _, counter_window in counter_merged_windows.iterrows():
        overlap_start = max(drive_window['effective_start'], counter_window['effective_start'])
        overlap_end = min(drive_window['effective_end'], counter_window['effective_end'])
        overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60
        # 21.2.1 Positive duration means both channels were actively
        # observed during the same clock period.
        if overlap_minutes > 0:
            overlap_rows.append(
                {
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "overlap_minutes": overlap_minutes
                }
            )

overlap_windows = pd.DataFrame(overlap_rows)

# ============================================================
# 21.3 COUNT COMPLETED ORDERS INSIDE EACH OVERLAP WINDOW
# ============================================================

overlap_results = []

for overlap_number, row in overlap_windows.iterrows():
    start = row["overlap_start"]
    end = row["overlap_end"]
    minutes = row["overlap_minutes"]

    # --------------------------------------------------------
    # 21.3.1 Drive-thru completed orders
    # --------------------------------------------------------
    drive_mask = (drive_thru_final['timestamp'] >= start) & (drive_thru_final['timestamp'] <= end)
    drive_orders = int(drive_mask.sum())

    # --------------------------------------------------------
    # 21.3.2 Counter completed orders
    #
    # Sum corrected_count rather than counting event rows.
    # --------------------------------------------------------
    counter_mask = (counter_orders['timestamp'] >= start) & (counter_orders['timestamp'] <= end)
    counter_completed = int(counter_orders.loc[counter_mask, 'corrected_count'].sum())

    # --------------------------------------------------------
    # 21.3.3 Exposure-adjusted rates
    # --------------------------------------------------------
    drive_rate = drive_orders / minutes * 60
    counter_rate = counter_completed / minutes * 60
    overlap_results.append(
        {
            "overlap_id": overlap_number + 1,
            "start": start,
            "end": end,
            "minutes": minutes,
            "drive_orders": drive_orders,
            "counter_orders": counter_completed,
            "drive_orders_per_hour": drive_rate,
            "counter_orders_per_hour": counter_rate
        }
    )

overlap_results = pd.DataFrame(overlap_results)

# ============================================================
# 21.4 POOLED OVERLAP-ONLY RESULTS
# ============================================================
#
# Instead of averaging the individual window rates, pool the
# actual completed orders and actual overlap exposure.
#
# This properly gives longer observation periods more weight.
# ============================================================

total_overlap_minutes = overlap_results['minutes'].sum()

total_overlap_drive_orders = overlap_results['drive_orders'].sum()

total_overlap_counter_orders = overlap_results['counter_orders'].sum()

pooled_drive_rate = total_overlap_drive_orders / total_overlap_minutes * 60

pooled_counter_rate = total_overlap_counter_orders / total_overlap_minutes * 60

rate_difference = pooled_drive_rate - pooled_counter_rate

rate_ratio = pooled_drive_rate / pooled_counter_rate

print("\n" + "=" * 72)
print("SIMULTANEOUS DRIVE-THRU VS COUNTER OBSERVATION")
print("=" * 72)

print(f'Overlap windows: {len(overlap_results):,}')

print(f'Total simultaneous observation: {total_overlap_minutes / 60:.2f} hours')

print()

print(f'Drive-thru completed orders: {total_overlap_drive_orders:,}')

print(f'Counter completed orders: {total_overlap_counter_orders:,}')

print()

print(f'Drive-thru throughput: {pooled_drive_rate:.1f} orders/hour')

print(f'Counter throughput: {pooled_counter_rate:.1f} orders/hour')

print()

print(f'Drive-thru minus counter: {rate_difference:.1f} orders/hour')

print(f'Drive-thru / counter ratio: {rate_ratio:.2f}')

# ============================================================
# 21.5 INDIVIDUAL OVERLAP WINDOWS
# ============================================================
#
# Extremely short overlaps can produce unstable hourly rates.
#
# For window-by-window comparisons, require at least 10 minutes
# of simultaneous observation.
#
# The pooled result above still uses ALL simultaneous minutes.
# ============================================================

overlap_comparison = overlap_results.loc[overlap_results['minutes'] >= 10].copy()

print("\n" + "=" * 72)
print("OVERLAP WINDOWS WITH AT LEAST 10 MINUTES OF EXPOSURE")
print("=" * 72)

print(
    overlap_comparison[
        [
            "start",
            "minutes",
            "drive_orders",
            "counter_orders",
            "drive_orders_per_hour",
            "counter_orders_per_hour"
        ]
    ].round(
        {
            "minutes": 1,
            "drive_orders_per_hour": 1,
            "counter_orders_per_hour": 1
        }
    ).to_string(
        index=False
    )
)

# ============================================================
# 21.6 PAIRED THROUGHPUT BY SIMULTANEOUS OBSERVATION START
# ============================================================
paired = overlap_comparison.sort_values('start').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14,7))
x = np.arange(len(paired))
for i, row in paired.iterrows():
    ax.plot([i,i], [row['drive_orders_per_hour'], row['counter_orders_per_hour']],
            color=brand_colors[5], alpha=.35, linewidth=1)
ax.scatter(x, paired['drive_orders_per_hour'], label='Drive-thru', color=brand_colors[0])
ax.scatter(x, paired['counter_orders_per_hour'], label='Counter', color=brand_colors[1])
ax.set_xticks(x, paired['start'].dt.strftime('%b %d %I:%M %p'), rotation=45, ha='right')
ax.set(title='Drive-Thru and Counter During the Same Observation Periods',
       xlabel='Observation Start', ylabel='Completed Orders per Observed Hour')
ax.title.set_fontfamily('Archivo Black')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.35)
fig.tight_layout()
fig.savefig(IMAGE_DIR / 'simultaneous_throughput_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)
paired.to_csv(CSV_DIR / 'simultaneous_throughput_comparison.csv', index=False)

# ============================================================
# 22. COMPLETION-GAP DISTRIBUTION CHECK
# ============================================================
#
# Compare each channel's observed completion gaps with an
# exponential distribution having the same mean.
#
# Exponential reference:
#   standard deviation / mean = 1
#   median / mean = approximately 0.693
#
# Important limitations:
# - These are completion gaps, not customer arrival gaps.
# - Counter gaps are between completion EVENTS; one event may
#   contain several completed orders.
# - Existing cleaning excludes gaps above 10 minutes for
#   drive-thru and 20 minutes for counter.
# - Pooling different times and sessions can mix different rates.
# - Distribution shape alone does not establish independence
#   or a constant rate, both needed for a homogeneous Poisson
#   process.
# ============================================================

# ------------------------------------------------------------
# 22.1 Prepare both channels' valid completion gaps
# ------------------------------------------------------------

gap_channels = {'Drive-thru': drive_valid_gaps, 'Counter completion events': counter_valid_gaps}

distribution_rows = []

# ------------------------------------------------------------
# 22.2 Calculate descriptive distribution statistics
# ------------------------------------------------------------

for channel_name, channel_gaps in gap_channels.items():

    gaps = pd.to_numeric(channel_gaps, errors='coerce').dropna()

    gaps = gaps.loc[gaps > 0].copy()

    if len(gaps) < 2:
        raise ValueError(
            f"Too few valid gaps for {channel_name}."
        )

    gap_channels[channel_name] = gaps

    mean_gap = gaps.mean()
    median_gap = gaps.median()
    sd_gap = gaps.std(ddof=1)

    distribution_rows.append({
        "channel": channel_name,
        "intervals": len(gaps),
        "mean_seconds": mean_gap,
        "median_seconds": median_gap,
        "sd_seconds": sd_gap,
        "coefficient_of_variation": sd_gap / mean_gap,
        "median_mean_ratio": median_gap / mean_gap,
        "exponential_median_seconds": mean_gap * 0.69314718056,
        "p90_seconds": gaps.quantile(0.90),
        "p95_seconds": gaps.quantile(0.95),
    })

gap_distribution_summary = pd.DataFrame(distribution_rows).set_index('channel')

# ------------------------------------------------------------
# 22.3 Print observed statistics and exponential references
# ------------------------------------------------------------

print("\n" + "=" * 76)
print("COMPLETION-GAP DISTRIBUTION CHECK")
print("=" * 76)

print(gap_distribution_summary.to_string(float_format=lambda value: f'{value:.3f}'))

print("\nExponential reference values:")
print("  Coefficient of variation: 1.000")
print("  Median / mean:            0.693")

print("\nInterpretation:")
print("  CV below 1 suggests more regular spacing than the reference.")
print("  CV above 1 suggests greater variability than the reference.")
print("  These are descriptive comparisons, not formal test results.")
print("  Existing gap cutoffs can affect these statistics.")

# ------------------------------------------------------------
# 22.4 Plot observed and reference survival curves
# ------------------------------------------------------------
#
# Survival probability = proportion of gaps longer than x.
#
# Each channel uses its own observed mean for the exponential
# reference. The global plot palette controls the line colors.
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for ax, (channel_name, gaps) in zip(
    axes,
    gap_channels.items()
):

    mean_gap = gaps.mean()

    # --------------------------------------------------------
    # 22.4.1 Calculate empirical survival with ties handled
    # --------------------------------------------------------

    gap_counts = gaps.value_counts().sort_index()

    survival_values = 1 - gap_counts.cumsum() / len(gaps)

    empirical_x = [0.0] + gap_counts.index.tolist()
    empirical_y = [1.0] + survival_values.tolist()

    ax.step(empirical_x, empirical_y, where='post', linewidth=2, label='Observed gaps')

    # --------------------------------------------------------
    # 22.4.2 Calculate the same-mean exponential reference
    # --------------------------------------------------------

    plot_max = float(gaps.max())

    reference_x = [plot_max * step / 400 for step in range(401)]

    reference_y = [2.718281828459045 ** (-seconds / mean_gap) for seconds in reference_x]

    ax.plot(reference_x, reference_y, linestyle='--', linewidth=2, label='Exponential reference')

    # --------------------------------------------------------
    # 22.4.3 Format the channel panel
    # --------------------------------------------------------

    ax.set_title(channel_name)
    ax.set_xlabel("Completion Gap (Seconds)")
    ax.set_xlim(0, plot_max)
    ax.set_ylim(0, 1.02)

    ax.grid(axis='both', alpha=0.4)

    ax.legend(frameon=False)

axes[0].set_ylabel("Proportion of Gaps Longer Than x")

fig.suptitle(
    "Observed Completion Gaps vs. Exponential Reference",
    fontfamily="Archivo Black",
    fontsize=18
)

fig.tight_layout()

# ------------------------------------------------------------
# 22.5 Save the completion-gap distribution comparison
# ------------------------------------------------------------

fig.savefig(IMAGE_DIR / 'completion_gap_exponential_comparison.png', dpi=200, bbox_inches='tight')

plt.close(fig)

# ============================================================
# 23. MOVEMENT PAUSES AND COMPLETION-GAP DISTRIBUTIONS
# ============================================================
# A block is monitored when its drive-thru window has either
# movement marker. Unmarked gaps in those blocks mean no material
# pause recorded. Exponential curves are descriptive references,
# not proof of a Poisson process or causal throughput losses.

# ------------------------------------------------------------
# 23.1 Assign markers to existing drive-thru observation blocks
# ------------------------------------------------------------
movement_markers = analysis_events.loc[
    analysis_events['kind'].isin(['not_moving_up', 'moved_up'])
].sort_values('timestamp').copy()
# 23.1.1 Remove retracted movement-marker entries
# seq 2041/2057: accidental September 1 pause and its closing tap.
# seq 873: August 23 indicator reset; note says no delay occurred.
retracted_marker_mask = (
    (movement_markers['source_row'] == 4)
    & movement_markers['seq'].isin([873, 2041, 2057])
)
print(f'Note-based movement markers excluded: {retracted_marker_mask.sum()}')
movement_markers = movement_markers.loc[~retracted_marker_mask].copy()
movement_markers['movement_block'] = pd.NA

for window in drive_windows.itertuples():
    inside = movement_markers['timestamp'].between(window.effective_start, window.effective_end)
    if movement_markers.loc[inside, 'movement_block'].notna().any():
        raise ValueError('Movement marker belongs to overlapping drive windows.')
    movement_markers.loc[inside, 'movement_block'] = window.segment_id

monitored_blocks = movement_markers['movement_block'].dropna().unique()
movement_audit = []
movement_pairs = []

# ------------------------------------------------------------
# 23.2 Pair markers within each block and flag incomplete episodes
# ------------------------------------------------------------
for block_id in monitored_blocks:
    markers = movement_markers.loc[movement_markers['movement_block'] == block_id]
    pending = None
    for marker in markers.itertuples():
        if marker.kind == 'not_moving_up':
            if pending is None:
                pending = marker.timestamp
            else:
                movement_audit.append((block_id, marker.timestamp, 'Repeated pause start'))
        elif pending is None:
            movement_audit.append((block_id, marker.timestamp, 'Up without pause start'))
        else:
            movement_pairs.append({
                'segment_id': block_id,
                'pause_start': pending,
                'up_time': marker.timestamp,
                'pause_seconds': (marker.timestamp - pending).total_seconds(),
            })
            pending = None
    if pending is not None:
        movement_audit.append((block_id, pending, 'Pause without closing up'))

pause_pairs = pd.DataFrame(movement_pairs)
if pause_pairs.empty:
    raise ValueError('No paired movement pauses found.')

# ------------------------------------------------------------
# 23.3 Match paired pauses to consecutive completed cars
# ------------------------------------------------------------
monitored_gaps = drive_segment_events.loc[
    drive_segment_events['segment_id'].isin(monitored_blocks)
].sort_values(['segment_id', 'timestamp']).copy()
monitored_gaps['previous_completion'] = monitored_gaps.groupby('segment_id')['timestamp'].shift()
monitored_gaps['completion_gap_seconds'] = (
    monitored_gaps['timestamp'] - monitored_gaps['previous_completion']
).dt.total_seconds()
monitored_gaps = monitored_gaps.loc[
    monitored_gaps['completion_gap_seconds'].between(0, 600, inclusive='right')
].reset_index(drop=True)
# 23.3.1 Exclude gaps explicitly identified as unreliable
# Keep observed completions; this exclusion affects this distribution
# sample only. Exposure corrections must be implemented separately.
unreliable = unreliable_drive_gap(monitored_gaps)
print(f'Unreliable monitored gaps excluded: {unreliable.sum()}')
monitored_gaps = monitored_gaps.loc[~unreliable].reset_index(drop=True)
monitored_gaps['has_pause_marker'] = False
matched_episodes = []

for gap_id, gap in monitored_gaps.iterrows():
    block_markers = movement_markers.loc[movement_markers['movement_block'] == gap['segment_id']]
    contains_marker = (
        (block_markers['timestamp'] > gap['previous_completion'])
        & (block_markers['timestamp'] <= gap['timestamp'])
    ).any()
    monitored_gaps.loc[gap_id, 'has_pause_marker'] = contains_marker
    pairs = pause_pairs.loc[
        (pause_pairs['segment_id'] == gap['segment_id'])
        & (pause_pairs['pause_start'] >= gap['previous_completion'])
        & (pause_pairs['up_time'] <= gap['timestamp'])
        & (pause_pairs['pause_seconds'] > 0)
    ]
    # Clean decomposition: exactly one complete episode and two markers.
    markers_in_gap = block_markers.loc[
        (block_markers['timestamp'] >= gap['previous_completion'])
        & (block_markers['timestamp'] <= gap['timestamp'])
    ]
    if len(pairs) == 1 and len(markers_in_gap) == 2:
        pair = pairs.iloc[0]
        matched_episodes.append({
            'segment_id': gap['segment_id'],
            'completion_time': gap['timestamp'],
            'up_time': pair['up_time'],
            'completion_gap_seconds': gap['completion_gap_seconds'],
            'pause_seconds': pair['pause_seconds'],
            'up_to_completion_seconds': (
                gap['timestamp'] - pair['up_time']
            ).total_seconds(),
        })

pause_completion_matches = pd.DataFrame(matched_episodes)
if pause_completion_matches.empty:
    raise ValueError('No clean pause-to-completion matches; inspect markers.')

# ------------------------------------------------------------
# 23.4 Flag explicitly late up taps and print diagnostics
# ------------------------------------------------------------
print('\n' + '=' * 76)
print('MOVEMENT-PAUSE DISTRIBUTION ANALYSIS')
print('=' * 76)
print(f'Monitored drive-thru blocks: {len(monitored_blocks)}')
print(f'Markers outside drive windows: {movement_markers["movement_block"].isna().sum()}')
print(f'Paired pause episodes: {len(pause_pairs)}')
print(f'Clean episodes matched within one completion gap: {len(pause_completion_matches)}')
print('\nCompletion-gap groups:')
print(monitored_gaps['has_pause_marker'].value_counts().rename(
    index={True: 'Movement marker present', False: 'No material pause recorded'}
))
print(f'Pairing audit flags: {len(movement_audit)}')
for item in movement_audit[:12]:
    print(item)
# 23.4.1 Flag candidate matches around explicit late-up notes
# Flagging is deliberately conservative: nearest preceding up within
# two minutes. It identifies sensitivity cases without changing times.
late_up_notes = analysis_events.loc[
    (analysis_events['source_row'] == 4)
    & analysis_events['seq'].isin([1568, 1597, 1781])
]
pause_completion_matches['late_up_flag'] = False
for note in late_up_notes.itertuples():
    elapsed = (note.timestamp - pause_completion_matches['up_time']).dt.total_seconds()
    candidates = elapsed.loc[elapsed.between(0, 120)]
    if not candidates.empty:
        pause_completion_matches.loc[candidates.idxmin(), 'late_up_flag'] = True
print(f'Clean matches flagged near late-up notes: {pause_completion_matches["late_up_flag"].sum()}')
print('Late-up flags identify candidates; recorded times are not shifted.')
print('Approximate timestamp uncertainty: 5 seconds (observer estimate).')
print('A measured pause is not automatically time lost from throughput.')

# ------------------------------------------------------------
# 23.5 Compare survival curves and exponential Q-Q distributions
# ------------------------------------------------------------
def draw_gap_survival(ax, values, label):
    counts = values.value_counts().sort_index()
    ax.step([0.0] + counts.index.tolist(),
            [1.0] + (1 - counts.cumsum() / len(values)).tolist(),
            where='post', label=label, linewidth=2)

movement_gap_groups = {
    'No material pause recorded': monitored_gaps.loc[
        ~monitored_gaps['has_pause_marker'], 'completion_gap_seconds'],
    'Clean paired pause': pause_completion_matches['completion_gap_seconds'],
}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for label, values in movement_gap_groups.items():
    draw_gap_survival(axes[0], values, label)
axes[0].set_title('Completion-Gap Distributions')
axes[0].set_xlabel('Completion Gap (Seconds)')
axes[0].set_ylabel('Proportion Longer Than x')
axes[0].legend(frameon=False)

for ax, (label, values) in zip(axes[1:], movement_gap_groups.items()):
    observed = sorted(values.tolist())
    mean_gap = values.mean()
    theoretical = [
        -mean_gap * math.log(1 - (i + 0.5) / len(observed))
        for i in range(len(observed))
    ]
    ax.scatter(theoretical, observed, s=18, alpha=0.65, label='Observed quantiles')
    limit = max(max(theoretical), max(observed))
    ax.plot([0, limit], [0, limit], linestyle='--', label='Exponential reference')
    ax.set_title(label)
    ax.set_xlabel('Same-Mean Exponential Quantile (Seconds)')
    ax.set_ylabel('Observed Gap Quantile (Seconds)')
    ax.legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.35)
fig.tight_layout()
fig.savefig(IMAGE_DIR / 'movement_gap_distributions.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------
# 23.6 Examine pause duration versus time after advancing
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(
    pause_completion_matches['pause_seconds'],
    pause_completion_matches['up_to_completion_seconds'],
    alpha=0.65, s=30
)
axes[0].set_title('What Happens After the Car Advances?')
axes[0].set_xlabel('Recorded Advance Pause (Seconds)')
axes[0].set_ylabel('Up to Next Completion (Seconds)')
axes[0].set_xlim(left=0)
axes[0].set_ylim(bottom=0)
for column, label in [
    ('pause_seconds', 'Recorded advance pause'),
    ('up_to_completion_seconds', 'Up to next completion'),
]:
    draw_gap_survival(axes[1], pause_completion_matches[column], label)
axes[1].set_title('Pause and Post-Advance Distributions')
axes[1].set_xlabel('Seconds')
axes[1].set_ylabel('Proportion Longer Than x')
axes[1].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.35)
fig.tight_layout()
fig.savefig(IMAGE_DIR / 'pause_vs_post_advance_distributions.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ============================================================
# 24. EXPONENTIAL VS GAMMA DISTRIBUTION FITS
# ============================================================
# Exponential is the gamma special case with shape = 1.
# Models have location fixed at zero.
# Completion-gap fits account for the existing 600-second cutoff.
# Post-advance fits describe selected, clean matched episodes;
# their selection by TOTAL gap cannot be corrected with a simple
# upper cutoff on post-advance time alone. These are exploratory
# shape comparisons, not population estimates or causal effects.

# ------------------------------------------------------------
# 24.1 Prepare the two distribution samples
# ------------------------------------------------------------
fit_samples = {
    'Completion gaps without a recorded pause': (
        monitored_gaps.loc[
            ~monitored_gaps['has_pause_marker'], 'completion_gap_seconds'
        ], 600.0
    ),
    'Time from advancing to pickup': (
        pause_completion_matches['up_to_completion_seconds'], None
    ),
}
fit_rows = []
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ------------------------------------------------------------
# 24.2 Fit positive-parameter models by maximum likelihood
# ------------------------------------------------------------
def fit_movement_model(values, model_name, upper_limit):
    shape0, _, scale0 = stats.gamma.fit(values, floc=0)
    def objective(parameters):
        if model_name == 'Gamma':
            shape, scale = np.exp(parameters)
            model = stats.gamma(shape, loc=0, scale=scale)
        else:
            shape = 1.0
            scale = np.exp(parameters[0])
            model = stats.expon(loc=0, scale=scale)
        log_likelihood = model.logpdf(values)
        if upper_limit is not None:
            log_likelihood = log_likelihood - model.logcdf(upper_limit)
        if not np.isfinite(log_likelihood).all():
            return 1e100
        return -log_likelihood.sum()
    starts = (
        [np.log([shape0, scale0]), np.log([1.0, values.mean()])]
        if model_name == 'Gamma' else [np.log([values.mean()])]
    )
    # 24.2.1 Use derivative-free optimization to avoid numerical
    # differentiation across invalid likelihood evaluations.
    solutions = [optimize.minimize(
        objective, start, method='Nelder-Mead',
        bounds=[(-9, 14)] * len(start),
        options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-8}
    ) for start in starts]
    successful = [s for s in solutions if s.success and np.isfinite(s.fun) and s.fun < 1e99]
    if not successful:
        raise ValueError(f'{model_name} fit failed: {solutions[0].message}')
    solution = min(successful, key=lambda s: s.fun)
    if model_name == 'Gamma':
        shape, scale = np.exp(solution.x)
        model = stats.gamma(shape, loc=0, scale=scale)
        parameter_count = 2
    else:
        shape = 1.0
        scale = np.exp(solution.x[0])
        model = stats.expon(loc=0, scale=scale)
        parameter_count = 1
    return model, shape, scale, 2 * parameter_count + 2 * solution.fun

# ------------------------------------------------------------
# 24.3 Fit and draw survival and Q-Q comparisons for each sample
# ------------------------------------------------------------
for row_id, (sample_name, (series, upper_limit)) in enumerate(fit_samples.items()):
    values = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
    zero_count = int((values == 0).sum())
    values = np.sort(values[values > 0])
    if len(values) < 10:
        raise ValueError(f'Too few positive observations for {sample_name}.')
    probabilities = (np.arange(len(values)) + 0.5) / len(values)
    grid = np.linspace(0, upper_limit or values.max(), 500)
    draw_gap_survival(axes[row_id, 0], pd.Series(values), 'Observed')
    for column_id, model_name in enumerate(['Exponential', 'Gamma'], start=1):
        model, shape, scale, aic = fit_movement_model(values, model_name, upper_limit)
        # ----------------------------------------------------
        # 24.3.1 Condition fitted curves on the completion cutoff
        # ----------------------------------------------------
        normalization = model.cdf(upper_limit) if upper_limit else 1.0
        survival = np.clip(1 - model.cdf(grid) / normalization, 0, 1)
        expected = model.ppf(probabilities * normalization)
        axes[row_id, 0].plot(grid, survival,
                             label=model_name, linewidth=2, linestyle='--' if model_name == 'Gamma' else ':')
        ax = axes[row_id, column_id]
        ax.scatter(expected, values, s=18, alpha=0.65)
        limit = max(expected.max(), values.max())
        ax.plot([0, limit], [0, limit], linestyle='--')
        ax.set_xlim(0, expected.max()*1.05)
        ks_distance = stats.kstest(values, lambda x: np.clip(model.cdf(x)/normalization,0,1)).statistic
        ax.set_title(f'{model_name} Q-Q | n={len(values)}', fontfamily='Archivo Black')
        ax.text(.03,.97, f'shape={shape:.2f} | model mean={model.mean():.1f}s\nKS distance={ks_distance:.3f}',
                transform=ax.transAxes, va='top', fontsize=10)
        ax.set_xlabel('Fitted Model Quantile (Seconds)')
        ax.set_ylabel('Observed Quantile (Seconds)')
        fit_rows.append({
            'sample': sample_name, 'model': model_name,
            'n': len(values), 'gamma_shape': shape,
            'scale_seconds': scale, 'model_mean_seconds': model.mean(),
            'KS_distance': ks_distance, 'AIC': aic,
            'zero_times_omitted': zero_count,
        })
    axes[row_id, 0].set_title(sample_name, fontfamily='Archivo Black')
    axes[row_id, 0].set_xlabel('Seconds')
    axes[row_id, 0].set_ylabel('Proportion Longer Than x')
    axes[row_id, 0].legend(frameon=False)
for ax in axes.flat:
    ax.grid(alpha=0.35)
fig.tight_layout()

# ------------------------------------------------------------
# 24.4 Print within-sample model comparisons
# ------------------------------------------------------------
movement_model_results = pd.DataFrame(fit_rows)
movement_model_results['delta_AIC'] = (
    movement_model_results['AIC']
    - movement_model_results.groupby('sample')['AIC'].transform('min')
)
print('\n' + '=' * 76)
print('EXPONENTIAL VS GAMMA: MOVEMENT DISTRIBUTIONS')
print('=' * 76)
print(movement_model_results.to_string(index=False, float_format=lambda value: f'{value:.3f}'))
print('\nCompare AIC only within the same sample: smaller fits better')
print('after penalizing extra parameters; it does not prove a good fit.')
print('Gamma shape > 1 allows more concentrated gaps than exponential.')
print('These pooled fits do not test independence or establish a Poisson process.')
print('Post-advance fits describe the selected clean episodes only.')
movement_model_results.to_csv(CSV_DIR / 'movement_model_fits.csv', index=False)

# ------------------------------------------------------------
# 24.5 Save the distribution-model comparison
# ------------------------------------------------------------
fig.savefig(IMAGE_DIR / 'movement_gamma_exponential_fits.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ============================================================
# 25. CONSECUTIVE COMPLETION GAPS
# ============================================================
# Counter gaps describe announcement events, not individual orders.
# Correlation does not prove independence or identify a bottleneck.

# ------------------------------------------------------------
# 25.1 Preserve event order and split at unreliable intervals
# ------------------------------------------------------------
def prepare_gap_sequence(events, mode):
    sequence = events.sort_values(['segment_id', 'timestamp']).copy()
    groups = sequence.groupby('segment_id', sort=False)
    sequence['previous_completion'] = groups['timestamp'].shift()
    sequence['gap_seconds'] = (
        sequence['timestamp'] - sequence['previous_completion']
    ).dt.total_seconds()
    cutoff = 600 if mode == 'drive_thru' else 1200
    valid = sequence['gap_seconds'].between(0, cutoff, inclusive='right')
    if mode == 'drive_thru':
        valid &= ~unreliable_drive_gap(sequence)
    # An invalid gap breaks adjacency; filtering must not join its neighbors.
    sequence['chain'] = (~valid).groupby(sequence['segment_id']).cumsum()
    sequence = sequence.loc[valid].copy()
    sequence['centered_gap'] = sequence['gap_seconds'] - sequence.groupby(
        'segment_id')['gap_seconds'].transform('mean')
    # Midnight to 2 AM belongs to the preceding operating day.
    sequence['operating_day'] = (sequence['timestamp'] - pd.Timedelta(hours=2)).dt.date
    return sequence

# ------------------------------------------------------------
# 25.2 Build actual neighboring pairs at lags 1 through 10
# ------------------------------------------------------------
def neighboring_gap_pairs(sequence, lag):
    groups = sequence.groupby(['segment_id', 'chain'], sort=False)
    pairs = sequence[['operating_day', 'gap_seconds', 'centered_gap']].copy()
    pairs['previous_gap'] = groups['gap_seconds'].shift(lag)
    pairs['previous_centered'] = groups['centered_gap'].shift(lag)
    pairs['previous_day'] = groups['operating_day'].shift(lag)
    pairs = pairs.dropna(subset=['previous_gap'])
    # Keep each pair within one operating day for whole-day resampling.
    return pairs.loc[pairs['operating_day'].eq(pairs['previous_day'])].copy()

# ------------------------------------------------------------
# 25.3 Resample operating days using sufficient statistics
# ------------------------------------------------------------
def correlation_from_totals(totals):
    n, sx, sy, sxx, syy, sxy = np.asarray(totals).T
    numerator = sxy - sx * sy / np.maximum(n, 1)
    denominator = np.sqrt(
        np.maximum(sxx - sx * sx / np.maximum(n, 1), 0)
        * np.maximum(syy - sy * sy / np.maximum(n, 1), 0)
    )
    return np.divide(numerator, denominator,
                     out=np.full_like(numerator, np.nan, dtype=float),
                     where=(n >= 3) & (denominator > 0))

def day_cluster_interval(pairs, x_column, y_column, rng, draws=1000):
    x = pairs[x_column].to_numpy(dtype=float)
    y = pairs[y_column].to_numpy(dtype=float)
    moments = pd.DataFrame({
        'day': pairs['operating_day'].to_numpy(),
        'n': np.ones(len(pairs)), 'sx': x, 'sy': y,
        'sxx': x*x, 'syy': y*y, 'sxy': x*y,
    })
    daily = moments.groupby('day').sum().to_numpy()
    estimate = float(correlation_from_totals(daily.sum(axis=0)))
    if len(daily) < 3:
        return estimate, np.nan, np.nan, len(daily)
    sampled_days = rng.integers(0, len(daily), size=(draws, len(daily)))
    estimates = correlation_from_totals(daily[sampled_days].sum(axis=1))
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) < draws * 0.9:
        return estimate, np.nan, np.nan, len(daily)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return estimate, lower, upper, len(daily)

# ------------------------------------------------------------
# 25.4 Calculate pooled and within-block correlations
# ------------------------------------------------------------
gap_sequence_rng = np.random.default_rng(2506)
gap_sequence_results = []
gap_sequence_samples = {}
gap_lag_one_pairs = {}

for channel, event_data in [
    ('drive_thru', drive_segment_events),
    ('counter', counter_segment_events),
]:
    sequence = prepare_gap_sequence(event_data, channel)
    gap_sequence_samples[channel] = sequence
    for lag in range(1, 11):
        pairs = neighboring_gap_pairs(sequence, lag)
        if lag == 1:
            gap_lag_one_pairs[channel] = pairs
        if len(pairs) < 3:
            continue
        for label, x_column, y_column in [
            ('Pooled', 'previous_gap', 'gap_seconds'),
            ('Within block', 'previous_centered', 'centered_gap'),
        ]:
            r, lower, upper, days = day_cluster_interval(
                pairs, x_column, y_column, gap_sequence_rng)
            gap_sequence_results.append({
                'channel': channel, 'lag': lag, 'analysis': label,
                'pairs': len(pairs), 'days': days, 'correlation': r,
                'lower_95': lower, 'upper_95': upper,
            })

gap_sequence_table = pd.DataFrame(gap_sequence_results)
print('\n' + '=' * 76)
print('CONSECUTIVE GAP CORRELATIONS — WHOLE OPERATING-DAY BOOTSTRAP')
print('=' * 76)
print(gap_sequence_table.round(3).to_string(index=False))
print('\nIntervals are pointwise, exploratory estimates, not simultaneous tests.')
print('Few sampled days can make bootstrap intervals unstable.')
print('Within-block centering is descriptive and can affect short sequences.')
print('Counter gaps are between announcement events; some contain several orders.')

# ------------------------------------------------------------
# 25.5 Plot adjacent gaps and correlations across lags
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for row, (channel, title) in enumerate([
    ('drive_thru', 'Drive-thru completions'),
    ('counter', 'Counter announcement events'),
]):
    pairs = gap_lag_one_pairs[channel]
    ax = axes[row, 0]
    ax.scatter(pairs['previous_gap'], pairs['gap_seconds'], s=16, alpha=0.35)
    ax.set(xlabel='Previous gap (seconds)', ylabel='Next gap (seconds)')
    ax.set_title(f'{title}: consecutive gaps', fontfamily='Archivo Black')
    ax.text(0.03, 0.97, f'n = {len(pairs):,} neighboring pairs', transform=ax.transAxes, va='top')
    ax.grid(alpha=0.3)
    ax = axes[row, 1]
    for label in ['Pooled', 'Within block']:
        table = gap_sequence_table.loc[
            gap_sequence_table['channel'].eq(channel)
            & gap_sequence_table['analysis'].eq(label)
        ]
        line, = ax.plot(table['lag'], table['correlation'], marker='o', label=label)
        ax.fill_between(table['lag'], table['lower_95'], table['upper_95'],
                        color=line.get_color(), alpha=0.15)
    ax.axhline(0, linestyle=':', linewidth=1)
    ax.set(xlabel='Lag (number of gaps)', ylabel='Gap correlation',
           xticks=range(1, 11), ylim=(-1, 1))
    ax.set_title(f'{title}: sequence dependence', fontfamily='Archivo Black')
    ax.legend()
    ax.grid(alpha=0.3)
fig.text(0.5, 0.015,
         'Shading: pointwise 95% whole-day bootstrap intervals. '
         'Pairs never cross observation breaks or excluded gaps.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(IMAGE_DIR / 'consecutive_gap_dependence.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------
# 25.6 Save the numeric diagnostic table
# ------------------------------------------------------------
gap_sequence_table.to_csv(CSV_DIR / 'consecutive_gap_correlations.csv', index=False)

# ============================================================
# 26. DO FAST COMPLETIONS PRECEDE ADVANCE PAUSES?
# ============================================================
# Uses corrected movement markers and monitored blocks from Section 23.
# This describes recorded pause onset, not proven payment delay.

# ------------------------------------------------------------
# 26.1 Identify markers in each actual completion interval
# ------------------------------------------------------------
burst_events = prepare_gap_sequence(drive_segment_events, 'drive_thru')
burst_events = burst_events.loc[burst_events['segment_id'].isin(monitored_blocks)].copy()
burst_events['pause_onset'] = False
burst_events['any_movement_marker'] = False
for block_id, block in burst_events.groupby('segment_id'):
    markers = movement_markers.loc[movement_markers['movement_block'].eq(block_id)]
    for row_id, event in block.iterrows():
        inside = markers['timestamp'].gt(event['previous_completion']) & (
            markers['timestamp'].le(event['timestamp']))
        burst_events.loc[row_id, 'any_movement_marker'] = bool(inside.any())
        burst_events.loc[row_id, 'pause_onset'] = bool(
            (inside & markers['kind'].eq('not_moving_up')).any())

# ------------------------------------------------------------
# 26.2 Use the preceding three gaps, excluding earlier pauses
# ------------------------------------------------------------
# Current gap is the outcome interval. Its duration is never a predictor.
# Keeping the original chains prevents excluded intervals becoming neighbors.
burst_groups = burst_events.groupby(['segment_id', 'chain'], sort=False)
for lag in (1, 2, 3):
    burst_events[f'prior_gap_{lag}'] = burst_groups['gap_seconds'].shift(lag)
    burst_events[f'prior_marker_{lag}'] = burst_groups['any_movement_marker'].shift(lag)
    burst_events[f'prior_day_{lag}'] = burst_groups['operating_day'].shift(lag)
prior_gap_columns = [f'prior_gap_{lag}' for lag in (1, 2, 3)]
prior_marker_columns = [f'prior_marker_{lag}' for lag in (1, 2, 3)]
eligible = burst_events[prior_gap_columns].notna().all(axis=1)
eligible &= burst_events[prior_marker_columns].eq(False).all(axis=1)
for lag in (1, 2, 3):
    eligible &= burst_events[f'prior_day_{lag}'].eq(burst_events['operating_day'])
burst_events['prior_three_mean_seconds'] = burst_events[prior_gap_columns].mean(axis=1)

# ------------------------------------------------------------
# 26.3 Compare speed relative to the same observation block
# ------------------------------------------------------------
# Reference uses all positive, reliable unmarked gaps in that block.
# It is descriptive, not an out-of-sample forecast.
block_reference = burst_events.loc[
    ~burst_events['any_movement_marker']
].groupby('segment_id')['gap_seconds'].agg(['median', 'count'])
burst_events['block_reference_seconds'] = burst_events['segment_id'].map(block_reference['median'])
burst_events['block_reference_n'] = burst_events['segment_id'].map(block_reference['count'])
eligible &= burst_events['block_reference_n'].ge(10)
burst_candidates = burst_events.loc[eligible].copy()
burst_candidates['relative_prior_speed'] = (
    burst_candidates['prior_three_mean_seconds']
    / burst_candidates['block_reference_seconds'])

# ------------------------------------------------------------
# 26.4 Divide eligible sequences into four speed groups
# ------------------------------------------------------------
# Quantile ties stay together; do not invent distinct groups for tied values.
if len(burst_candidates) < 20:
    raise ValueError('Section 26 needs at least 20 eligible three-gap sequences.')
burst_candidates['speed_group'], burst_edges = pd.qcut(
    burst_candidates['relative_prior_speed'], q=4,
    labels=False, retbins=True, duplicates='drop')
burst_group_count = len(burst_edges) - 1
burst_summary_rows = []
burst_rng = np.random.default_rng(2606)
for group_id in range(burst_group_count):
    group = burst_candidates.loc[burst_candidates['speed_group'].eq(group_id)]
    # Include zero-exposure group/day cells so all monitored candidate days
    # participate in each resample, rather than sampling only days in a bin.
    all_days = sorted(burst_candidates['operating_day'].unique())
    daily = group.groupby('operating_day')['pause_onset'].agg(['size', 'sum'])
    daily = daily.reindex(all_days, fill_value=0).to_numpy(dtype=float)
    draws = burst_rng.integers(0, len(daily), size=(2000, len(daily)))
    totals = daily[draws].sum(axis=1)
    rates = np.divide(totals[:, 1], totals[:, 0],
                      out=np.full(len(totals), np.nan), where=totals[:, 0] > 0)
    finite_rates = rates[np.isfinite(rates)]
    lower, upper = (np.quantile(finite_rates, [0.025, 0.975])
                    if group['operating_day'].nunique() >= 3
                    and len(finite_rates) >= 1800 else (np.nan, np.nan))
    burst_summary_rows.append({
        'speed_group': group_id + 1,
        'relative_speed_lower': burst_edges[group_id],
        'relative_speed_upper': burst_edges[group_id + 1],
        'sequences': len(group), 'days': group['operating_day'].nunique(),
        'pause_onsets': int(group['pause_onset'].sum()),
        'pause_fraction': group['pause_onset'].mean(),
        'lower_95': lower, 'upper_95': upper,
    })
burst_summary = pd.DataFrame(burst_summary_rows)

# ------------------------------------------------------------
# 26.5 Print results and save the event-level audit
# ------------------------------------------------------------
print('\n' + '=' * 76)
print('FAST COMPLETIONS BEFORE RECORDED ADVANCE PAUSES')
print('=' * 76)
print(f'Eligible three-gap sequences: {len(burst_candidates):,}')
print(f'Sequences followed by a recorded pause onset: {burst_candidates.pause_onset.sum():,}')
print(burst_summary.round(3).to_string(index=False))
print('Group 1 = fastest relative to its block; highest group = slowest.')
print('Fraction = probability of a recorded onset in the next completion interval,')
print('not a pause rate per minute. Sequences overlap and are resampled by day.')
print('This excludes earlier marked pauses and blocks with fewer than 10 reference gaps.')
print('Associations do not establish payment delay or lost capacity.')
burst_summary.to_csv(CSV_DIR / 'burst_before_pause_summary.csv', index=False)
burst_candidates.to_csv(CSV_DIR / 'burst_before_pause_audit.csv', index=False)

# ------------------------------------------------------------
# 26.6 Plot preceding-gap distributions and pause fractions
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for outcome, label in [(False, 'No recorded pause onset'),
                       (True, 'Recorded pause onset')]:
    values = np.sort(burst_candidates.loc[
        burst_candidates['pause_onset'].eq(outcome), 'relative_prior_speed'
    ].to_numpy())
    if len(values):
        axes[0].step(values, np.arange(1, len(values)+1)/len(values),
                     where='post', label=f'{label} (n={len(values)})')
axes[0].set(xlabel='Mean of preceding 3 gaps / block median gap',
            ylabel='Fraction at or below this value', ylim=(0, 1))
axes[0].set_title('Completion pace before advance pauses', fontfamily='Archivo Black')
axes[0].legend()
axes[0].grid(alpha=0.3)
x = burst_summary['speed_group'].to_numpy()
axes[1].plot(x, 100*burst_summary['pause_fraction'], marker='o')
axes[1].vlines(x, 100*burst_summary['lower_95'], 100*burst_summary['upper_95'])
for record in burst_summary.itertuples():
    axes[1].annotate(f'n={record.sequences}',
                     (record.speed_group, 100*record.pause_fraction),
                     xytext=(0, 12), textcoords='offset points', ha='center')
axes[1].set(xlabel='Preceding pace: fastest group → slowest group',
            ylabel='Next interval with recorded pause onset (%)', xticks=x)
axes[1].set_ylim(bottom=0)
axes[1].set_title('Do pauses follow faster completions?', fontfamily='Archivo Black')
axes[1].grid(alpha=0.3)
fig.text(0.5, 0.015, 'Intervals: pointwise 95% whole-operating-day bootstrap. '
         'Only blocks with movement monitoring are included.', ha='center', fontsize=10)
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(IMAGE_DIR / 'burst_before_pause.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# 27. COUNTER ANNOUNCEMENT BATCHES AND SHORT BURSTS
# ============================================================
# Logged announcements are the observations; several corrected orders may
# share a timestamp. Do not invent separate pickup times for those orders.

# ------------------------------------------------------------
# 27.1 Prepare corrected positive-count announcement events
# ------------------------------------------------------------
counter_burst_events = counter_segment_events.sort_values(['segment_id', 'timestamp']).copy()
if not counter_burst_events['corrected_count'].gt(0).all():
    raise ValueError('Counter burst input contains nonpositive order counts.')
announcement_sizes = counter_burst_events['corrected_count'].value_counts().sort_index()
announcement_summary = announcement_sizes.rename('announcements').reset_index()
announcement_summary.columns = ['orders_in_announcement', 'announcements']
announcement_summary['orders'] = (
    announcement_summary['orders_in_announcement'] * announcement_summary['announcements'])

# ------------------------------------------------------------
# 27.2 Group announcements within a bounded total time span
# ------------------------------------------------------------
def build_counter_bursts(events, span_seconds):
    records = []
    for segment_id, block in events.groupby('segment_id', sort=False):
        current = None
        for event in block.itertuples():
            if current is None or (
                event.timestamp - current['start']).total_seconds() > span_seconds:
                if current is not None:
                    records.append(current)
                current = {
                    'segment_id': segment_id, 'start': event.timestamp,
                    'end': event.timestamp, 'orders': float(event.corrected_count),
                    'announcements': 1,
                }
            else:
                current['end'] = event.timestamp
                current['orders'] += float(event.corrected_count)
                current['announcements'] += 1
        if current is not None:
            records.append(current)
    bursts = pd.DataFrame(records)
    bursts['span_seconds'] = (bursts['end'] - bursts['start']).dt.total_seconds()
    bursts['previous_burst_end'] = bursts.groupby('segment_id')['end'].shift()
    bursts['preceding_quiet_seconds'] = (
        bursts['start'] - bursts['previous_burst_end']).dt.total_seconds()
    bursts['operating_day'] = (bursts['start'] - pd.Timedelta(hours=2)).dt.date
    # A first burst has an unknown preceding gap; never assign it zero.
    bursts['size_group'] = np.where(bursts['orders'].ge(3), '3+ orders',
                           np.where(bursts['orders'].eq(2), '2 orders', '1 order'))
    return bursts

# ------------------------------------------------------------
# 27.3 Check sensitivity to 5-, 10-, and 15-second spans
# ------------------------------------------------------------
counter_burst_samples = {}
counter_burst_sensitivity_rows = []
for threshold in (5, 10, 15):
    bursts = build_counter_bursts(counter_burst_events, threshold)
    if not np.isclose(bursts['orders'].sum(), counter_burst_events['corrected_count'].sum()):
        raise ValueError('Burst grouping changed the corrected order total.')
    if bursts['span_seconds'].gt(threshold).any():
        raise ValueError('Burst exceeds its specified total span.')
    counter_burst_samples[threshold] = bursts
    counter_burst_sensitivity_rows.append({
        'maximum_span_seconds': threshold,
        'burst_groups': len(bursts),
        'groups_with_2plus_orders': int(bursts['orders'].ge(2).sum()),
        'groups_with_3plus_orders': int(bursts['orders'].ge(3).sum()),
        'fraction_orders_in_2plus_groups': (
            bursts.loc[bursts['orders'].ge(2), 'orders'].sum() / bursts['orders'].sum()),
        'groups_with_multiple_announcements': int(bursts['announcements'].ge(2).sum()),
        'corrected_orders': bursts['orders'].sum(),
    })
counter_burst_sensitivity = pd.DataFrame(counter_burst_sensitivity_rows)
counter_bursts = counter_burst_samples[10].copy()

# ------------------------------------------------------------
# 27.4 Print distributions and save audit tables
# ------------------------------------------------------------
print('\n' + '=' * 76)
print('COUNTER ANNOUNCEMENT BATCHES AND SHORT BURSTS')
print('=' * 76)
print('\nOrders within one logged announcement:')
print(announcement_summary.to_string(index=False))
print('\nSensitivity to the maximum TOTAL burst span:')
print(counter_burst_sensitivity.round(3).to_string(index=False))
print('\n10-second burst-size distribution:')
print(counter_bursts['orders'].value_counts().sort_index().to_string())
print('Grouping is anchored at the first announcement, not chained indefinitely.')
print('First bursts in observation blocks have unknown preceding gaps.')
print('These are recorded announcement patterns, not individual food-ready times.')
print('Quiet-gap comparisons pool operating conditions and are descriptive.')
announcement_summary.to_csv(CSV_DIR / 'counter_announcement_sizes.csv', index=False)
counter_burst_sensitivity.to_csv(CSV_DIR / 'counter_burst_sensitivity.csv', index=False)
for threshold, bursts in counter_burst_samples.items():
    bursts.to_csv(CSV_DIR / f'counter_burst_audit_{threshold}s.csv', index=False)

# ------------------------------------------------------------
# 27.5 Plot announcement sizes and grouped burst sizes
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(announcement_summary['orders_in_announcement'], announcement_summary['announcements'])
axes[0].set(xlabel='Corrected orders in one announcement', ylabel='Announcement count')
axes[0].set_title('Orders announced together', fontfamily='Archivo Black')
for threshold, bursts in counter_burst_samples.items():
    size_counts = bursts['orders'].value_counts().sort_index()
    axes[1].plot(size_counts.index, size_counts / len(bursts), marker='o',
                 label=f'{threshold}-second maximum span')
axes[1].set(xlabel='Corrected orders in burst group', ylabel='Fraction of burst groups')
axes[1].set_title('Counter burst-size distributions', fontfamily='Archivo Black')
axes[1].legend()
for ax in axes:
    ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / 'counter_burst_sizes.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------
# 27.6 Compare preceding quiet-gap distributions by burst size
# ------------------------------------------------------------
quiet_gap_summary = counter_bursts.dropna(subset=['preceding_quiet_seconds']).groupby(
    'size_group')['preceding_quiet_seconds'].agg(
        n='size', median='median', q25=lambda x:x.quantile(.25),
        q75=lambda x:x.quantile(.75))
quiet_gap_summary.to_csv(CSV_DIR / 'counter_quiet_gap_summary.csv')
print('Preceding quiet gaps by burst size (seconds):')
print(quiet_gap_summary.to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
known_gaps = counter_bursts.dropna(subset=['preceding_quiet_seconds'])
for label in ('1 order', '2 orders', '3+ orders'):
    group = known_gaps.loc[known_gaps['size_group'].eq(label)]
    values = np.sort(group['preceding_quiet_seconds'].to_numpy())
    if len(values):
        axes[0].step(values, np.arange(1, len(values)+1)/len(values),
                     where='post', label=f'{label} (n={len(values)})')
axes[0].set(xlabel='Quiet gap before burst (seconds)',
            ylabel='Fraction at or below this gap', ylim=(0, 1))
axes[0].set_title('Quiet gaps before counter bursts', fontfamily='Archivo Black')
axes[0].legend()
axes[1].scatter(known_gaps['preceding_quiet_seconds'], known_gaps['orders'], alpha=0.35, s=22)
axes[1].set(xlabel='Quiet gap before burst (seconds)', ylabel='Orders in burst')
axes[1].set_title('Does a longer gap precede more orders?', fontfamily='Archivo Black')
for ax in axes:
    ax.grid(alpha=0.3)
fig.text(0.5, 0.015, '10-second maximum burst span. '
         'Quiet gap runs from the preceding burst end to this burst start.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(IMAGE_DIR / 'counter_quiet_gap_vs_burst.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# 28. REVIEW LINE-LENGTH AND REGISTER OBSERVATIONS
# ============================================================
# Extract candidates for review; these suggestions are NOT model inputs.
# A note can refer to cars, customers, groups, seating, or a past condition.

# ------------------------------------------------------------
# 28.1 Collect candidate notes without changing their wording
# ------------------------------------------------------------
line_notes = analysis_events.loc[analysis_events['kind'].eq('note')].copy()
# Historical text schemas and structured JSON may use different text fields.
line_notes['original_note'] = ''
for text_column in ('text', 'note', 'message'):
    if text_column in line_notes.columns:
        available_text = line_notes[text_column].fillna('').astype(str)
        use_text = line_notes['original_note'].eq('') & available_text.ne('')
        line_notes.loc[use_text, 'original_note'] = available_text.loc[use_text]
line_text = line_notes['original_note'].str.lower()
line_candidate = line_text.str.contains(
    r'\bline\b|\bqueue\b|\breg(?:isters?)?\b|\bteg\b|\bcashier\b|\bat cash\b',
    regex=True, na=False)
line_notes = line_notes.loc[line_candidate].copy()
line_text = line_notes['original_note'].str.lower()

# ------------------------------------------------------------
# 28.2 Suggest numbers and flag context that needs judgment
# ------------------------------------------------------------
# Broad extraction deliberately retains ranges, lower bounds and mixed notes.
line_notes['suggested_number_before_line'] = line_text.str.extract(
    r'(\d+(?:\s*[-–]\s*\d+)?\+?)\s+(?:in\s+)?(?:inside\s+)?line\b', expand=False)
line_notes['suggested_number_after_line'] = line_text.str.extract(
    r'\bline\s+(?:is\s+|about\s+|like\s+|down to\s+)?(\d+(?:\s*[-–]\s*\d+)?\+?)',
    expand=False)
line_notes['suggested_register_count'] = line_text.str.extract(
    r'(\d+)\s*(?:reg(?:isters?)?|teg)\b', expand=False)
line_notes['mentions_empty_line'] = line_text.str.contains(
    r'no (?:one in |inside )?line|line (?:gone|zero)|line.*\b0\b', regex=True)
line_notes['mentions_drive_thru'] = line_text.str.contains(
    r'drive|\bdtl\b|\bdt\b|\bcars?\b', regex=True)
line_notes['mentions_inside'] = line_text.str.contains(
    r'inside|counter|register|\breg\b|cashier', regex=True)
line_notes['mentions_past_or_change'] = line_text.str.contains(
    r'\bwas\b|earlier|ago|before|went|now|then|since|for awhile', regex=True)
line_notes['review_status'] = 'UNREVIEWED'
line_notes['reviewed_channel'] = ''
line_notes['reviewed_line_lower'] = np.nan
line_notes['reviewed_line_upper'] = np.nan
line_notes['reviewed_line_unit'] = ''
line_notes['reviewed_register_count'] = np.nan
line_notes['review_comment'] = ''

# ------------------------------------------------------------
# 28.3 Locate notes within counter observation windows
# ------------------------------------------------------------
# Being inside a counter window does not prove a note describes its line.
line_notes['counter_segment_id'] = pd.NA
for window in counter_windows.itertuples():
    inside = line_notes['timestamp'].between(window.effective_start, window.effective_end)
    if line_notes.loc[inside, 'counter_segment_id'].notna().any():
        raise ValueError('Line note matches overlapping counter windows.')
    line_notes.loc[inside, 'counter_segment_id'] = window.segment_id
line_notes = line_notes.sort_values('timestamp')

# ------------------------------------------------------------
# 28.4 Export the review sheet and print coverage
# ------------------------------------------------------------
review_columns = [
    'source_row', 'seq', 'timestamp', 'mode', 'original_note',
    'suggested_number_before_line', 'suggested_number_after_line',
    'suggested_register_count', 'mentions_empty_line', 'mentions_drive_thru',
    'mentions_inside', 'mentions_past_or_change', 'counter_segment_id',
    'review_status', 'reviewed_channel', 'reviewed_line_lower',
    'reviewed_line_upper', 'reviewed_line_unit', 'reviewed_register_count',
    'review_comment',
]
line_notes[review_columns].to_csv(CSV_DIR / 'line_register_notes_review.csv', index=False)
print('\n' + '=' * 76)
print('LINE-LENGTH AND REGISTER NOTES — REVIEW INVENTORY')
print('=' * 76)
print(f'Candidate notes: {len(line_notes):,}')
print(f'Candidates inside counter windows: {line_notes.counter_segment_id.notna().sum():,}')
print(f'Counter blocks containing candidates: {line_notes.counter_segment_id.nunique()}')
print(f'Calendar dates represented: {line_notes.timestamp.dt.date.nunique()}')
print('Numeric suggestions and empty-line flags require reading the full note.')
print('Distinguish customers from ordering groups and people at the register.')
print('No line values have been carried forward or used in an analysis.')
print('Saved: csv/line_register_notes_review.csv')

# ============================================================
# 29. COUNTER BURSTS AFTER LINE-LENGTH OBSERVATIONS
# ============================================================
# Reviewed classifications are tied to source, sequence and exact note text.
# This compares later announcements with earlier snapshots, not measured
# order-placement times or a continuously observed ordering queue.

# ------------------------------------------------------------
# 29.1 Include the reviewed numeric and explicit empty-line notes
# ------------------------------------------------------------
# Tuple fields: source_row, seq, original_note, category, count convention.
reviewed_line_records = [
    (0, 2, 'Slow night no one in line.', 'empty', 'explicit_empty'),
    (0, 6, 'No one in line', 'empty', 'explicit_empty'),
    (2, 35, 'When I got here a few minutes ago no line. Now 6 in line', '4-7', 'reported_value'),
    (2, 40, 'Line gone', 'empty', 'explicit_empty'),
    (2, 41, '1 in line', '1-3', 'reported_value'),
    (2, 42, '2 in line - lull in I side orders likely demand driven from lull in orders a bit ago.', '1-3', 'reported_value'),
    (2, 52, 'Line 7', '4-7', 'reported_value'),
    (4, 11, 'No one in line or manning registers they are doing a lot of clean up', 'empty', 'explicit_empty'),
    (4, 67, 'One at cash one person waiting behind', '1-3', 'waiting_behind_registers'),
    (4, 91, 'No line.', 'empty', 'explicit_empty'),
    (4, 98, 'Burst of orders coming out from line earlier. No line now inside', 'empty', 'explicit_empty'),
    (4, 144, '10 in line', '8+', 'reported_value'),
    (4, 217, 'Line about 4', '4-7', 'reported_value'),
    (4, 230, '2 teg open 8 in line', '8+', 'reported_value'),
    (4, 236, '3 reg open 12 in line', '8+', 'reported_value'),
    (4, 250, 'No line. Huge surge got handled.', 'empty', 'explicit_empty'),
    (4, 255, 'Expect a surge of orders after big line orders gets processed. No line now', 'empty', 'explicit_empty'),
    (4, 265, '3 reg up still many in line. 6 behind the groups at regs', '4-7', 'waiting_behind_registers'),
    (4, 271, '6 in line with 3 reg open', '4-7', 'reported_value'),
    (4, 396, 'No inside line', 'empty', 'explicit_empty'),
    (4, 549, 'No line inside', 'empty', 'explicit_empty'),
    (4, 603, 'No line', 'empty', 'explicit_empty'),
    (4, 759, '2 reg open . 9 in line inc at reg', '8+', 'includes_at_register'),
    (4, 763, '8 in line but 3 are new', '8+', 'reported_value'),
    (4, 767, '10 in line', '8+', 'reported_value'),
    (4, 778, 'No inside line', 'empty', 'explicit_empty'),
    (4, 844, 'Line went from moderate to over 12. 3 reg open. Inside line e is bursary', '8+', 'reported_lower_bound'),
    (4, 886, 'Large scale snide line. 2 reg. 10+ in line', '8+', 'reported_lower_bound'),
    (4, 896, '13+ in line.', '8+', 'reported_lower_bound'),
    (4, 911, '9 in line. Line has not died down', '8+', 'reported_value'),
    (4, 923, '12 in line was longer', '8+', 'reported_value'),
    (4, 1039, '6 in line was zero not long ago', '4-7', 'reported_value'),
    (4, 1051, 'No line for awhile inside. Mostly empty inside now', 'empty', 'explicit_empty'),
    (4, 1059, 'A line inside about 3-5 people steady now', 'range_crosses_categories', 'reported_range'),
    (4, 1062, 'No line', 'empty', 'explicit_empty'),
    (4, 1064, 'No inside line', 'empty', 'explicit_empty'),
    (4, 1072, 'Inside over half full - line about 6', '4-7', 'reported_value'),
    (4, 1131, 'Inside less than half full no line', 'empty', 'explicit_empty'),
    (4, 1222, '11 tables occupied inside. Line 2', '1-3', 'reported_value'),
    (4, 1392, 'Only 2 in line inside', '1-3', 'reported_value'),
    (4, 1400, '2 in line at inside', '1-3', 'reported_value'),
    (4, 1407, '8 in line inside.', '8+', 'reported_value'),
    (4, 1452, 'No line inside but dining area pretty full', 'empty', 'explicit_empty'),
    (4, 1456, 'Small line 2-3', '1-3', 'reported_range'),
    (4, 1467, 'No I side line', 'empty', 'explicit_empty'),
    (4, 1479, 'Inside line has 6', '4-7', 'reported_value'),
    (4, 1495, 'No inside line. Dining area emptying out slowly about half full', 'empty', 'explicit_empty'),
    (4, 1497, 'DTL 2-3 cars from full while inside has 8 in line', '8+', 'reported_value'),
    (4, 1509, 'Inside line 6 with two reg', '4-7', 'reported_value'),
    (4, 1524, 'Inside line 4', '4-7', 'reported_value'),
    (4, 1545, 'Inside line 5', '4-7', 'reported_value'),
    (4, 1556, 'No inside line.', 'empty', 'explicit_empty'),
    (4, 1605, 'Inside line about 6', '4-7', 'reported_value'),
    (4, 1617, 'Line around 6 inside', '4-7', 'reported_value'),
    (4, 1623, 'Inside line about 8', '8+', 'reported_value'),
    (4, 1652, 'Line is 13+ inside', '8+', 'reported_lower_bound'),
    (4, 1677, 'No inside line', 'empty', 'explicit_empty'),
    (4, 1683, 'No inside line', 'empty', 'explicit_empty'),
    (4, 1739, '8 in line', '8+', 'reported_value'),
    (4, 1787, 'Inside line is at least 10. 2 reg open', '8+', 'reported_lower_bound'),
    (4, 1804, '10-11+ in line and not dying down wth 3 reg open', '8+', 'reported_lower_bound'),
    (4, 1910, 'Half the tables empty now but inside line about 7', '4-7', 'reported_value'),
    (4, 1922, 'Was no inside line for awhile then a burst back to no inside line. DTL full', 'empty', 'explicit_empty'),
    (4, 1929, 'No line inside. . DTL full.', 'empty', 'explicit_empty'),
    (4, 1939, '2 in line inside. Not huge not empty. Inside dining area just under half full', '1-3', 'reported_value'),
    (4, 1941, 'Inside line 6', '4-7', 'reported_value'),
    (4, 1967, 'Small line like 2 inside. It’s not been over 3 for most of the observed window. And 0 much of the time observing', '1-3', 'reported_value'),
    (4, 1978, 'No inside line for awhile', 'empty', 'explicit_empty'),
    (4, 1983, 'One group at reg of 3 people no line.', 'empty', 'waiting_behind_registers'),
    (4, 1992, 'Inside has a line out of nowhere 9!!', '8+', 'reported_value'),
    (4, 1997, 'Line down to 2', '1-3', 'reported_value'),
    (4, 1998, 'Inside line is 3', '1-3', 'reported_value'),
    (4, 2017, 'No inside line', 'empty', 'explicit_empty'),
    (4, 2030, '1 inside line', '1-3', 'reported_value'),
    (4, 2037, 'No inside line', 'empty', 'explicit_empty'),
    (4, 2050, 'Inside line was zero for awhile now 5', '4-7', 'reported_value'),
    (4, 2065, 'No inside line', 'empty', 'explicit_empty'),
    (4, 2097, 'Big inside group now. Line is like 8', '8+', 'reported_value'),
    (4, 2229, 'No line', 'empty', 'explicit_empty'),
    (4, 2252, '7 inside line', '4-7', 'reported_value'),
    (4, 2307, 'Giant line inside out of nowhere. Over 10', '8+', 'reported_lower_bound'),
    (4, 2310, 'Inside line 8 2reg up', '8+', 'reported_value'),
    (4, 2426, 'Inside has small line 4-5', '4-7', 'reported_range'),
    (4, 2465, 'Inside line zero', 'empty', 'explicit_empty'),
    (4, 2520, 'Now 5 line', '4-7', 'reported_value'),
    (4, 2527, 'Line 7+', 'lower_bound', 'reported_lower_bound'),
    (4, 2540, '7 line, DTL full', '4-7', 'reported_value'),
    (4, 2549, 'Line 2', '1-3', 'reported_value'),
    (4, 2601, 'Inside line 3', '1-3', 'reported_value'),
    (4, 2619, 'Inside line 6', '4-7', 'reported_value'),
 ]
reviewed_line_lookup = pd.DataFrame(reviewed_line_records, columns=[
    'source_row', 'seq', 'expected_note', 'line_category', 'count_convention'])

# ------------------------------------------------------------
# 29.2 Validate notes and restrict the main comparison
# ------------------------------------------------------------
line_snapshot_audit = reviewed_line_lookup.merge(
    line_notes[['source_row', 'seq', 'timestamp', 'original_note',
                'counter_segment_id']],
    on=['source_row', 'seq'], how='left', validate='one_to_one')
if line_snapshot_audit['timestamp'].isna().any():
    raise ValueError('A reviewed line note is missing; check source identities.')
if not line_snapshot_audit['expected_note'].eq(
        line_snapshot_audit['original_note']).all():
    raise ValueError('Reviewed note text changed; inspect before using its classification.')
line_snapshot_audit['usable'] = (
    line_snapshot_audit['counter_segment_id'].notna()
    & line_snapshot_audit['line_category'].isin(['empty', '1-3', '4-7', '8+'])
    & ~line_snapshot_audit['count_convention'].isin(
        ['includes_at_register', 'waiting_behind_registers']))
line_snapshot_audit['exclusion_reason'] = np.select([
    line_snapshot_audit['counter_segment_id'].isna(),
    ~line_snapshot_audit['line_category'].isin(['empty', '1-3', '4-7', '8+']),
    line_snapshot_audit['count_convention'].isin(
        ['includes_at_register', 'waiting_behind_registers']),
], ['Outside counter observation window', 'Range crosses categories or lower bound insufficient',
    'Explicit register inclusion/exclusion convention differs'], default='')
line_snapshots = line_snapshot_audit.loc[line_snapshot_audit['usable']].copy()

# ------------------------------------------------------------
# 29.3 Match each burst to one recent eligible snapshot per lag band
# ------------------------------------------------------------
# Compare 0-5, 5-10 and 10-15 minutes after the note.
# Within each lag band, choose the most recent usable eligible note.
# A burst contributes at most once per band, though it can appear in other
# bands. An intervening note does not invalidate an EARLIER lagged snapshot:
# we are testing association with past conditions, not declaring them current.
line_lag_records = []
for burst_id, burst in counter_bursts.iterrows():
    same_block = line_snapshots.loc[line_snapshots['counter_segment_id'].eq(burst['segment_id'])]
    age_minutes = (burst['start'] - same_block['timestamp']).dt.total_seconds()/60
    for lower, upper in [(0, 5), (5, 10), (10, 15)]:
        eligible = same_block.loc[age_minutes.gt(lower) & age_minutes.le(upper)]
        if eligible.empty:
            continue
        note = eligible.sort_values('timestamp').iloc[-1]
        line_lag_records.append({
            'burst_id': burst_id, 'segment_id': burst['segment_id'],
            'burst_start': burst['start'], 'orders': burst['orders'],
            'operating_day': burst['operating_day'],
            'lag_band': f'{lower}-{upper} min', 'line_category': note['line_category'],
            'note_source_row': note['source_row'], 'note_seq': note['seq'],
            'note_timestamp': note['timestamp'],
            'note_age_minutes': (burst['start'] - note['timestamp']).total_seconds()/60,
        })
line_burst_matches = pd.DataFrame(line_lag_records)
if line_burst_matches.empty:
    raise ValueError('No line snapshots match subsequent counter bursts.')
if line_burst_matches.duplicated(['burst_id', 'lag_band']).any():
    raise ValueError('Duplicate burst matches within one lag band.')

# ------------------------------------------------------------
# 29.4 Summarize burst distributions with whole-day uncertainty
# ------------------------------------------------------------
line_burst_rng = np.random.default_rng(2906)
line_burst_summary_rows = []
for lag_band in ['0-5 min', '5-10 min', '10-15 min']:
    band = line_burst_matches.loc[line_burst_matches['lag_band'].eq(lag_band)]
    all_days = sorted(band['operating_day'].unique())
    for category in ['empty', '1-3', '4-7', '8+']:
        group = band.loc[band['line_category'].eq(category)].copy()
        if group.empty:
            continue
        group['multiple'] = group['orders'].ge(2).astype(int)
        daily = group.groupby('operating_day')['multiple'].agg(['size', 'sum'])
        daily = daily.reindex(all_days, fill_value=0).to_numpy(dtype=float)
        lower_95 = upper_95 = np.nan
        if group['operating_day'].nunique() >= 3:
            draws = line_burst_rng.integers(0, len(daily), size=(2000, len(daily)))
            totals = daily[draws].sum(axis=1)
            rates = np.divide(totals[:, 1], totals[:, 0],
                              out=np.full(len(totals), np.nan), where=totals[:, 0]>0)
            rates = rates[np.isfinite(rates)]
            if len(rates) >= 1800:
                lower_95, upper_95 = np.quantile(rates, [0.025, 0.975])
        line_burst_summary_rows.append({
            'lag_band': lag_band, 'line_category': category,
            'burst_groups': len(group), 'days': group['operating_day'].nunique(),
            'blocks': group['segment_id'].nunique(),
            'distinct_notes': group[['note_source_row', 'note_seq']].drop_duplicates().shape[0],
            'fraction_1_order': group['orders'].eq(1).mean(),
            'fraction_2_orders': group['orders'].eq(2).mean(),
            'fraction_3plus_orders': group['orders'].ge(3).mean(),
            'fraction_2plus_orders': group['multiple'].mean(),
            'lower_95': lower_95, 'upper_95': upper_95,
        })
line_burst_summary = pd.DataFrame(line_burst_summary_rows)

# ------------------------------------------------------------
# 29.5 Print results and save reproducible audit tables
# ------------------------------------------------------------
print('\n' + '=' * 88)
print('COUNTER BURST DISTRIBUTIONS AFTER LINE SNAPSHOTS')
print('=' * 88)
print(f'Reviewed numeric/empty snapshots: {len(line_snapshot_audit)}')
print(f'Snapshots eligible for main comparison: {len(line_snapshots)}')
print(f'Distinct matched bursts: {line_burst_matches.burst_id.nunique()}')
print(line_burst_summary.round(3).to_string(index=False))
print('Line categories describe reported counts; people versus groups is often unspecified.')
print('Each burst occurs once per lag band. The same burst may occur in different bands.')
print('Intervals are pointwise whole-day bootstrap estimates; few days limit precision.')
print('Sparse snapshots and differences between blocks can explain apparent associations.')
print('These fractions describe burst groups, not the fraction of all orders in batches.')
line_snapshot_audit.to_csv(CSV_DIR / 'reviewed_line_snapshot_audit.csv', index=False)
line_burst_matches.to_csv(CSV_DIR / 'line_snapshot_burst_matches.csv', index=False)
line_burst_summary.to_csv(CSV_DIR / 'line_snapshot_burst_summary.csv', index=False)

# ------------------------------------------------------------
# 29.6 Plot complete burst-size distributions by line condition
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, lag_band in zip(axes, ['0-5 min', '5-10 min', '10-15 min']):
    band = line_burst_matches.loc[line_burst_matches['lag_band'].eq(lag_band)]
    for category in ['empty', '1-3', '4-7', '8+']:
        group = band.loc[band['line_category'].eq(category)]
        if group.empty:
            continue
        values = np.sort(group['orders'].to_numpy())
        # Survival at integer k: fraction of groups with at least k orders.
        sizes = np.arange(1, int(counter_bursts['orders'].max()) + 1)
        survival = [(values >= size).mean() for size in sizes]
        label = 'Empty line' if category == 'empty' else f'Line {category}'
        ax.plot(sizes, survival, marker='o', label=f'{label} (n={len(group)})')
    ax.set(xlabel='At least this many orders in burst', xticks=sizes, ylim=(0, 1.05))
    ax.set_title(f'{lag_band} after line note', fontfamily='Archivo Black')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_ylabel('Fraction of burst groups')
fig.text(0.5, 0.015, '10-second burst grouping. Line snapshots matched within the same '
         'counter block. Separate lag bands reuse some bursts.', ha='center', fontsize=10)
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(IMAGE_DIR / 'counter_bursts_after_line_snapshots.png', dpi=300, bbox_inches='tight')
plt.close(fig)
# ============================================================
# 30. COUNTER OUTPUT AFTER LINE SNAPSHOTS AND RECENT OUTPUT
# ============================================================
# The prior-hour rate uses only the SAME continuous counter block.
# Require at least 20 observed minutes; shorter histories stay unclassified.
# High/low recent output is a descriptive median split, not a capacity claim.

# ------------------------------------------------------------
# 30.1 Measure observed output before each eligible line snapshot
# ------------------------------------------------------------
def count_counter_orders(events, start, end):
    return float(events.loc[
        events['timestamp'].gt(start) & events['timestamp'].le(end),
        'corrected_count'].sum())

counter_window_lookup = counter_windows.set_index('segment_id')
line_output_snapshots = line_snapshots.copy()
line_output_snapshots['prior_minutes'] = np.nan
line_output_snapshots['prior_orders'] = np.nan
line_output_snapshots['prior_orders_per_hour'] = np.nan
for row_id, note in line_output_snapshots.iterrows():
    window = counter_window_lookup.loc[note['counter_segment_id']]
    begin = max(window['effective_start'], note['timestamp']-pd.Timedelta(hours=1))
    minutes = (note['timestamp']-begin).total_seconds()/60
    events = counter_segment_events.loc[
        counter_segment_events['segment_id'].eq(note['counter_segment_id'])]
    orders = count_counter_orders(events, begin, note['timestamp'])
    line_output_snapshots.loc[row_id, ['prior_minutes', 'prior_orders']] = [minutes, orders]
    if minutes >= 20:
        line_output_snapshots.loc[row_id, 'prior_orders_per_hour'] = 60*orders/minutes

# ------------------------------------------------------------
# 30.2 Define joint line-size and recent-output conditions
# ------------------------------------------------------------
recent_output_cutoff = line_output_snapshots['prior_orders_per_hour'].median()
if not np.isfinite(recent_output_cutoff):
    raise ValueError('No line snapshot has 20 minutes of preceding observation.')
line_output_snapshots['line_condition'] = np.where(
    line_output_snapshots['line_category'].eq('8+'), 'Line 8+', 'Line 0-7')
line_output_snapshots['recent_output_condition'] = np.where(
    line_output_snapshots['prior_orders_per_hour'].ge(recent_output_cutoff),
    'higher recent output', 'lower recent output')
line_output_snapshots.loc[
    line_output_snapshots['prior_orders_per_hour'].isna(),
    'recent_output_condition'] = 'insufficient history'
line_output_snapshots['condition'] = (
    line_output_snapshots['line_condition'] + '; '
    + line_output_snapshots['recent_output_condition'])
classified_snapshots = line_output_snapshots.loc[
    line_output_snapshots['prior_orders_per_hour'].notna()].copy()

# ------------------------------------------------------------
# 30.3 Partition overlapping follow-up intervals without double counting
# ------------------------------------------------------------
# Each lag band is a separate comparison. Within one band, partition the
# union of eligible follow-up time and assign overlaps to the latest note.
# A minute or completion belongs to at most ONE condition per lag band.
# Follow-up is clipped at the same observation block's end, never extended
# through an unobserved break. Intervals are (start, end].
line_output_records = []
for lower, upper in [(0, 15), (15, 25), (25, 35), (35, 45)]:
    band_name = f'{lower}-{upper} min'
    for segment_id, notes in classified_snapshots.groupby('counter_segment_id'):
        window = counter_window_lookup.loc[segment_id]
        events = counter_segment_events.loc[counter_segment_events['segment_id'].eq(segment_id)]
        intervals = []
        for note_id, note in notes.iterrows():
            start = max(note['timestamp']+pd.Timedelta(minutes=lower), window['effective_start'])
            end = min(note['timestamp']+pd.Timedelta(minutes=upper), window['effective_end'])
            if end > start:
                intervals.append((start, end, note_id))
        boundaries = sorted({time for start, end, note_id in intervals for time in (start, end)})
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            active = [note_id for left, right, note_id in intervals if left <= start and right >= end]
            if not active:
                continue
            note_id = max(active, key=lambda key: notes.loc[key, 'timestamp'])
            note = notes.loc[note_id]
            minutes = (end-start).total_seconds()/60
            line_output_records.append({
                'lag_band': band_name, 'segment_id': segment_id,
                'start': start, 'end': end, 'minutes': minutes,
                'orders': count_counter_orders(events, start, end),
                'condition': note['condition'], 'note_source_row': note['source_row'],
                'note_seq': note['seq'], 'note_timestamp': note['timestamp'],
                'prior_orders_per_hour': note['prior_orders_per_hour'],
                'baseline_expected_orders': minutes*note['prior_orders_per_hour']/60,
                'operating_day': (note['timestamp']-pd.Timedelta(hours=2)).date(),
            })
line_output_intervals = pd.DataFrame(line_output_records)
if line_output_intervals.empty:
    raise ValueError('No observed follow-up time after classified line snapshots.')

# ------------------------------------------------------------
# 30.4 Calculate rates and whole-day bootstrap intervals
# ------------------------------------------------------------
line_output_rng = np.random.default_rng(3006)
line_output_summary_rows = []
for (band, condition), group in line_output_intervals.groupby(['lag_band', 'condition'], sort=False):
    daily = group.groupby('operating_day')[['orders', 'minutes', 'baseline_expected_orders']].sum()
    # Resample whole days, including zero-contribution days in this band.
    all_days = sorted(line_output_intervals.loc[
        line_output_intervals['lag_band'].eq(band), 'operating_day'].unique())
    daily_array = daily.reindex(all_days, fill_value=0).to_numpy()
    lower_95 = upper_95 = change_lower = change_upper = np.nan
    if len(daily) >= 3:
        draws = line_output_rng.integers(0, len(all_days), size=(2000, len(all_days)))
        totals = daily_array[draws].sum(axis=1)
        valid = totals[:, 1] > 0
        if valid.sum() >= 1800:
            rates = 60*totals[valid, 0]/totals[valid, 1]
            changes = 60*(totals[valid, 0]-totals[valid, 2])/totals[valid, 1]
            lower_95, upper_95 = np.quantile(rates, [0.025, 0.975])
            change_lower, change_upper = np.quantile(changes, [0.025, 0.975])
    rate = 60*group['orders'].sum()/group['minutes'].sum()
    baseline = 60*group['baseline_expected_orders'].sum()/group['minutes'].sum()
    line_output_summary_rows.append({
        'lag_band': band, 'condition': condition,
        'days': len(daily), 'blocks': group['segment_id'].nunique(),
        'notes': group[['note_source_row', 'note_seq']].drop_duplicates().shape[0],
        'observed_minutes': group['minutes'].sum(), 'orders': group['orders'].sum(),
        'orders_per_hour': rate, 'lower_95': lower_95, 'upper_95': upper_95,
        'prior_rate_weighted': baseline, 'change_from_prior': rate-baseline,
        'change_lower_95': change_lower, 'change_upper_95': change_upper,
    })
line_output_summary = pd.DataFrame(line_output_summary_rows)

# ------------------------------------------------------------
# 30.5 Print coverage, results and limitations; save audits
# ------------------------------------------------------------
print('\n' + '='*100)
print('COUNTER OUTPUT AFTER LINE SNAPSHOTS, BY RECENT OUTPUT')
print('='*100)
print(f'Higher recent output threshold: {recent_output_cutoff:.1f} orders/observed hour')
print(f'Snapshots with >=20 minutes of preceding observation: {len(classified_snapshots)}')
print(f'Snapshots excluded for insufficient history: {len(line_output_snapshots)-len(classified_snapshots)}')
print(line_output_summary.round(2).to_string(index=False))
print('Within a lag band, follow-up exposure and orders are counted once.')
print('Different lag bands can reuse clock time; they are not independent experiments.')
print('Zero-order observed intervals are included; unobserved time is excluded.')
print('Partial follow-up contributes its actual duration; late bands can contain different notes.')
print('Pointwise day-bootstrap intervals are descriptive; sparse days limit precision.')
print('Higher/lower is a sample-relative split. Prior rates are estimated from limited exposure.')
print('Changes from prior output can include regression to the mean and changing conditions.')
print('Line snapshots are not order-placement times; no preparation time is estimated here.')
line_output_snapshots.to_csv(CSV_DIR / 'line_output_snapshot_audit.csv', index=False)
line_output_intervals.to_csv(CSV_DIR / 'line_output_interval_audit.csv', index=False)
line_output_summary.to_csv(CSV_DIR / 'line_output_summary.csv', index=False)

# ------------------------------------------------------------
# 30.6 Plot output rates and changes across delay bands
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
band_order = ['0-15 min', '15-25 min', '25-35 min', '35-45 min']
condition_order = [
    'Line 0-7; lower recent output', 'Line 0-7; higher recent output',
    'Line 8+; lower recent output', 'Line 8+; higher recent output']
for condition in condition_order:
    table = line_output_summary.loc[line_output_summary['condition'].eq(condition)]
    table = table.set_index('lag_band').reindex(band_order)
    for ax, value_column, low_column, high_column in [
        (axes[0], 'orders_per_hour', 'lower_95', 'upper_95'),
        (axes[1], 'change_from_prior', 'change_lower_95', 'change_upper_95')]:
        line, = ax.plot(range(4), table[value_column], marker='o', label=condition)
        ax.fill_between(range(4), table[low_column], table[high_column],
                        color=line.get_color(), alpha=0.12)
axes[0].set_title('Counter output after line observations', fontfamily='Archivo Black')
axes[0].set_ylabel('Orders per observed hour')
axes[0].set_ylim(bottom=0)
axes[1].set_title('Change from preceding output rate', fontfamily='Archivo Black')
axes[1].set_ylabel('Change in orders per observed hour')
axes[1].axhline(0, linestyle=':', linewidth=1)
for ax in axes:
    ax.set_xticks(range(4), band_order)
    ax.set_xlabel('Time after line snapshot')
    ax.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.045))
fig.text(0.5, 0.015, 'Shading: pointwise 95% whole-day bootstrap. '
         'Overlaps resolved within each lag band; observed zeros retained.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(IMAGE_DIR / 'counter_output_after_line_snapshots.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# 31. COUNTER ANNOUNCEMENT SPACING VS RANDOM TIMING
# ============================================================
# Reference: preserve each block's number of announcements and window,
# then place its announcements independently and uniformly in that window.
# This is a constant-rate Poisson timing reference conditional on the count.
# It tests announcement spacing, not orders within one announcement.

# ------------------------------------------------------------
# 31.1 Collect actual adjacent gaps inside observed counter blocks
# ------------------------------------------------------------
announcement_spacing_blocks = []
for window in counter_windows.itertuples():
    events = counter_segment_events.loc[
        counter_segment_events['segment_id'].eq(window.segment_id)
    ].sort_values('timestamp')
    duration = (window.effective_end-window.effective_start).total_seconds()
    if len(events) < 2 or duration <= 0:
        continue
    if not events['timestamp'].between(window.effective_start, window.effective_end).all():
        raise ValueError('Counter announcement lies outside its observation window.')
    observed_gaps = events['timestamp'].diff().dt.total_seconds().dropna().to_numpy()
    announcement_spacing_blocks.append({
        'segment_id': window.segment_id, 'duration_seconds': duration,
        'announcements': len(events), 'gaps': observed_gaps,
        'day': (window.effective_start-pd.Timedelta(hours=2)).date(),
    })
if not announcement_spacing_blocks:
    raise ValueError('No eligible counter blocks for announcement spacing.')

# ------------------------------------------------------------
# 31.2 Simulate timing while preserving counts and boundaries
# ------------------------------------------------------------
spacing_rng = np.random.default_rng(3106)
spacing_thresholds = np.array([5, 10, 15])
spacing_draws = 2000
spacing_observed_counts = np.zeros(3)
spacing_simulated_counts = np.zeros((spacing_draws, 3))
spacing_gap_total = 0
spacing_block_rows = []
for block in announcement_spacing_blocks:
    gaps = block['gaps']
    simulated_times = spacing_rng.uniform(
        0, block['duration_seconds'], size=(spacing_draws, block['announcements']))
    simulated_gaps = np.diff(np.sort(simulated_times, axis=1), axis=1)
    # Preserve whole-second logging resolution in the reference.
    simulated_times_rounded = np.rint(simulated_times)
    simulated_gaps_rounded = np.diff(np.sort(simulated_times_rounded, axis=1), axis=1)
    spacing_gap_total += len(gaps)
    for column, threshold in enumerate(spacing_thresholds):
        actual = int((gaps <= threshold).sum())
        simulated = (simulated_gaps_rounded <= threshold).sum(axis=1)
        spacing_observed_counts[column] += actual
        spacing_simulated_counts[:, column] += simulated
        spacing_block_rows.append({
            'segment_id': block['segment_id'], 'operating_day': block['day'],
            'observed_minutes': block['duration_seconds']/60,
            'announcements': block['announcements'], 'adjacent_gaps': len(gaps),
            'threshold_seconds': threshold, 'observed_short_gaps': actual,
            'reference_expected_short_gaps': simulated.mean(),
        })

# ------------------------------------------------------------
# 31.3 Print matched-denominator comparisons and save results
# ------------------------------------------------------------
spacing_rows = []
for column, threshold in enumerate(spacing_thresholds):
    fractions = spacing_simulated_counts[:, column]/spacing_gap_total
    lower, upper = np.quantile(fractions, [0.025, 0.975])
    spacing_rows.append({
        'threshold_seconds': threshold,
        'adjacent_gaps': spacing_gap_total,
        'observed_short_gaps': int(spacing_observed_counts[column]),
        'observed_fraction': spacing_observed_counts[column]/spacing_gap_total,
        'reference_mean_fraction': fractions.mean(),
        'reference_lower_95': lower, 'reference_upper_95': upper,
    })
announcement_spacing_summary = pd.DataFrame(spacing_rows)
announcement_spacing_audit = pd.DataFrame(spacing_block_rows)
print('\n'+'='*88)
print('COUNTER ANNOUNCEMENT SPACING VS CONDITIONAL RANDOM TIMING')
print('='*88)
print(announcement_spacing_summary.round(4).to_string(index=False))
print(f'Eligible blocks: {len(announcement_spacing_blocks)}')
print('Each comparison uses actual adjacent gaps, including same-second gaps.')
print('Reference preserves block counts, window boundaries and whole-second resolution.')
print('Reference envelopes are simulated variability under random timing, NOT confidence')
print('intervals for the observed fraction. Within-block rate changes are not modeled.')
print('A departure can reflect pacing, batching, changing rates or logging effects.')
print('No formal significance claim is made across these overlapping thresholds.')
announcement_spacing_summary.to_csv(
    CSV_DIR / 'counter_announcement_random_reference.csv', index=False)
announcement_spacing_audit.to_csv(CSV_DIR / 'counter_announcement_spacing_audit.csv', index=False)

# ------------------------------------------------------------
# 31.4 Plot observed fractions against the random reference
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
table = announcement_spacing_summary
ax.plot(table['threshold_seconds'], table['observed_fraction']*100,
        marker='o', label='Observed adjacent announcement gaps')
line, = ax.plot(table['threshold_seconds'], table['reference_mean_fraction']*100,
               marker='o', linestyle=':', label='Random timing within each block')
ax.fill_between(table['threshold_seconds'], table['reference_lower_95']*100,
                table['reference_upper_95']*100, color=line.get_color(), alpha=0.15,
                label='95% random-reference envelope')
ax.set(xlabel='Gap threshold (seconds)', ylabel='Adjacent gaps at or below threshold (%)',
       xticks=[5, 10, 15])
ax.set_ylim(bottom=0)
ax.set_title('How closely spaced are counter announcements?', fontfamily='Archivo Black')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGE_DIR / 'counter_announcement_random_reference.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# 32. FINAL AUDIT AND RECONCILIATION
# ============================================================
# 32.1 Export applied event corrections and unresolved decisions
counter_audit = counter_orders.loc[
    counter_orders['source_row'].eq(4) & counter_orders['seq'].isin(
        [1964,1971,1810,2320,940,1701]),
    ['source_row','seq','timestamp','numbers','numbers_clean','exclude_manual',
     'corrected_count','repeat_count']]
counter_audit.to_csv(CSV_DIR / 'counter_correction_audit.csv', index=False)
drive_thru_orders.loc[drive_thru_orders['exclude_manual']].to_csv(
    CSV_DIR / 'drive_exclusion_audit.csv', index=False)
observation_windows.to_csv(CSV_DIR / 'observation_windows.csv', index=False)
monitored_gaps.to_csv(CSV_DIR / 'movement_gap_audit.csv', index=False)
pause_completion_matches.to_csv(CSV_DIR / 'movement_episode_audit.csv', index=False)
pd.DataFrame(movement_audit, columns=['segment_id','timestamp','flag']).to_csv(
    CSV_DIR / 'movement_pairing_flags.csv', index=False)
pd.DataFrame([
    {'item':'Car 305', 'status':'Unresolved; retained',
     'reason':'Tracked car 11 vanished; closure may have generated this completion'},
    {'item':'Cars 296/297', 'status':'Retained',
     'reason':'Same timestamp alone does not establish an accidental duplicate'},
]).to_csv(CSV_DIR / 'unresolved_decisions.csv', index=False)

# 32.1.1 Save unreliable intervals separately from completion exclusions
drive_intervals.loc[unreliable_drive_gap(drive_intervals)].to_csv(
    CSV_DIR / 'drive_unreliable_gap_audit.csv', index=False)

# 32.2 Check positive exposure and event coverage
assert observation_windows['duration_minutes'].gt(0).all(), 'Nonpositive observation window'
for mode, frame in [('drive_thru', drive_thru_final),
                    ('counter', counter_orders.loc[counter_orders['corrected_count'].gt(0)])]:
    windows = observation_windows.loc[observation_windows['mode'].eq(mode)]
    covered = pd.Series(False, index=frame.index)
    for window in windows.itertuples():
        covered |= frame['timestamp'].between(window.effective_start, window.effective_end)
    assert covered.all(), f'{mode}: counted events outside observed time'
print('Audit passed: every counted completion has positive observation exposure.')
print(f'Images: {IMAGE_DIR}\nCSV tables: {CSV_DIR}')

# ============================================================
# 33. HOURLY UNCERTAINTY AND DAILY ORDER-OUTPUT SCENARIOS
# ============================================================
# Append after Section 32 in the audited burgers_clean.py.
# Uses existing imports, corrected events, exposure, palette and directories.
# These are projections from sampled hours, not observed full-day totals.

# ------------------------------------------------------------
# 33.1 Build matching order and exposure totals by operating day
# ------------------------------------------------------------
projection_events = []
for mode, frame in [('drive_thru', drive_orders_hourly),
                    ('counter', counter_orders_hourly)]:
    part = frame.loc[frame['operating_hour_bin'].between(10, 25)].copy()
    part['mode'] = mode
    part['operating_hour'] = part['operating_hour_bin']
    part['operating_day'] = (part['timestamp'] - pd.Timedelta(hours=2)).dt.date
    part['orders'] = 1 if mode == 'drive_thru' else part['corrected_count']
    projection_events.append(part[['mode', 'operating_day', 'operating_hour', 'orders']])
projection_orders = pd.concat(projection_events, ignore_index=True)
projection_exposure = hourly_exposure.groupby(
    ['operating_day', 'operating_hour', 'mode'])['observed_minutes'].sum()
projection_counts = projection_orders.groupby(
    ['operating_day', 'operating_hour', 'mode'])['orders'].sum()
projection_days = sorted(hourly_exposure['operating_day'].unique())
projection_cells = pd.MultiIndex.from_product(
    [range(10, 26), ['drive_thru', 'counter']], names=['operating_hour', 'mode'])

# ------------------------------------------------------------
# 33.2 Resample entire operating days, preserving both channels
# ------------------------------------------------------------
def bootstrap_hourly_output(exposure, counts, days, cells, draws=3000):
    """Return pooled rates and day-cluster bootstrap rates on a fixed grid."""
    minutes = exposure.unstack(['operating_hour', 'mode']).reindex(
        index=days, columns=cells, fill_value=0).fillna(0).to_numpy()
    orders = counts.unstack(['operating_hour', 'mode']).reindex(
        index=days, columns=cells, fill_value=0).fillna(0).to_numpy()
    if ((orders > 0) & (minutes == 0)).any():
        raise ValueError('Orders without matching day/hour exposure; inspect boundaries.')
    total_minutes = minutes.sum(axis=0)
    rates = np.divide(orders.sum(axis=0)*60, total_minutes,
                      out=np.full(len(cells), np.nan), where=total_minutes > 0)
    rng = np.random.default_rng(3306)
    weights = rng.multinomial(len(days), np.full(len(days), 1/len(days)), size=draws)
    sampled_minutes = weights @ minutes
    sampled_orders = weights @ orders
    sampled_rates = np.divide(sampled_orders*60, sampled_minutes,
                              out=np.full(sampled_minutes.shape, np.nan),
                              where=sampled_minutes > 0)
    return rates, sampled_rates, (minutes > 0).sum(axis=0), total_minutes

projection_rates, projection_draws, projection_day_counts, projection_minutes = (
    bootstrap_hourly_output(projection_exposure, projection_counts,
                            projection_days, projection_cells))

# ------------------------------------------------------------
# 33.3 Report hourly coverage and pointwise bootstrap intervals
# ------------------------------------------------------------
projection_hourly_rows = []
for column, (hour, mode) in enumerate(projection_cells):
    values = projection_draws[:, column]
    finite = values[np.isfinite(values)]
    usable_fraction = len(finite)/len(values)
    days = int(projection_day_counts[column])
    # Few-day bins remain preliminary. Suppress unstable/missing intervals.
    show_interval = days >= 3 and usable_fraction >= .90
    low, high = np.quantile(finite, [.025, .975]) if show_interval else (np.nan, np.nan)
    projection_hourly_rows.append({
        'operating_hour': hour, 'mode': mode,
        'orders_per_hour': projection_rates[column],
        'observed_minutes': projection_minutes[column], 'observed_days': days,
        'preliminary': days < 3, 'bootstrap_usable_fraction': usable_fraction,
        'bootstrap_low': low, 'bootstrap_high': high,
    })
projection_hourly = pd.DataFrame(projection_hourly_rows)
projection_hourly.to_csv(CSV_DIR / 'daily_projection_hourly_rates.csv', index=False)

# ------------------------------------------------------------
# 33.4 Integrate hourly rates under explicit opening/closing scenarios
# ------------------------------------------------------------
# 25 means 1 AM; 25.5 means 1:30 AM. These are illustrative schedules.
# Fractional hours assume a constant rate within that hour.
# No estimate is supplied for an unobserved hour.
projection_scenario_rows = []
for schedule, opening, closing in [
    ('10 AM to 1 AM', 10, 25), ('10 AM to 1:30 AM', 10, 25.5)]:
    for mode in ['drive_thru', 'counter', 'combined']:
        cell_weights = np.array([
            max(0, min(hour+1, closing)-max(hour, opening))
            if mode == 'combined' or channel == mode else 0
            for hour, channel in projection_cells])
        required = cell_weights > 0
        supported = required & np.isfinite(projection_rates)
        missing = required & ~np.isfinite(projection_rates)
        contribution = np.sum(projection_rates[supported]*cell_weights[supported])
        # Sum within each joint resample, rather than adding separate CIs.
        complete_draws = np.isfinite(projection_draws[:, required]).all(axis=1)
        complete_fraction = complete_draws.mean()
        full_estimate = contribution if not missing.any() else np.nan
        low = high = np.nan
        if not missing.any() and complete_fraction >= .90:
            totals = projection_draws[complete_draws][:, required] @ cell_weights[required]
            low, high = np.quantile(totals, [.025, .975])
        missing_labels = '; '.join(
            f'{channel} {operating_labels[hour-10]}'
            for (hour, channel), absent in zip(projection_cells, missing) if absent)
        projection_scenario_rows.append({
            'schedule': schedule, 'mode': mode,
            'supported_hour_contribution': contribution,
            'full_day_estimate': full_estimate,
            'bootstrap_low': low, 'bootstrap_high': high,
            'complete_bootstrap_fraction': complete_fraction,
            'preliminary_hour_cells': int((required & (projection_day_counts < 3)).sum()),
            'missing_hour_cells': missing_labels,
        })
projection_scenarios = pd.DataFrame(projection_scenario_rows)
projection_scenarios.to_csv(CSV_DIR / 'daily_order_output_scenarios.csv', index=False)

# ------------------------------------------------------------
# 33.5 Print what can and cannot be estimated from current coverage
# ------------------------------------------------------------
print('\n' + '='*90)
print('DAILY ORDER-OUTPUT SCENARIOS — CURRENT DATA ONLY')
print('='*90)
print(projection_scenarios.to_string(index=False, float_format=lambda x:f'{x:.1f}'))
print('\nSupported-hour contribution is a subtotal, not a full-day total.')
print('An unobserved hour remains missing; observed zero orders remain zero.')
print('Intervals resample entire operating days and are pointwise, exploratory ranges.')
print('Full-scenario intervals require at least 90% complete bootstrap draws.')
print('Sparse hours, selective observation, and unmeasured seasonal changes remain limitations.')
print('Schedules are assumptions; partial closing hours assume constant within-hour output.')
print('No annual revenue estimate is produced until order values and coverage are available.')

# ------------------------------------------------------------
# 33.6 Plot hourly output with coverage and uncertainty
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
for ax, mode, title, color in zip(
        axes, ['drive_thru', 'counter'], ['Drive-Thru', 'Counter'], brand_colors[:2]):
    rows = projection_hourly.loc[projection_hourly['mode'].eq(mode)]
    centers = rows['operating_hour'] + .5
    rates = rows['orders_per_hour']
    ax.scatter(centers, rates, color=color, label='Hourly rate', zorder=3)
    intervals = rows['bootstrap_low'].notna()
    ax.vlines(centers.loc[intervals], rows.loc[intervals, 'bootstrap_low'],
              rows.loc[intervals, 'bootstrap_high'], color=color,
              linewidth=2, label='95% day-bootstrap interval')
    for row in rows.itertuples():
        x = row.operating_hour+.5
        if pd.isna(row.orders_per_hour):
            ax.text(x, .03, 'No data', transform=ax.get_xaxis_transform(),
                    rotation=90, ha='center', fontsize=9)
        else:
            ax.annotate(f'{row.observed_days}d' + ('*' if row.preliminary else ''),
                        (x, row.orders_per_hour), xytext=(0,7),
                        textcoords='offset points', ha='center', fontsize=9)
    ax.set_title(f'{title}: Hourly Output for Daily Projections', fontfamily='Archivo Black')
    ax.set_ylabel('Completed Orders per Observed Hour')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.3)
    ax.legend(frameon=False)
axes[-1].set_xticks(np.arange(10, 26)+.5, operating_labels[:-1], rotation=45, ha='right')
axes[-1].set_xlabel('Hour Beginning')
fig.text(.5, .015, '* Fewer than 3 observed days; intervals suppressed. Missing hours are not zeros.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0,.04,1,1))
fig.savefig(IMAGE_DIR / 'daily_projection_hourly_uncertainty.png', dpi=250,
            bbox_inches='tight')
plt.close(fig)

# ============================================================
# 34. CUP, FOOD, PACKAGING AND ORDER-VALUE INPUTS
# ============================================================
# Append after Section 33. Existing imports and CSV_DIR are reused.
# Empty means unknown; zero means observed none. Input files are preserved.
# Packaging counts are proxies, not automatic completed-order counts.

# ------------------------------------------------------------
# 34.1 Create persistent input templates only when absent
# ------------------------------------------------------------
survey_count_columns = [
    'cups_small', 'cups_medium', 'cups_large', 'cups_unknown_size',
    'regular_fries', 'animal_fries', 'bags', 'boxes', 'trays',
    'hamburgers', 'cheeseburgers', 'double_doubles', 'other_burgers',
    'known_burgers', 'known_orders',
]
survey_columns = [
    'sample_id', 'timestamp', 'channel', 'scope', 'sample_type',
    'order_id', 'counts_complete', *survey_count_columns, 'notes',
]
# sample_type: cups / fries / packaging / complete_order / paired_packaging
# scope: single_order / observed_group / table / aggregate
# channel: counter / drive_thru / unknown
# counts_complete: yes only when all relevant items in that scope were counted.
# known_burgers / known_orders: direct counts, not guesses from packaging.
survey_input_path = CSV_DIR / 'composition_samples_input.csv'
price_input_path = CSV_DIR / 'menu_prices_input.csv'
if not survey_input_path.exists():
    pd.DataFrame(columns=survey_columns).to_csv(survey_input_path, index=False)
if not price_input_path.exists():
    pd.DataFrame({
        'item': ['hamburgers', 'cheeseburgers', 'double_doubles',
                 'regular_fries', 'animal_fries',
                 'cups_small', 'cups_medium', 'cups_large'],
        'unit_price': np.nan, 'tax_included': '', 'price_date': '', 'notes': '',
    }).to_csv(price_input_path, index=False)

# ------------------------------------------------------------
# 34.2 Validate entered counts without replacing missing values
# ------------------------------------------------------------
composition_samples = pd.read_csv(survey_input_path)
missing_columns = set(survey_columns) - set(composition_samples.columns)
if missing_columns:
    raise ValueError(f'Composition input lacks columns: {sorted(missing_columns)}')
if not composition_samples.empty:
    if composition_samples['sample_id'].isna().any() or composition_samples['sample_id'].duplicated().any():
        raise ValueError('Every composition sample needs a unique sample_id.')
    for column in survey_count_columns:
        entered = composition_samples[column].notna()
        numeric = pd.to_numeric(composition_samples[column], errors='coerce')
        invalid = entered & (numeric.isna() | numeric.lt(0) | numeric.mod(1).ne(0))
        if invalid.any():
            raise ValueError(f'{column}: counts must be nonnegative whole numbers or blank.')
        composition_samples[column] = numeric
    for column, allowed in [
        ('channel', ['counter', 'drive_thru', 'unknown']),
        ('scope', ['single_order', 'observed_group', 'table', 'aggregate']),
        ('sample_type', ['cups', 'fries', 'packaging', 'complete_order', 'paired_packaging']),
        ('counts_complete', ['yes', 'no']),
    ]:
        values = composition_samples[column].fillna('').astype(str).str.strip().str.lower()
        if not values.isin(allowed).all():
            raise ValueError(f'{column}: use one of {allowed}.')
        composition_samples[column] = values

# ------------------------------------------------------------
# 34.3 Summarize complete cup and fries surveys separately
# ------------------------------------------------------------
composition_summary_rows = []
for survey_type, columns in [
        ('cups', ['cups_small', 'cups_medium', 'cups_large', 'cups_unknown_size']),
        ('fries', ['regular_fries', 'animal_fries'])]:
    for channel in ['counter', 'drive_thru', 'unknown']:
        rows = composition_samples.loc[
            composition_samples['sample_type'].eq(survey_type)
            & composition_samples['channel'].eq(channel)
            & composition_samples['counts_complete'].eq('yes')]
        # Require explicit zeroes for absent categories; do not turn blanks into zeroes.
        complete = rows.dropna(subset=columns)
        totals = complete[columns].sum()
        denominator = totals.sum()
        for item in columns:
            composition_summary_rows.append({
                'survey_type': survey_type, 'channel': channel, 'item': item,
                'sample_groups': len(complete), 'count': totals[item],
                'fraction_of_counted_items': totals[item]/denominator if denominator else np.nan,
            })
composition_survey_summary = pd.DataFrame(composition_summary_rows)
composition_survey_summary.to_csv(CSV_DIR / 'composition_survey_summary.csv', index=False)

# ------------------------------------------------------------
# 34.4 Calibrate packaging only from paired, fully counted samples
# ------------------------------------------------------------
packaging_calibration_rows = []
for channel in ['counter', 'drive_thru', 'unknown']:
    for package in ['bags', 'boxes', 'trays']:
        other_packages = [name for name in ['bags', 'boxes', 'trays'] if name != package]
        paired = composition_samples.loc[
            composition_samples['sample_type'].eq('paired_packaging')
            & composition_samples['counts_complete'].eq('yes')
            & composition_samples['channel'].eq(channel)
            & composition_samples[package].gt(0)
            & composition_samples[other_packages].eq(0).all(axis=1)]
        # Mixed packaging cannot identify which container held which contents.
        for outcome in ['known_burgers', 'known_orders']:
            usable = paired.dropna(subset=[outcome])
            quantity = usable[package].sum()
            packaging_calibration_rows.append({
                'channel': channel, 'package_type': package, 'outcome': outcome,
                'paired_sample_groups': len(usable), 'counted_packages': quantity,
                'counted_contents': usable[outcome].sum(),
                'contents_per_package': usable[outcome].sum()/quantity if quantity else np.nan,
            })
packaging_calibration = pd.DataFrame(packaging_calibration_rows)
packaging_calibration.to_csv(CSV_DIR / 'packaging_calibration.csv', index=False)

# ------------------------------------------------------------
# 34.5 Validate recorded prices and report available inputs
# ------------------------------------------------------------
menu_price_inputs = pd.read_csv(price_input_path)
for column in ['item', 'unit_price', 'tax_included', 'price_date', 'notes']:
    if column not in menu_price_inputs:
        raise ValueError(f'Menu-price input lacks column: {column}')
entered_prices = menu_price_inputs['unit_price'].notna()
numeric_prices = pd.to_numeric(menu_price_inputs['unit_price'], errors='coerce')
if (entered_prices & (numeric_prices.isna() | numeric_prices.le(0))).any():
    raise ValueError('Menu prices must be positive numbers or blank.')
menu_price_inputs['unit_price'] = numeric_prices
if menu_price_inputs['item'].isna().any() or menu_price_inputs['item'].duplicated().any():
    raise ValueError('Menu-price items must be present and unique.')
price_tax_flags = menu_price_inputs['tax_included'].fillna('').astype(str).str.lower().str.strip()
if not price_tax_flags.loc[entered_prices].isin(['yes', 'no']).all():
    raise ValueError('Each entered price needs tax_included = yes or no.')

print('\n' + '='*80)
print('COMPOSITION AND PACKAGING INPUT INVENTORY')
print('='*80)
print(f'Entered observation groups: {len(composition_samples)}')
print(f'Entered menu prices: {entered_prices.sum()}')
print(f'Composition input: {survey_input_path}')
print(f'Price input: {price_input_path}')
if not composition_samples.empty:
    print(composition_samples.groupby(['channel', 'sample_type']).size().to_string())
    print('\nPackaging calibration:')
    print(packaging_calibration.to_string(index=False))
print('Blank counts mean unknown. Use zero only when that category was checked.')
print('Table/group surveys describe item composition; they are not automatically individual orders.')
print('Paired packaging samples calibrate proxies. Mixed packaging stays in the input audit.')
print('No order value or revenue is inferred from packaging alone.')

# ============================================================
# 35. DAY INFLUENCE AND LATE-MARKER SENSITIVITY
# ============================================================
# Append after Section 34 in the audited script. No new imports.
# Leave-one-day-out ranges measure influence; they are not confidence intervals.

# ------------------------------------------------------------
# 35.1 Remove one operating day at a time on the same hourly grid
# ------------------------------------------------------------
influence_minutes = projection_exposure.unstack(['operating_hour', 'mode']).reindex(
    index=projection_days, columns=projection_cells, fill_value=0).fillna(0).to_numpy()
influence_orders = projection_counts.unstack(['operating_hour', 'mode']).reindex(
    index=projection_days, columns=projection_cells, fill_value=0).fillna(0).to_numpy()
remaining_minutes = influence_minutes.sum(axis=0) - influence_minutes
remaining_orders = influence_orders.sum(axis=0) - influence_orders
leave_day_rates = np.divide(
    remaining_orders*60, remaining_minutes,
    out=np.full(remaining_minutes.shape, np.nan), where=remaining_minutes > 0)

hour_influence_rows = []
for day_index, day in enumerate(projection_days):
    for column, (hour, mode) in enumerate(projection_cells):
        baseline = projection_rates[column]
        revised = leave_day_rates[day_index, column]
        hour_influence_rows.append({
            'removed_day': day, 'operating_hour': hour, 'mode': mode,
            'baseline_rate': baseline, 'rate_without_day': revised,
            'removed_minutes': influence_minutes[day_index, column],
            'remaining_minutes': remaining_minutes[day_index, column],
            'remaining_days': int(projection_day_counts[column]
                                  - (influence_minutes[day_index, column] > 0)),
            'change_orders_per_hour': revised-baseline,
            'percent_change': (revised-baseline)/baseline*100 if baseline > 0 else np.nan,
            'lost_hour_coverage': bool(np.isfinite(baseline) and not np.isfinite(revised)),
        })
hour_influence_audit = pd.DataFrame(hour_influence_rows)
hour_influence_audit.to_csv(CSV_DIR / 'leave_one_day_out_hourly_audit.csv', index=False)

# ------------------------------------------------------------
# 35.2 Summarize each hour's most influential contributing day
# ------------------------------------------------------------
hour_influence_summary_rows = []
for (hour, mode), group in hour_influence_audit.groupby(['operating_hour', 'mode']):
    contributing = group.loc[group['removed_minutes'].gt(0)]
    finite = contributing.dropna(subset=['rate_without_day'])
    influential = (finite.loc[finite['change_orders_per_hour'].abs().idxmax()]
                   if not finite.empty else None)
    column = projection_cells.get_loc((hour, mode))
    hour_influence_summary_rows.append({
        'operating_hour': hour, 'mode': mode,
        'baseline_rate': projection_rates[column],
        'observed_days': int(projection_day_counts[column]),
        'minimum_rate_without_day': finite['rate_without_day'].min(),
        'maximum_rate_without_day': finite['rate_without_day'].max(),
        'most_influential_day': influential['removed_day'] if influential is not None else None,
        'largest_absolute_change': finite['change_orders_per_hour'].abs().max(),
        'largest_absolute_percent_change': finite['percent_change'].abs().max(),
        'deletions_losing_hour_coverage': int(contributing['lost_hour_coverage'].sum()),
    })
hour_influence_summary = pd.DataFrame(hour_influence_summary_rows)
hour_influence_summary.to_csv(CSV_DIR / 'leave_one_day_out_hourly_summary.csv', index=False)

# ------------------------------------------------------------
# 35.3 Compare projected subtotals without silently dropping hours
# ------------------------------------------------------------
# Same illustrative 10 AM to 1 AM schedule used in Section 33.
# Each comparison retains ALL originally supported hours for that channel.
# Missing coverage after a deletion invalidates that subtotal comparison.
daily_influence_rows = []
for mode in ['drive_thru', 'counter', 'combined']:
    required = np.array([
        hour < 25 and (mode == 'combined' or channel == mode)
        for hour, channel in projection_cells])
    supported = required & np.isfinite(projection_rates)
    baseline = projection_rates[supported].sum()
    baseline_complete = bool(np.isfinite(projection_rates[required]).all())
    for day_index, day in enumerate(projection_days):
        covered = bool(np.isfinite(leave_day_rates[day_index, supported]).all())
        revised = leave_day_rates[day_index, supported].sum() if covered else np.nan
        lost = supported & ~np.isfinite(leave_day_rates[day_index])
        daily_influence_rows.append({
            'removed_day': day, 'mode': mode, 'schedule': '10 AM to 1 AM',
            'baseline_supported_hour_contribution': baseline,
            'contribution_without_day': revised,
            'change_orders': revised-baseline,
            'percent_change': (revised-baseline)/baseline*100 if baseline > 0 else np.nan,
            'baseline_covers_full_schedule': baseline_complete,
            'lost_supported_hours': '; '.join(
                f'{channel} {operating_labels[hour-10]}'
                for (hour, channel), missing in zip(projection_cells, lost) if missing),
        })
daily_influence_audit = pd.DataFrame(daily_influence_rows)
daily_influence_audit.to_csv(CSV_DIR / 'leave_one_day_out_daily_audit.csv', index=False)

print('\n' + '='*85)
print('LEAVE-ONE-DAY-OUT INFLUENCE — CURRENT DATA')
print('='*85)
print('Hourly estimates most affected by a contributing day:')
print(hour_influence_summary.sort_values('largest_absolute_change', ascending=False).head(12)
      .to_string(index=False, float_format=lambda x:f'{x:.2f}'))
for mode, group in daily_influence_audit.groupby('mode'):
    finite = group.dropna(subset=['contribution_without_day'])
    print(f'\n{mode}: baseline supported-hour contribution '
          f'{group["baseline_supported_hour_contribution"].iloc[0]:.1f}')
    print(f'Deletions losing required coverage: {group["contribution_without_day"].isna().sum()}')
    if not finite.empty:
        print(f'Comparable deletion range: {finite["contribution_without_day"].min():.1f}'
              f' to {finite["contribution_without_day"].max():.1f}')
        worst = finite.loc[finite['change_orders'].abs().idxmax()]
        print(f'Largest comparable change: remove {worst["removed_day"]}, '
              f'{worst["change_orders"]:+.1f} orders ({worst["percent_change"]:+.1f}%).')
print('Where baseline_covers_full_schedule is false, the comparison is a supported-hours subtotal.')
print('Deletion ranges describe day influence, not prediction or confidence intervals.')

# ------------------------------------------------------------
# 35.4 Refit post-advance distributions without flagged late taps
# ------------------------------------------------------------
late_marker_fit_rows = []
for sample_name, frame in [
        ('All matched episodes', pause_completion_matches),
        ('Flagged late taps excluded', pause_completion_matches.loc[
            ~pause_completion_matches['late_up_flag']])]:
    values = pd.to_numeric(frame['up_to_completion_seconds'], errors='coerce').dropna().to_numpy()
    values = np.sort(values[values > 0])
    if len(values) < 10:
        print(f'Skipping {sample_name}: fewer than 10 positive episodes.')
        continue
    for model_name in ['Exponential', 'Gamma']:
        model, shape, scale, aic = fit_movement_model(values, model_name, None)
        late_marker_fit_rows.append({
            'sample': sample_name, 'model': model_name, 'n': len(values),
            'observed_mean_seconds': values.mean(), 'observed_median_seconds': np.median(values),
            'gamma_shape': shape, 'scale_seconds': scale,
            'KS_distance': stats.kstest(values, model.cdf).statistic, 'AIC': aic,
        })
late_marker_fit_summary = pd.DataFrame(late_marker_fit_rows)
if not late_marker_fit_summary.empty:
    late_marker_fit_summary['delta_AIC'] = (
        late_marker_fit_summary['AIC']
        - late_marker_fit_summary.groupby('sample')['AIC'].transform('min'))
    late_marker_fit_summary.to_csv(CSV_DIR / 'late_marker_fit_sensitivity.csv', index=False)
    print('\nPOST-ADVANCE FIT SENSITIVITY TO FLAGGED LATE TAPS')
    print(late_marker_fit_summary.to_string(index=False, float_format=lambda x:f'{x:.3f}'))
    print('Compare delta AIC within each sample; raw AIC differs with sample size.')
    print('Flags identify candidate late taps, not confirmed timing errors.')
    print('Both samples remain selected matched episodes; this does not measure lost capacity.')

# ------------------------------------------------------------
# 35.5 Plot hourly deletion ranges and observed post-advance curves
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for ax, mode, title, color in zip(
        axes[:2], ['drive_thru', 'counter'], ['Drive-Thru', 'Counter'], brand_colors[:2]):
    rows = hour_influence_summary.loc[hour_influence_summary['mode'].eq(mode)]
    centers = rows['operating_hour']+.5
    ax.scatter(centers, rows['baseline_rate'], color=color, zorder=3, label='All observed days')
    ax.vlines(centers, rows['minimum_rate_without_day'], rows['maximum_rate_without_day'],
              color=color, linewidth=2, label='Leave-one-day-out range')
    for row in rows.itertuples():
        if row.observed_days <= 1:
            label = 'No data' if row.observed_days == 0 else 'Only 1 day'
            ax.text(row.operating_hour+.5, .03, label, transform=ax.get_xaxis_transform(),
                    rotation=90, ha='center', fontsize=8)
    ax.set_title(f'{title}: Day Influence', fontfamily='Archivo Black')
    ax.set(xlabel='Hour Beginning', ylabel='Orders per Observed Hour')
    ax.set_xticks(np.arange(10,26)+.5, operating_labels[:-1], rotation=90)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
for label, frame in [('All matched episodes', pause_completion_matches),
                     ('Late-tap flags excluded', pause_completion_matches.loc[
                         ~pause_completion_matches['late_up_flag']])]:
    values = frame['up_to_completion_seconds']
    draw_gap_survival(axes[2], values.loc[values.gt(0)], f'{label} (n={values.gt(0).sum()})')
axes[2].set_title('Post-Advance Timing Sensitivity', fontfamily='Archivo Black')
axes[2].set(xlabel='Seconds After Advancing', ylabel='Proportion Longer Than x')
axes[2].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.grid(axis='y', alpha=.3)
fig.text(.5,.015, 'Day-deletion ranges are influence checks, not confidence intervals. '
         'Late-tap exclusions are a sensitivity analysis.', ha='center', fontsize=10)
fig.tight_layout(rect=(0,.045,1,1))
fig.savefig(IMAGE_DIR / 'day_and_late_marker_sensitivity.png', dpi=250, bbox_inches='tight')
plt.close(fig)
# ============================================================
# 36. DOES A LONGER ADVANCE PAUSE PRECEDE A FASTER PICKUP?
# ============================================================
# Append after Section 35. No new imports.
# Post-advance time includes advancing and pickup, not just food preparation.
# Matched episodes have total completion gaps <=600s. That selection can
# itself induce a negative pause/post-advance association. Results are descriptive.

# ------------------------------------------------------------
# 36.1 Prepare matched episodes and explicit pause-length groups
# ------------------------------------------------------------
pause_length_episodes = pause_completion_matches.loc[
    pause_completion_matches['pause_seconds'].gt(0)
    & pause_completion_matches['up_to_completion_seconds'].gt(0)].copy()
pause_length_episodes['operating_day'] = (
    pause_length_episodes['completion_time']-pd.Timedelta(hours=2)).dt.date
pause_length_episodes['pause_group'] = pd.cut(
    pause_length_episodes['pause_seconds'], bins=[0,30,60,np.inf],
    labels=['Up to 30 seconds', 'Over 30 to 60 seconds', 'Over 60 seconds'])
pause_length_samples = {
    'All matched episodes': pause_length_episodes,
    'Flagged late taps excluded': pause_length_episodes.loc[
        ~pause_length_episodes['late_up_flag']].copy(),
}

# ------------------------------------------------------------
# 36.2 Compare distributions and rapid completions after advancing
# ------------------------------------------------------------
pause_length_rows = []
for sample_name, sample in pause_length_samples.items():
    for group_name, group in sample.groupby('pause_group', observed=True):
        values = group['up_to_completion_seconds']
        pause_length_rows.append({
            'sample':sample_name, 'pause_group':str(group_name), 'episodes':len(group),
            'days':group['operating_day'].nunique(), 'blocks':group['segment_id'].nunique(),
            'median_pause_seconds':group['pause_seconds'].median(),
            'post_advance_q25':values.quantile(.25),
            'post_advance_median':values.median(), 'post_advance_q75':values.quantile(.75),
            'fraction_completed_within_10s':values.le(10).mean(),
            'fraction_completed_within_20s':values.le(20).mean(),
        })
pause_length_summary = pd.DataFrame(pause_length_rows)
pause_length_summary.to_csv(CSV_DIR / 'pause_length_distribution_summary.csv', index=False)
pause_length_episodes.to_csv(CSV_DIR / 'pause_length_episode_audit.csv', index=False)

# ------------------------------------------------------------
# 36.3 Compare associations pooled and within observation blocks
# ------------------------------------------------------------
def pause_length_association(sample, within_block=False, draws=2000):
    """Descriptive log-log slope with whole-day resampling."""
    frame = sample.copy()
    if within_block:
        # Single-episode blocks cannot inform a within-block comparison.
        sizes = frame.groupby('segment_id')['segment_id'].transform('size')
        frame = frame.loc[sizes >= 2].copy()
    frame['x'] = np.log(frame['pause_seconds'])
    frame['y'] = np.log(frame['up_to_completion_seconds'])
    if within_block:
        frame['x'] -= frame.groupby('segment_id')['x'].transform('mean')
        frame['y'] -= frame.groupby('segment_id')['y'].transform('mean')
    if frame.empty:
        return {'episodes':0,'days':0,'blocks':0,'slope':np.nan,
                'doubling_pause_multiplier':np.nan,'lower_95':np.nan,'upper_95':np.nan}
    for name, values in [('xx',frame['x']**2), ('xy',frame['x']*frame['y'])]:
        frame[name] = values
    frame['n'] = 1
    totals = frame.groupby('operating_day')[['n','x','y','xx','xy']].sum().to_numpy()
    def slope(moment):
        n,x,y,xx,xy = np.moveaxis(moment,-1,0)
        denominator = xx if within_block else xx-x*x/n
        numerator = xy if within_block else xy-x*y/n
        return np.divide(numerator,denominator,
                         out=np.full(np.shape(denominator),np.nan),where=denominator>1e-10)
    estimate = float(slope(totals.sum(axis=0)))
    low = high = np.nan
    if len(totals) >= 5:
        rng = np.random.default_rng(3606)
        weights = rng.multinomial(len(totals),np.full(len(totals),1/len(totals)),size=draws)
        slopes = slope(weights @ totals)
        finite = slopes[np.isfinite(slopes)]
        if len(finite) >= .90*draws:
            low,high = np.quantile(finite,[.025,.975])
    return {'episodes':len(frame),'days':len(totals),'blocks':frame['segment_id'].nunique(),
            'slope':estimate,'doubling_pause_multiplier':2**estimate,
            'lower_95':low,'upper_95':high}

pause_association_rows = []
for sample_name,sample in pause_length_samples.items():
    for within_block in [False,True]:
        result = pause_length_association(sample,within_block)
        pause_association_rows.append({
            'sample':sample_name,'comparison':'Within block' if within_block else 'Pooled',
            **result})
pause_association_summary = pd.DataFrame(pause_association_rows)
pause_association_summary.to_csv(CSV_DIR / 'pause_length_association_summary.csv', index=False)

# ------------------------------------------------------------
# 36.4 Print results and their operational meaning
# ------------------------------------------------------------
print('\n'+'='*95)
print('PAUSE LENGTH VS TIME FROM ADVANCING TO COMPLETION')
print('='*95)
print(pause_length_summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print('\nAssociation accounting for different observation-block levels:')
print(pause_association_summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print('Doubling-pause multiplier <1 describes shorter post-advance times; >1 describes longer times.')
print('Within-block comparison removes each block\'s average log pause and pickup time.')
print('Rapid-completion thresholds of 10s and 20s are descriptive, not physical minimums.')
print('The 600s TOTAL-gap selection can induce a negative association; resampling does not fix it.')
print('Marker uncertainty is approximately 5s; values near group boundaries can change category.')
print('Food-ready time is unobserved. This comparison does not establish causal lost capacity.')

# ------------------------------------------------------------
# 36.5 Plot conditional distributions and marker sensitivity
# ------------------------------------------------------------
fig,axes = plt.subplots(1,3,figsize=(19,6))
for ax,(sample_name,sample) in zip(axes[:2],pause_length_samples.items()):
    for group_name,group in sample.groupby('pause_group',observed=True):
        draw_gap_survival(ax,group['up_to_completion_seconds'],
                          f'{group_name} (n={len(group)})')
    ax.set_title('All Matched Episodes' if sample_name=='All matched episodes'
                 else 'Flagged Late Taps Excluded',fontfamily='Archivo Black')
    ax.set(xlabel='Seconds After Advancing',ylabel='Proportion Longer Than x',ylim=(0,1))
    ax.legend(frameon=False,fontsize=9)
for label,sample in pause_length_samples.items():
    axes[2].scatter(sample['pause_seconds'],sample['up_to_completion_seconds'],
                    label=label,s=28,alpha=.45)
axes[2].set_title('Pause vs Post-Advance Time',fontfamily='Archivo Black')
axes[2].set(xlabel='Recorded Advance Pause (Seconds)',ylabel='Seconds After Advancing')
axes[2].legend(frameon=False,fontsize=9)
for ax in axes:
    ax.grid(alpha=.3)
fig.text(.5,.015,'Selected matched episodes; total completion gap <=600s. '
         'Associations cannot establish when food became ready.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.05,1,1))
fig.savefig(IMAGE_DIR / 'pause_length_vs_post_advance.png',dpi=250,bbox_inches='tight')
plt.close(fig)

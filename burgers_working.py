import pandas as pd


# =============================================================================
# 0. CLEAR TERMINAL OUTPUT
# =============================================================================

# Clear the terminal and move the cursor back to the top-left corner
# before each run so only the current analysis output is visible.
print("\033[2J\033[H", end="")


# =============================================================================
# IN-N-OUT QUEUE ANALYSIS
# =============================================================================
#
# Raw observation period:
# August 17, 2026 through September 6, 2026
#
# This script preserves both the exploratory inspection work and the
# later cleaning/analysis steps. Diagnostics are intentionally retained
# because the process of discovering and correcting problems is part
# of the project.
#
#
# TABLE OF CONTENTS
# -----------------
#
# 1. Project settings and raw-data load
#
# 2. Optional raw-data inspection
#    2.1 Top-level structure
#    2.2 Nested session structure
#    2.3 Historical "original" JSON structure
#
# 3. Flatten mixed historical formats
#
# 4. Validate flattened event data
#
# 5. Prepare core throughput events
#    5.1 Convert timestamps
#    5.2 Separate drive-thru and counter events
#    5.3 Recalculate counter-order repeats
#    5.4 Validate corrected throughput counts
#
# 6. Define preliminary observation windows
#    6.1 Timer-boundary rules
#    6.2 Build preliminary windows
#
# 7. Validate preliminary observation windows
#
# 8. Diagnose suspicious preliminary windows
#    8.1 Longest windows
#    8.2 Missing run IDs
#
# 9. Diagnose gaps inside historical run IDs
#
# 10. Create cleaned observation segments
#     10.1 Mode-specific inactivity rules
#     10.2 Assign cleaned segment IDs
#     10.3 Identify gap-capped segment endings
#
# 11. Build cleaned observation windows
#     11.1 Apply timer-start rule
#     11.2 Apply timer-end / gap-cap rule
#
# 12. Validate cleaned observation windows
#
# =============================================================================


# =============================================================================
# 1. PROJECT SETTINGS AND RAW-DATA LOAD
# =============================================================================

START_DATE = "2026-08-17"
END_DATE = "2026-09-06"


# Keep the inspection code in the file, but suppress the large amount
# of exploratory console output during normal runs.
RUN_INSPECTION = False


# Load the combined raw JSON export.
#
# The raw file is treated as immutable.
df = pd.read_json("innout_log_2026.json")


# =============================================================================
# 2. OPTIONAL RAW-DATA INSPECTION
# =============================================================================

if RUN_INSPECTION:

    # -------------------------------------------------------------------------
    # 2.1 Top-level structure
    # -------------------------------------------------------------------------

    print("\n========================================")
    print("TOP-LEVEL DATA INSPECTION")
    print("========================================")

    print(df.head())

    df.info()

    print("\nTop-level shape:")
    print(df.shape)

    print("\nMissing values by top-level field:")
    print(df.isna().sum())


    # -------------------------------------------------------------------------
    # 2.2 Nested session structure
    # -------------------------------------------------------------------------

    print("\n========================================")
    print("SESSION STRUCTURE INSPECTION")
    print("========================================")

    for index, row in df.iterrows():

        session = row["sessions"]

        print("\n----------------------------------------")
        print(f"Top-level row: {index}")
        print(f"Type: {type(session)}")

        if isinstance(session, dict):

            print(f"Keys: {session.keys()}")

        elif isinstance(session, list):

            print(f"Number of items: {len(session)}")

            if len(session) > 0:

                print(
                    f"First item type: "
                    f"{type(session[0])}"
                )

                if isinstance(session[0], dict):

                    print(
                        f"First item keys: "
                        f"{session[0].keys()}"
                    )


    # -------------------------------------------------------------------------
    # 2.3 Historical "original" JSON structure
    # -------------------------------------------------------------------------

    print("\n========================================")
    print("'ORIGINAL' STRUCTURE INSPECTION")
    print("========================================")

    for index in [3, 4]:

        session = df.loc[index, "sessions"]
        original = session["original"]

        print("\n----------------------------------------")
        print(f"Top-level row: {index}")
        print(
            f"'original' type: "
            f"{type(original)}"
        )

        if isinstance(original, dict):

            print(
                f"'original' keys: "
                f"{original.keys()}"
            )

        elif isinstance(original, list):

            print(
                f"Number of items: "
                f"{len(original)}"
            )

            if len(original) > 0:

                print(
                    f"First item type: "
                    f"{type(original[0])}"
                )

                if isinstance(original[0], dict):

                    print(
                        f"First item keys: "
                        f"{original[0].keys()}"
                    )


# =============================================================================
# 3. FLATTEN MIXED HISTORICAL FORMATS
# =============================================================================

# Earlier converted text logs store events directly under:
#
#     session["events"]
#
# Later structured JSON exports preserve the original export under:
#
#     session["original"]["events"]
#
# Normalize both structures into one event-level DataFrame.

all_events = []


for index, row in df.iterrows():

    session = row["sessions"]


    if "events" in session:

        events = session["events"]

    elif "original" in session:

        original = session["original"]

        if (
            isinstance(original, dict)
            and "events" in original
        ):

            events = original["events"]

        else:

            raise ValueError(
                "Could not find events inside "
                f"'original' for row {index}"
            )

    else:

        raise ValueError(
            "Could not find an event list for "
            f"top-level row {index}"
        )


    for event in events:

        # Copy rather than alter the nested raw object.
        event_record = event.copy()

        # Preserve provenance.
        event_record["source_file"] = (
            session.get("source_file")
        )

        event_record["session_id"] = (
            session.get("session_id")
        )

        event_record["source_row"] = index

        all_events.append(event_record)


events_df = pd.DataFrame(all_events)


# =============================================================================
# 4. VALIDATE FLATTENED EVENT DATA
# =============================================================================

print("\n========================================")
print("FLATTENED EVENT DATA")
print("========================================")


print(f"Shape: {events_df.shape}")


print("\nEvent counts by kind:")

print(
    events_df["kind"].value_counts(
        dropna=False
    )
)


print("\nMissing timestamps:")

print(
    events_df["timestamp"].isna().sum()
)


print("\nEarliest timestamp:")
print(events_df["timestamp"].min())


print("\nLatest timestamp:")
print(events_df["timestamp"].max())


print("\nFirst 10 flattened events:")

print(
    events_df[
        [
            "timestamp",
            "mode",
            "kind",
            "text",
            "source_file",
            "source_row",
        ]
    ].head(10)
)


# =============================================================================
# 5. PREPARE CORE THROUGHPUT EVENTS
# =============================================================================


# -----------------------------------------------------------------------------
# 5.1 Convert timestamps
# -----------------------------------------------------------------------------

events_df["timestamp"] = pd.to_datetime(
    events_df["timestamp"]
)


# -----------------------------------------------------------------------------
# 5.2 Separate drive-thru and counter events
# -----------------------------------------------------------------------------

# Every car event represents one completed drive-thru order.
drive_events = events_df[
    events_df["kind"] == "car"
].copy()


# Counter events can represent multiple completed orders at one timestamp.
counter_events = events_df[
    events_df["kind"] == "order"
].copy()


# -----------------------------------------------------------------------------
# 5.3 Recalculate counter-order repeats
# -----------------------------------------------------------------------------

# Historical versions of the logger could incorrectly classify an
# order number as a repeat because the same number had appeared in an
# earlier order-number cycle.
#
# Analytical rule:
#
# An announced counter order counts as a repeat ONLY when the same
# order number appeared during the previous 30 minutes.

REPEAT_WINDOW = pd.Timedelta(minutes=30)

last_seen_order = {}
corrected_counts = []


counter_events = counter_events.sort_values(
    "timestamp"
).copy()


for _, row in counter_events.iterrows():

    timestamp = row["timestamp"]
    numbers = row.get("numbers")


    if isinstance(numbers, list) and len(numbers) > 0:

        new_orders_in_event = 0

        for order_number in numbers:

            previous_time = last_seen_order.get(
                order_number
            )


            if (
                previous_time is None
                or timestamp - previous_time > REPEAT_WINDOW
            ):

                new_orders_in_event += 1


            last_seen_order[order_number] = timestamp


        corrected_counts.append(
            new_orders_in_event
        )


    else:

        # Unannounced orders cannot be matched by number.
        corrected_counts.append(1)


counter_events["corrected_counted"] = (
    corrected_counts
)


# -----------------------------------------------------------------------------
# 5.4 Validate corrected throughput counts
# -----------------------------------------------------------------------------

print("\n========================================")
print("CORE THROUGHPUT EVENTS")
print("========================================")


print(
    f"Drive-thru orders: "
    f"{len(drive_events)}"
)


print(
    "Counter order-event rows: "
    f"{len(counter_events)}"
)


print(
    "Counter orders after 30-minute repeat correction: "
    f"{int(counter_events['corrected_counted'].sum())}"
)


print(
    "Counter orders using original logger counts: "
    f"{int(counter_events['counted'].fillna(1).sum())}"
)


# =============================================================================
# 6. DEFINE PRELIMINARY OBSERVATION WINDOWS
# =============================================================================


# -----------------------------------------------------------------------------
# 6.1 Timer-boundary rules
# -----------------------------------------------------------------------------

# Timer metadata is useful but not automatically trusted.
#
# START:
#
# If a timer began no more than 8 minutes before the first qualifying
# throughput event, use the timer start.
#
# If it began more than 8 minutes before the first qualifying event,
# use the first actual throughput event instead.
#
#
# END:
#
# If a timer ended no more than 10 minutes after the last qualifying
# throughput event, use the timer end.
#
# If it ended more than 10 minutes later, use the last actual
# throughput event instead.

START_ALLOWANCE = pd.Timedelta(minutes=8)
END_ALLOWANCE = pd.Timedelta(minutes=10)


# -----------------------------------------------------------------------------
# 6.2 Build preliminary windows
# -----------------------------------------------------------------------------

core_events = events_df[
    events_df["kind"].isin(
        ["car", "order"]
    )
].copy()


observation_windows = []


group_columns = [
    "source_row",
    "run_id",
    "mode",
]


for group_key, group in core_events.groupby(
    group_columns,
    dropna=False
):

    source_row, run_id, mode = group_key


    group = group.sort_values(
        "timestamp"
    ).copy()


    first_event = group["timestamp"].iloc[0]
    last_event = group["timestamp"].iloc[-1]


    effective_start = first_event
    effective_end = last_event

    start_source = "first_event"
    end_source = "last_event"


    timer_events = events_df[
        (events_df["source_row"] == source_row)
        & (events_df["run_id"] == run_id)
        & (events_df["mode"] == mode)
        & (
            events_df["kind"].isin(
                [
                    "timer_start",
                    "timer_end",
                ]
            )
        )
    ].copy()


    # -------------------------------------------------------------------------
    # Timer start
    # -------------------------------------------------------------------------

    timer_starts = timer_events[
        timer_events["kind"] == "timer_start"
    ]["timestamp"]


    timer_starts = timer_starts[
        timer_starts <= first_event
    ]


    if not timer_starts.empty:

        timer_start = timer_starts.max()

        if (
            first_event - timer_start
            <= START_ALLOWANCE
        ):

            effective_start = timer_start
            start_source = "timer"


    # -------------------------------------------------------------------------
    # Timer end
    # -------------------------------------------------------------------------

    timer_ends = timer_events[
        timer_events["kind"] == "timer_end"
    ]["timestamp"]


    timer_ends = timer_ends[
        timer_ends >= last_event
    ]


    if not timer_ends.empty:

        timer_end = timer_ends.min()

        if (
            timer_end - last_event
            <= END_ALLOWANCE
        ):

            effective_end = timer_end
            end_source = "timer"


    duration_minutes = (
        effective_end - effective_start
    ).total_seconds() / 60


    observation_windows.append(
        {
            "source_row": source_row,
            "run_id": run_id,
            "mode": mode,
            "first_event": first_event,
            "last_event": last_event,
            "effective_start": effective_start,
            "effective_end": effective_end,
            "start_source": start_source,
            "end_source": end_source,
            "duration_minutes": duration_minutes,
        }
    )


windows_df = pd.DataFrame(
    observation_windows
)


# =============================================================================
# 7. VALIDATE PRELIMINARY OBSERVATION WINDOWS
# =============================================================================

print("\n========================================")
print("OBSERVATION WINDOWS")
print("========================================")


print(
    f"Number of windows: "
    f"{len(windows_df)}"
)


print("\nWindows by mode:")

print(
    windows_df["mode"].value_counts(
        dropna=False
    )
)


print("\nPreliminary observed minutes by mode:")

print(
    windows_df
    .groupby(
        "mode"
    )["duration_minutes"]
    .sum()
    .round(1)
)


print("\nStart boundary source:")

print(
    windows_df["start_source"].value_counts(
        dropna=False
    )
)


print("\nEnd boundary source:")

print(
    windows_df["end_source"].value_counts(
        dropna=False
    )
)


print("\nFirst 10 observation windows:")

print(
    windows_df[
        [
            "source_row",
            "run_id",
            "mode",
            "first_event",
            "effective_start",
            "start_source",
            "last_event",
            "effective_end",
            "end_source",
            "duration_minutes",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# =============================================================================
# 8. DIAGNOSE SUSPICIOUS PRELIMINARY WINDOWS
# =============================================================================


# -----------------------------------------------------------------------------
# 8.1 Longest windows
# -----------------------------------------------------------------------------

# Initial calculations revealed implausibly long windows.
#
# These diagnostics remain here to document the issue rather than
# hiding it after the later cleaning correction.

print("\n========================================")
print("LONGEST OBSERVATION WINDOWS")
print("========================================")


print(
    windows_df[
        [
            "source_row",
            "run_id",
            "mode",
            "effective_start",
            "effective_end",
            "duration_minutes",
        ]
    ]
    .sort_values(
        "duration_minutes",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


# -----------------------------------------------------------------------------
# 8.2 Missing run IDs
# -----------------------------------------------------------------------------

# One early hypothesis was that missing run IDs had caused unrelated
# events to be grouped together.
#
# This test ruled that explanation out.

print("\n========================================")
print("CORE EVENTS WITH MISSING RUN ID")
print("========================================")


print(
    core_events[
        core_events["run_id"].isna()
    ]["mode"].value_counts(
        dropna=False
    )
)


print(
    "\nTotal core events with missing run_id:",
    core_events["run_id"].isna().sum()
)


# =============================================================================
# 9. DIAGNOSE GAPS INSIDE HISTORICAL RUN IDS
# =============================================================================

# The next diagnostic showed that some historical run IDs themselves
# contain very long periods without throughput events.

gap_diagnostics = []


for group_key, group in core_events.groupby(
    [
        "source_row",
        "run_id",
        "mode",
    ],
    dropna=False
):

    source_row, run_id, mode = group_key


    group = group.sort_values(
        "timestamp"
    ).copy()


    group["gap_minutes"] = (
        group["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60
    )


    max_gap = group["gap_minutes"].max()


    gap_diagnostics.append(
        {
            "source_row": source_row,
            "run_id": run_id,
            "mode": mode,
            "event_count": len(group),
            "max_gap_minutes": max_gap,
        }
    )


gap_df = pd.DataFrame(
    gap_diagnostics
)


print("\n========================================")
print("LARGEST GAPS WITHIN RUN IDS")
print("========================================")


print(
    gap_df
    .sort_values(
        "max_gap_minutes",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# 10. CREATE CLEANED OBSERVATION SEGMENTS
# =============================================================================


# -----------------------------------------------------------------------------
# 10.1 Mode-specific inactivity rules
# -----------------------------------------------------------------------------

# Historical run IDs cannot always be treated as one continuous
# observation period.
#
# DRIVE-THRU:
#
# If MORE THAN 10 minutes pass between logged cars:
#
#     - the previous cleaned segment ends at the previous car event
#     - the next car starts a new cleaned segment
#
#
# COUNTER:
#
# If MORE THAN 20 minutes pass between logged completed counter orders:
#
#     - the previous cleaned segment ends at the previous order event
#     - the next order starts a new cleaned segment
#
#
# Any segment ending because of one of these inactivity rules is
# capped at the final logged throughput event. Timer metadata cannot
# extend that segment across the inactivity break.

DRIVE_THRU_SPLIT_GAP = pd.Timedelta(
    minutes=10
)

COUNTER_SPLIT_GAP = pd.Timedelta(
    minutes=20
)


# -----------------------------------------------------------------------------
# 10.2 Assign cleaned segment IDs
# -----------------------------------------------------------------------------

clean_core_events = core_events.sort_values(
    [
        "source_row",
        "run_id",
        "mode",
        "timestamp",
    ]
).copy()


clean_core_events["clean_segment_id"] = 0


for group_key, group in clean_core_events.groupby(
    [
        "source_row",
        "run_id",
        "mode",
    ],
    dropna=False
):

    source_row, run_id, mode = group_key


    group = group.sort_values(
        "timestamp"
    ).copy()


    gaps = group["timestamp"].diff()

    segment_numbers = []
    current_segment = 1


    for gap in gaps:

        if pd.isna(gap):

            segment_numbers.append(
                current_segment
            )

            continue


        if mode == "drive_thru":

            if gap > DRIVE_THRU_SPLIT_GAP:

                current_segment += 1


        elif mode == "counter":

            if gap > COUNTER_SPLIT_GAP:

                current_segment += 1


        segment_numbers.append(
            current_segment
        )


    clean_core_events.loc[
        group.index,
        "clean_segment_id"
    ] = segment_numbers


clean_core_events["clean_segment_id"] = (
    clean_core_events[
        "clean_segment_id"
    ].astype(int)
)


# -----------------------------------------------------------------------------
# 10.3 Identify gap-capped segment endings
# -----------------------------------------------------------------------------

segment_summary = (
    clean_core_events
    .groupby(
        [
            "source_row",
            "run_id",
            "mode",
            "clean_segment_id",
        ]
    )
    .agg(
        first_event=(
            "timestamp",
            "min"
        ),
        last_event=(
            "timestamp",
            "max"
        ),
        throughput_event_rows=(
            "timestamp",
            "size"
        ),
    )
    .reset_index()
)


segment_summary["max_segment_id"] = (
    segment_summary
    .groupby(
        [
            "source_row",
            "run_id",
            "mode",
        ]
    )["clean_segment_id"]
    .transform("max")
)


segment_summary["ended_by_gap_split"] = (
    segment_summary["clean_segment_id"]
    < segment_summary["max_segment_id"]
)


run_segment_counts = (
    segment_summary
    .groupby(
        [
            "source_row",
            "run_id",
            "mode",
        ]
    )["clean_segment_id"]
    .max()
    .reset_index(
        name="clean_segment_count"
    )
)


print("\n========================================")
print("CLEANED SEGMENT COUNTS")
print("========================================")


print(
    "Historical run/mode groups: "
    f"{len(run_segment_counts)}"
)


print(
    "Groups split into multiple cleaned segments: "
    f"{(run_segment_counts['clean_segment_count'] > 1).sum()}"
)


print("\nGroups that were split:")


split_groups = run_segment_counts[
    run_segment_counts[
        "clean_segment_count"
    ] > 1
]


if split_groups.empty:

    print("None")

else:

    print(
        split_groups
        .sort_values(
            [
                "source_row",
                "run_id",
                "mode",
            ]
        )
        .to_string(index=False)
    )


# =============================================================================
# 11. BUILD CLEANED OBSERVATION WINDOWS
# =============================================================================

clean_windows = []


for _, segment in segment_summary.iterrows():

    source_row = segment["source_row"]
    run_id = segment["run_id"]
    mode = segment["mode"]

    clean_segment_id = segment[
        "clean_segment_id"
    ]

    first_event = segment[
        "first_event"
    ]

    last_event = segment[
        "last_event"
    ]

    ended_by_gap_split = segment[
        "ended_by_gap_split"
    ]


    effective_start = first_event
    effective_end = last_event

    start_source = "first_event"
    end_source = "last_event"


    timer_events = events_df[
        (events_df["source_row"] == source_row)
        & (events_df["run_id"] == run_id)
        & (events_df["mode"] == mode)
        & (
            events_df["kind"].isin(
                [
                    "timer_start",
                    "timer_end",
                ]
            )
        )
    ].copy()


    # -------------------------------------------------------------------------
    # 11.1 Apply timer-start rule
    # -------------------------------------------------------------------------

    timer_starts = timer_events[
        timer_events["kind"] == "timer_start"
    ]["timestamp"]


    timer_starts = timer_starts[
        timer_starts <= first_event
    ]


    if not timer_starts.empty:

        timer_start = timer_starts.max()

        if (
            first_event - timer_start
            <= START_ALLOWANCE
        ):

            effective_start = timer_start
            start_source = "timer"


    # -------------------------------------------------------------------------
    # 11.2 Apply timer-end / gap-cap rule
    # -------------------------------------------------------------------------

    if ended_by_gap_split:

        # A segment ended because the mode-specific inactivity
        # threshold was exceeded.
        #
        # Do not allow a timer to extend the segment beyond the
        # final actual throughput event.
        effective_end = last_event
        end_source = "gap_cap"


    else:

        timer_ends = timer_events[
            timer_events["kind"] == "timer_end"
        ]["timestamp"]


        timer_ends = timer_ends[
            timer_ends >= last_event
        ]


        if not timer_ends.empty:

            timer_end = timer_ends.min()

            if (
                timer_end - last_event
                <= END_ALLOWANCE
            ):

                effective_end = timer_end
                end_source = "timer"


    duration_minutes = (
        effective_end - effective_start
    ).total_seconds() / 60


    clean_windows.append(
        {
            "source_row": source_row,
            "run_id": run_id,
            "mode": mode,
            "clean_segment_id": clean_segment_id,
            "first_event": first_event,
            "last_event": last_event,
            "effective_start": effective_start,
            "effective_end": effective_end,
            "start_source": start_source,
            "end_source": end_source,
            "ended_by_gap_split": ended_by_gap_split,
            "duration_minutes": duration_minutes,
        }
    )


clean_windows_df = pd.DataFrame(
    clean_windows
)


# =============================================================================
# 12. VALIDATE CLEANED OBSERVATION WINDOWS
# =============================================================================

print("\n========================================")
print("CLEANED OBSERVATION WINDOWS")
print("========================================")


print(
    "Number of cleaned windows: "
    f"{len(clean_windows_df)}"
)


print("\nCleaned windows by mode:")

print(
    clean_windows_df[
        "mode"
    ].value_counts(
        dropna=False
    )
)


print("\nCleaned observed minutes by mode:")

print(
    clean_windows_df
    .groupby(
        "mode"
    )["duration_minutes"]
    .sum()
    .round(1)
)


print("\nStart boundary source:")

print(
    clean_windows_df[
        "start_source"
    ].value_counts(
        dropna=False
    )
)


print("\nEnd boundary source:")

print(
    clean_windows_df[
        "end_source"
    ].value_counts(
        dropna=False
    )
)


print("\nLongest cleaned observation windows:")

print(
    clean_windows_df[
        [
            "source_row",
            "run_id",
            "mode",
            "clean_segment_id",
            "effective_start",
            "effective_end",
            "start_source",
            "end_source",
            "duration_minutes",
        ]
    ]
    .sort_values(
        "duration_minutes",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

# =============================================================================
# 13. EXCLUDE KNOWN TEST / DEVELOPMENT SESSION
# =============================================================================

# One small August 19 export was created while the logging tool was
# being tested / transitioned into the newer structured JSON format.
#
# It contains only a few seconds of observation:
#
#     drive-thru window: about 27 seconds
#     counter window:    about 20 seconds
#
# These are valid raw timestamped events, so they remain preserved in
# the raw archive.
#
# However, they are not representative field-observation sessions and
# should not be included in the analytical throughput dataset.
#
# IMPORTANT:
# We do NOT delete these records from events_df or the source JSON.
# Instead, we create new analysis-only DataFrames.

TEST_SOURCE_ROWS = [3]


# -----------------------------------------------------------------------------
# 13.1 Create analysis-only event datasets
# -----------------------------------------------------------------------------

analysis_drive_events = drive_events[
    ~drive_events["source_row"].isin(
        TEST_SOURCE_ROWS
    )
].copy()


analysis_counter_events = counter_events[
    ~counter_events["source_row"].isin(
        TEST_SOURCE_ROWS
    )
].copy()


analysis_windows = clean_windows_df[
    ~clean_windows_df["source_row"].isin(
        TEST_SOURCE_ROWS
    )
].copy()


# -----------------------------------------------------------------------------
# 13.2 Show exactly what is being excluded
# -----------------------------------------------------------------------------

excluded_drive_events = drive_events[
    drive_events["source_row"].isin(
        TEST_SOURCE_ROWS
    )
]


excluded_counter_events = counter_events[
    counter_events["source_row"].isin(
        TEST_SOURCE_ROWS
    )
]


excluded_windows = clean_windows_df[
    clean_windows_df["source_row"].isin(
        TEST_SOURCE_ROWS
    )
]


print("\n========================================")
print("TEST SESSION EXCLUSION")
print("========================================")


print(
    "Excluded source rows:",
    TEST_SOURCE_ROWS
)


print(
    "Excluded drive-thru orders:",
    len(excluded_drive_events)
)


print(
    "Excluded counter order-event rows:",
    len(excluded_counter_events)
)


print(
    "Excluded corrected counter orders:",
    int(
        excluded_counter_events[
            "corrected_counted"
        ].sum()
    )
)


print(
    "Excluded observation windows:",
    len(excluded_windows)
)


print("\nExcluded windows:")

print(
    excluded_windows[
        [
            "source_row",
            "run_id",
            "mode",
            "effective_start",
            "effective_end",
            "duration_minutes",
        ]
    ].to_string(index=False)
)


# -----------------------------------------------------------------------------
# 13.3 Validate final analytical totals after test removal
# -----------------------------------------------------------------------------

print("\n========================================")
print("ANALYTICAL DATASET TOTALS")
print("========================================")


print(
    "Drive-thru orders:",
    len(analysis_drive_events)
)


print(
    "Counter order-event rows:",
    len(analysis_counter_events)
)


print(
    "Corrected counter orders:",
    int(
        analysis_counter_events[
            "corrected_counted"
        ].sum()
    )
)


print(
    "Observation windows:",
    len(analysis_windows)
)


print("\nObserved minutes by mode:")

print(
    analysis_windows
    .groupby(
        "mode"
    )["duration_minutes"]
    .sum()
    .round(1)
)


print("\nObserved hours by mode:")

print(
    (
        analysis_windows
        .groupby(
            "mode"
        )["duration_minutes"]
        .sum()
        / 60
    ).round(2)
)

# =============================================================================
# 14. FINAL INTEGRITY CHECKS
# =============================================================================

# Before saving any cleaned analytical files, run a final set of checks:
#
# 1. Look for duplicate throughput events.
# 2. Verify all cleaned windows have positive duration.
# 3. Verify analytical events fall inside their corresponding
#    cleaned observation windows.
# 4. Reconcile final analytical order totals.


# -----------------------------------------------------------------------------
# 14.1 Check for duplicate analytical throughput events
# -----------------------------------------------------------------------------

# For drive-thru events, timestamp + source_row + run_id + mode + kind
# should uniquely identify an observed car event in this dataset.

drive_duplicate_mask = analysis_drive_events.duplicated(
    subset=[
        "timestamp",
        "source_row",
        "run_id",
        "mode",
        "kind",
    ],
    keep=False,
)

drive_duplicates = analysis_drive_events[
    drive_duplicate_mask
].copy()


# Counter events may contain multiple order numbers at one timestamp,
# but each event row itself should still be unique by the same
# structural fields.

counter_duplicate_mask = analysis_counter_events.duplicated(
    subset=[
        "timestamp",
        "source_row",
        "run_id",
        "mode",
        "kind",
    ],
    keep=False,
)

counter_duplicates = analysis_counter_events[
    counter_duplicate_mask
].copy()


print("\n========================================")
print("DUPLICATE EVENT CHECK")
print("========================================")

print(
    "Duplicate drive-thru event rows:",
    len(drive_duplicates),
)

print(
    "Duplicate counter event rows:",
    len(counter_duplicates),
)


# -----------------------------------------------------------------------------
# 14.2 Check cleaned observation-window durations
# -----------------------------------------------------------------------------

nonpositive_windows = analysis_windows[
    analysis_windows["duration_minutes"] <= 0
].copy()


print("\n========================================")
print("WINDOW DURATION CHECK")
print("========================================")

print(
    "Windows with duration <= 0:",
    len(nonpositive_windows),
)


if not nonpositive_windows.empty:

    print(
        nonpositive_windows[
            [
                "source_row",
                "run_id",
                "mode",
                "clean_segment_id",
                "effective_start",
                "effective_end",
                "duration_minutes",
            ]
        ].to_string(index=False)
    )


# -----------------------------------------------------------------------------
# 14.3 Verify events fall inside cleaned windows
# -----------------------------------------------------------------------------

# We need to match each throughput event to the cleaned segment that
# contains it.
#
# Reuse clean_core_events because it already has clean_segment_id
# assigned to every core throughput event.

analysis_core_events = clean_core_events[
    ~clean_core_events["source_row"].isin(
        TEST_SOURCE_ROWS
    )
].copy()


event_window_check = analysis_core_events.merge(
    analysis_windows[
        [
            "source_row",
            "run_id",
            "mode",
            "clean_segment_id",
            "effective_start",
            "effective_end",
        ]
    ],
    on=[
        "source_row",
        "run_id",
        "mode",
        "clean_segment_id",
    ],
    how="left",
    validate="many_to_one",
)


# Any event outside its cleaned observation window indicates a
# segmentation or boundary error.

events_outside_windows = event_window_check[
    (event_window_check["timestamp"] < event_window_check["effective_start"])
    | (
        event_window_check["timestamp"]
        > event_window_check["effective_end"]
    )
].copy()


missing_window_matches = event_window_check[
    event_window_check["effective_start"].isna()
    | event_window_check["effective_end"].isna()
].copy()


print("\n========================================")
print("EVENT / WINDOW MATCH CHECK")
print("========================================")

print(
    "Events with no matching cleaned window:",
    len(missing_window_matches),
)

print(
    "Events outside their cleaned window:",
    len(events_outside_windows),
)


# -----------------------------------------------------------------------------
# 14.4 Reconcile final analytical totals
# -----------------------------------------------------------------------------

final_drive_orders = len(
    analysis_drive_events
)

final_counter_event_rows = len(
    analysis_counter_events
)

final_counter_orders = int(
    analysis_counter_events[
        "corrected_counted"
    ].sum()
)

final_window_count = len(
    analysis_windows
)


print("\n========================================")
print("FINAL RECONCILIATION")
print("========================================")

print(
    "Drive-thru orders:",
    final_drive_orders,
)

print(
    "Counter order-event rows:",
    final_counter_event_rows,
)

print(
    "Corrected counter orders:",
    final_counter_orders,
)

print(
    "Observation windows:",
    final_window_count,
)

print("\nObserved hours by mode:")

print(
    (
        analysis_windows
        .groupby("mode")["duration_minutes"]
        .sum()
        / 60
    ).round(2)
)

# =============================================================================
# 15. INSPECT FINAL EDGE CASES
# =============================================================================

# The final integrity checks identified:
#
# - possible duplicate drive-thru rows
# - cleaned windows with zero duration
#
# These are not automatically errors.
#
# Before removing or modifying anything, inspect the underlying events
# and surrounding context.


# -----------------------------------------------------------------------------
# 15.1 Inspect possible duplicate drive-thru events
# -----------------------------------------------------------------------------

print("\n========================================")
print("POSSIBLE DRIVE-THRU DUPLICATES")
print("========================================")


if drive_duplicates.empty:

    print("None")

else:

    duplicate_columns = [
        column
        for column in [
            "timestamp",
            "source_row",
            "run_id",
            "mode",
            "kind",
            "car_number",
            "bags",
            "boxes",
            "text",
        ]
        if column in drive_duplicates.columns
    ]

    print(
        drive_duplicates[
            duplicate_columns
        ]
        .sort_values("timestamp")
        .to_string(index=False)
    )


# -----------------------------------------------------------------------------
# 15.2 Inspect zero-duration windows
# -----------------------------------------------------------------------------

zero_duration_windows = analysis_windows[
    analysis_windows["duration_minutes"] == 0
].copy()


print("\n========================================")
print("ZERO-DURATION WINDOW DETAILS")
print("========================================")


for _, window in zero_duration_windows.iterrows():

    source_row = window["source_row"]
    run_id = window["run_id"]
    mode = window["mode"]
    segment_id = window["clean_segment_id"]

    print("\n----------------------------------------")

    print(
        f"source_row={source_row}, "
        f"run_id={run_id}, "
        f"mode={mode}, "
        f"segment={segment_id}"
    )

    print(
        "Window time:",
        window["effective_start"],
    )


    # Show all raw logged events from the same historical run close
    # to the single-event observation.
    nearby_start = (
        window["effective_start"]
        - pd.Timedelta(minutes=10)
    )

    nearby_end = (
        window["effective_end"]
        + pd.Timedelta(minutes=10)
    )


    nearby_events = events_df[
        (events_df["source_row"] == source_row)
        & (events_df["run_id"] == run_id)
        & (events_df["timestamp"] >= nearby_start)
        & (events_df["timestamp"] <= nearby_end)
    ].copy()


    nearby_columns = [
        column
        for column in [
            "timestamp",
            "mode",
            "kind",
            "text",
            "car_number",
            "numbers",
            "counted",
        ]
        if column in nearby_events.columns
    ]


    print(
        nearby_events[
            nearby_columns
        ]
        .sort_values("timestamp")
        .to_string(index=False)
    )


# =============================================================================
# 16. CALCULATE OVERLAPPING / COMBINED OBSERVATION TIME
# =============================================================================

# Counter and drive-thru were often observed at the same time.
#
# Therefore:
#
#     counter hours + drive-thru hours
#
# is NOT the same thing as total clock time spent observing.
#
# We want to distinguish:
#
#     1. Drive-thru channel-hours
#     2. Counter channel-hours
#     3. Combo hours:
#        time when BOTH channels were observed simultaneously
#     4. Unique observation hours:
#        total clock time observed, counting overlapping time once
#     5. Drive-thru-only hours
#     6. Counter-only hours


# -----------------------------------------------------------------------------
# 16.1 Helper function: merge overlapping time windows
# -----------------------------------------------------------------------------

def merge_time_windows(window_df):

    # Return an empty list if there are no windows.
    if window_df.empty:
        return []


    # Sort windows chronologically.
    sorted_windows = (
        window_df[
            [
                "effective_start",
                "effective_end",
            ]
        ]
        .sort_values("effective_start")
        .to_records(index=False)
    )


    merged = []


    for start, end in sorted_windows:

        # First interval starts the merged list.
        if not merged:

            merged.append(
                [start, end]
            )

            continue


        previous_start, previous_end = merged[-1]


        # If this interval overlaps or touches the previous one,
        # extend the previous merged interval.
        if start <= previous_end:

            merged[-1][1] = max(
                previous_end,
                end,
            )


        else:

            # Otherwise begin a new independent interval.
            merged.append(
                [start, end]
            )


    return merged


# -----------------------------------------------------------------------------
# 16.2 Merge windows separately by mode
# -----------------------------------------------------------------------------

counter_windows_merged = merge_time_windows(
    analysis_windows[
        analysis_windows["mode"] == "counter"
    ]
)


drive_windows_merged = merge_time_windows(
    analysis_windows[
        analysis_windows["mode"] == "drive_thru"
    ]
)


# -----------------------------------------------------------------------------
# 16.3 Calculate total time represented by merged intervals
# -----------------------------------------------------------------------------

def total_interval_minutes(intervals):

    return sum(
        (
            end - start
        ).total_seconds() / 60
        for start, end in intervals
    )


counter_minutes_merged = total_interval_minutes(
    counter_windows_merged
)


drive_minutes_merged = total_interval_minutes(
    drive_windows_merged
)


# -----------------------------------------------------------------------------
# 16.4 Calculate overlap between counter and drive-thru
# -----------------------------------------------------------------------------

overlap_minutes = 0


for counter_start, counter_end in counter_windows_merged:

    for drive_start, drive_end in drive_windows_merged:

        # The overlap starts at the later of the two starts.
        overlap_start = max(
            counter_start,
            drive_start,
        )


        # The overlap ends at the earlier of the two ends.
        overlap_end = min(
            counter_end,
            drive_end,
        )


        # If overlap_end is after overlap_start,
        # both modes were being observed simultaneously.
        if overlap_end > overlap_start:

            overlap_minutes += (
                overlap_end - overlap_start
            ).total_seconds() / 60


# -----------------------------------------------------------------------------
# 16.5 Calculate exclusive and unique observation time
# -----------------------------------------------------------------------------

drive_only_minutes = (
    drive_minutes_merged
    - overlap_minutes
)


counter_only_minutes = (
    counter_minutes_merged
    - overlap_minutes
)

# Inclusion-exclusion:
#
# total unique time
# =
# drive time
# + counter time
# - overlapping time

unique_observation_minutes = (
    drive_minutes_merged
    + counter_minutes_merged
    - overlap_minutes
)


# -----------------------------------------------------------------------------
# 16.6 Display observation-time breakdown
# -----------------------------------------------------------------------------

print("\n========================================")
print("OBSERVATION TIME BREAKDOWN")
print("========================================")


print(
    "Drive-thru channel-hours:",
    round(
        drive_minutes_merged / 60,
        2,
    ),
)

print(
    "Counter channel-hours:",
    round(
        counter_minutes_merged / 60,
        2,
    ),
)

print(
    "Combo hours (DTL + counter simultaneously):",
    round(
        overlap_minutes / 60,
        2,
    ),
)


print(
    "Drive-thru-only hours:",
    round(
        drive_only_minutes / 60,
        2,
    ),
)

print(
    "Counter-only hours:",
    round(
        counter_only_minutes / 60,
        2,
    ),
)

print(
    "Unique clock-hours observed:",
    round(
        unique_observation_minutes / 60,
        2,
    ),
)
# =============================================================================
# 17. FINAL AUTHORITATIVE ANALYTICAL CLEANUP
# =============================================================================
#
# Everything before this section documents the discovery and cleaning
# process.
#
# From this section forward, the authoritative analytical objects are:
#
#     final_events
#     final_core_events
#     final_windows
#     final_rate_events
#
# Earlier DataFrames are retained for diagnostics and documentation,
# but analysis should use ONLY the final_* objects created here.
#
# IMPORTANT:
# The raw JSON and events_df are never overwritten.
# =============================================================================


# =============================================================================
# 17.1 CREATE A FRESH ANALYTICAL COPY
# =============================================================================

final_events = events_df.copy()


# Preserve the original historical run ID.
final_events["original_run_id"] = final_events["run_id"]


# Create a working run ID that can be corrected without touching
# the raw value.
final_events["run_id_clean"] = final_events["run_id"]


# Flags used during cleaning.
final_events["exclude_from_throughput"] = False
final_events["exclude_from_rate"] = False
final_events["analysis_reason"] = ""


# -----------------------------------------------------------------------------
# Remove the known August 19 development/test export from analysis.
# -----------------------------------------------------------------------------

final_events.loc[
    final_events["source_row"] == 3,
    "exclude_from_throughput"
] = True

final_events.loc[
    final_events["source_row"] == 3,
    "exclude_from_rate"
] = True

final_events.loc[
    final_events["source_row"] == 3,
    "analysis_reason"
] = "Known logger test/development export"


# =============================================================================
# 17.2 BUILD CLEAN TIMER EVENTS
# =============================================================================
#
# A critical discovery was that:
#
#     mode
#
# records which screen/tab was active when a timer button was tapped.
#
# It does NOT necessarily describe which run the timer belongs to.
#
# The correct field for timer identity is:
#
#     run_mode
#
# However, once simultaneous counter + drive-thru collection began,
# the timer is best treated as the observation timer for the RUN.
#
# Therefore:
#
#     - run_mode is used to identify what the timer itself represents
#     - timer boundaries are attached to the run_id
#     - both channels may use that run's timer when they were genuinely
#       being observed during the same run
# =============================================================================


timer_events_final = final_events[
    final_events["kind"].isin(
        [
            "timer_start",
            "timer_end",
        ]
    )
].copy()


# When run_mode is unavailable in older records, fall back to mode.
timer_events_final["timer_run_mode"] = (
    timer_events_final["run_mode"]
    .fillna(
        timer_events_final["mode"]
    )
)


# Some late auto-end logger bugs produced negative timer durations.
# Those timer_end records cannot represent valid elapsed time.
timer_events_final["duration_sec_numeric"] = pd.to_numeric(
    timer_events_final["duration_sec"],
    errors="coerce",
)


invalid_negative_timer = (
    (timer_events_final["kind"] == "timer_end")
    & (
        timer_events_final[
            "duration_sec_numeric"
        ] < 0
    )
)


timer_events_final = timer_events_final[
    ~invalid_negative_timer
].copy()


# Some automatic timer endings were written more than once.
# Keep one copy of an identical timer event.
timer_events_final = (
    timer_events_final
    .drop_duplicates(
        subset=[
            "source_row",
            "run_id_clean",
            "kind",
            "timestamp",
            "timer_run_mode",
        ],
        keep="first",
    )
    .copy()
)


# =============================================================================
# 17.3 REASSIGN STRAY EVENTS THAT BELONG TO THE NEXT RUN
# =============================================================================
#
# Several throughput events were logged just before the next timer was
# started, even though their historical run_id still pointed to the
# preceding run.
#
# General rule:
#
# If:
#
#     1. an event occurs AFTER the valid timer_end of its historical run
#     2. the next timer_start for the SAME CHANNEL occurs within 2 minutes
#
# then assign the event to that next run.
#
# This handles cases such as:
#
#     counter event -> timer starts one second later
#     car marker     -> new drive-thru timer starts seconds later
#
# without manually hard-coding every one.
# =============================================================================


AUTO_REASSIGN_LIMIT = pd.Timedelta(
    minutes=2
)


core_candidate_mask = (
    final_events["kind"].isin(
        [
            "car",
            "order",
        ]
    )
    & (
        final_events["source_row"] != 3
    )
)


auto_reassignments = []


for event_index, event in final_events[
    core_candidate_mask
].iterrows():

    source_row = event["source_row"]
    current_run = event["run_id_clean"]
    event_mode = event["mode"]
    event_time = event["timestamp"]


    # ---------------------------------------------------------
    # Find timer endings belonging to the current historical run.
    # ---------------------------------------------------------

    current_run_ends = timer_events_final[
        (timer_events_final["source_row"] == source_row)
        & (
            timer_events_final["run_id_clean"]
            == current_run
        )
        & (
            timer_events_final["kind"]
            == "timer_end"
        )
        & (
            timer_events_final["timestamp"]
            < event_time
        )
    ]


    # If the current run did not clearly end before this event,
    # do not automatically move it.
    if current_run_ends.empty:

        continue


    last_current_end = (
        current_run_ends[
            "timestamp"
        ].max()
    )


    # ---------------------------------------------------------
    # Find the next timer start for the same observation channel.
    # ---------------------------------------------------------

    next_starts = timer_events_final[
        (timer_events_final["source_row"] == source_row)
        & (
            timer_events_final["kind"]
            == "timer_start"
        )
        & (
            timer_events_final["timer_run_mode"]
            == event_mode
        )
        & (
            timer_events_final["timestamp"]
            > event_time
        )
        & (
            timer_events_final["timestamp"]
            <= event_time
            + AUTO_REASSIGN_LIMIT
        )
    ].sort_values(
        "timestamp"
    )


    if next_starts.empty:

        continue


    next_timer = next_starts.iloc[0]
    new_run = next_timer["run_id_clean"]


    # Avoid changing the event if the next timer happens to have
    # the same run ID.
    if new_run == current_run:

        continue


    final_events.at[
        event_index,
        "run_id_clean"
    ] = new_run


    final_events.at[
        event_index,
        "analysis_reason"
    ] = (
        "Reassigned to next run because event occurred "
        "after prior timer_end and immediately before "
        "next same-mode timer_start"
    )


    auto_reassignments.append(
        {
            "timestamp": event_time,
            "mode": event_mode,
            "old_run": current_run,
            "new_run": new_run,
        }
    )


# -----------------------------------------------------------------------------
# Two additional known cases had malformed timer metadata and therefore
# cannot be repaired by the automatic rule above.
# -----------------------------------------------------------------------------

manual_run_reassignments = [
    {
        "timestamp": pd.Timestamp(
            "2026-09-05T18:13:49-07:00"
        ),
        "mode": "counter",
        "new_run": 60,
    },
    {
        "timestamp": pd.Timestamp(
            "2026-09-06T18:24:44-07:00"
        ),
        "mode": "drive_thru",
        "new_run": 64,
    },
]


for correction in manual_run_reassignments:

    mask = (
        (final_events["source_row"] == 4)
        & (
            final_events["timestamp"]
            == correction["timestamp"]
        )
        & (
            final_events["mode"]
            == correction["mode"]
        )
        & (
            final_events["kind"].isin(
                [
                    "car",
                    "order",
                ]
            )
        )
    )


    final_events.loc[
        mask,
        "run_id_clean"
    ] = correction["new_run"]


    final_events.loc[
        mask,
        "analysis_reason"
    ] = (
        "Manual run reassignment based on adjacent "
        "timer/session evidence"
    )


# =============================================================================
# 17.4 APPLY DOCUMENTED MANUAL EVENT CORRECTIONS
# =============================================================================
#
# These corrections come directly from notes made during field
# observation.
#
# Raw values remain untouched in events_df.
# =============================================================================


# -----------------------------------------------------------------------------
# 17.4.1 Counter order-number correction:
#
# Logged:
#     ORDERS 10, 12
#
# Note:
#     "12 was 11.. fat fingered it"
#
# Correct analytical value:
#     ORDERS 10, 11
# -----------------------------------------------------------------------------

mask = (
    (final_events["source_row"] == 4)
    & (final_events["seq"] == 1964)
)

final_events.loc[
    mask,
    "numbers"
] = final_events.loc[
    mask,
    "numbers"
].apply(
    lambda _: ["10", "11"]
)

final_events.loc[
    mask,
    "analysis_reason"
] = (
    "Corrected order number 12 to 11 from field note"
)


# The later ORDER 11 was entered only to compensate for the earlier
# mistyped 12. Once the original event is corrected, this compensation
# entry must not create another transaction.

mask = (
    (final_events["source_row"] == 4)
    & (final_events["seq"] == 1971)
)

final_events.loc[
    mask,
    "exclude_from_throughput"
] = True

final_events.loc[
    mask,
    "exclude_from_rate"
] = True

final_events.loc[
    mask,
    "analysis_reason"
] = (
    "Compensating ORDER 11 entry removed after correcting seq 1964"
)


# -----------------------------------------------------------------------------
# 17.4.2 Announced order 35 was corrected verbally to order 55.
# -----------------------------------------------------------------------------

mask = (
    (final_events["source_row"] == 4)
    & (final_events["seq"] == 1810)
)

final_events.loc[
    mask,
    "numbers"
] = final_events.loc[
    mask,
    "numbers"
].apply(
    lambda _: ["55"]
)

final_events.loc[
    mask,
    "analysis_reason"
] = (
    "Order 35 corrected to 55 from field note"
)


# -----------------------------------------------------------------------------
# 17.4.3 Explicit repeat overrides.
#
# Field notes explicitly state that these were NOT repeat orders:
#
#     seq 940  -> order 76
#     seq 1701 -> order 65
#
# These overrides are applied during repeat recalculation below.
# -----------------------------------------------------------------------------

FORCE_NEW_ORDER_NUMBERS = {
    940: {"76"},
    1701: {"65"},
}


# =============================================================================
# 17.5 REMOVE CLEAR ACCIDENTAL DRIVE-THRU DOUBLE TAPS
# =============================================================================


# -----------------------------------------------------------------------------
# Exact same-timestamp car pairs
# -----------------------------------------------------------------------------
#
# Earlier diagnostics found two pairs where two car events were
# recorded at the exact same timestamp in the same run.
#
# Keep the first event and exclude the second analytical tap.
# The raw records remain untouched.

car_candidates = final_events[
    (final_events["kind"] == "car")
    & (
        ~final_events[
            "exclude_from_throughput"
        ]
    )
].copy()


same_timestamp_car_duplicate = (
    car_candidates
    .duplicated(
        subset=[
            "source_row",
            "run_id_clean",
            "timestamp",
            "mode",
            "kind",
        ],
        keep="first",
    )
)


duplicate_indexes = car_candidates[
    same_timestamp_car_duplicate
].index


final_events.loc[
    duplicate_indexes,
    "exclude_from_throughput"
] = True

final_events.loc[
    duplicate_indexes,
    "exclude_from_rate"
] = True

final_events.loc[
    duplicate_indexes,
    "analysis_reason"
] = (
    "Excluded second car tap at identical timestamp"
)


# -----------------------------------------------------------------------------
# Car 1318 was explicitly identified in the field notes as the
# accidental second tap of a double-car entry.
# -----------------------------------------------------------------------------

mask = (
    (final_events["source_row"] == 4)
    & (
        final_events["car_number"]
        == 1318
    )
)

final_events.loc[
    mask,
    "exclude_from_throughput"
] = True

final_events.loc[
    mask,
    "exclude_from_rate"
] = True

final_events.loc[
    mask,
    "analysis_reason"
] = (
    "Explicitly documented accidental double-car tap"
)


# =============================================================================
# 17.6 PRESERVE BUT EXCLUDE TWO ORPHAN EVENTS FROM RATE CALCULATIONS
# =============================================================================
#
# These events are legitimate timestamped observations, so they remain
# in the observed-event dataset.
#
# However, they belong to abandoned / isolated timer starts and provide
# effectively no measurable exposure time.
#
# Therefore:
#
#     KEEP for descriptive transaction counts
#     EXCLUDE from rate calculations
# =============================================================================

orphan_rate_mask = (
    (final_events["source_row"] == 4)
    & (
        final_events["car_number"].isin(
            [
                1011,
                1284,
            ]
        )
    )
)


final_events.loc[
    orphan_rate_mask,
    "exclude_from_rate"
] = True

final_events.loc[
    orphan_rate_mask,
    "analysis_reason"
] = (
    "Valid isolated observation retained for count "
    "but excluded from rate denominator"
)


# =============================================================================
# 17.7 REBUILD FINAL COUNTER COUNTS
# =============================================================================

REPEAT_WINDOW_FINAL = pd.Timedelta(
    minutes=30
)


final_counter_events = final_events[
    (final_events["kind"] == "order")
    & (
        ~final_events[
            "exclude_from_throughput"
        ]
    )
].copy()


final_counter_events = (
    final_counter_events
    .sort_values(
        "timestamp"
    )
    .copy()
)


last_seen_order = {}
final_corrected_counts = []


for _, row in final_counter_events.iterrows():

    timestamp = row["timestamp"]
    numbers = row.get("numbers")
    seq = row.get("seq")


    # Normalize the stored order-number field into a Python list.
    if isinstance(numbers, list):

        number_list = numbers

    elif isinstance(numbers, tuple):

        number_list = list(numbers)

    elif hasattr(
        numbers,
        "tolist",
    ):

        number_list = numbers.tolist()

    else:

        number_list = []


    # Normalize number type so "65" and 65 compare identically.
    number_list = [
        str(number)
        for number in number_list
    ]


    # Unannounced order:
    # no order number is available for repeat matching.
    if len(number_list) == 0:

        final_corrected_counts.append(1)

        continue


    event_count = 0


    # Retrieve any explicit manual repeat override.
    if pd.notna(seq):

        seq_key = int(seq)

    else:

        seq_key = None


    forced_new = (
        FORCE_NEW_ORDER_NUMBERS.get(
            seq_key,
            set(),
        )
    )


    for order_number in number_list:

        previous_time = (
            last_seen_order.get(
                order_number
            )
        )


        # Explicit field-note correction overrides automatic
        # repeat detection.
        if order_number in forced_new:

            is_new = True


        elif previous_time is None:

            is_new = True


        elif (
            timestamp - previous_time
            > REPEAT_WINDOW_FINAL
        ):

            is_new = True


        else:

            is_new = False


        if is_new:

            event_count += 1


        last_seen_order[
            order_number
        ] = timestamp


    final_corrected_counts.append(
        event_count
    )


final_counter_events[
    "corrected_counted"
] = final_corrected_counts


# Write the corrected counts back to final_events.
final_events[
    "final_corrected_counted"
] = pd.NA


final_events.loc[
    final_counter_events.index,
    "final_corrected_counted"
] = final_counter_events[
    "corrected_counted"
]


# =============================================================================
# 17.8 BUILD FINAL THROUGHPUT EVENT TABLE
# =============================================================================


final_drive_events = final_events[
    (final_events["kind"] == "car")
    & (
        ~final_events[
            "exclude_from_throughput"
        ]
    )
].copy()


final_counter_events = final_events[
    (final_events["kind"] == "order")
    & (
        ~final_events[
            "exclude_from_throughput"
        ]
    )
].copy()


# Each drive-thru car represents one transaction.
final_drive_events[
    "throughput_count"
] = 1


# Counter rows may contain several simultaneous completed orders.
final_counter_events[
    "throughput_count"
] = pd.to_numeric(
    final_counter_events[
        "final_corrected_counted"
    ],
    errors="coerce",
).fillna(0)


final_core_events = pd.concat(
    [
        final_drive_events,
        final_counter_events,
    ],
    ignore_index=False,
).sort_values(
    "timestamp"
).copy()


# =============================================================================
# 17.9 REBUILD CLEAN SESSION SEGMENTS
# =============================================================================

DRIVE_THRU_FINAL_GAP = pd.Timedelta(
    minutes=10
)

COUNTER_FINAL_GAP = pd.Timedelta(
    minutes=20
)


final_core_events[
    "final_segment_id"
] = 0


for group_key, group in final_core_events.groupby(
    [
        "source_row",
        "run_id_clean",
        "mode",
    ],
    dropna=False,
):

    source_row, run_id, mode = group_key


    group = group.sort_values(
        "timestamp"
    )


    current_segment = 1
    segment_numbers = []


    previous_time = None


    for timestamp in group[
        "timestamp"
    ]:

        if previous_time is None:

            segment_numbers.append(
                current_segment
            )

            previous_time = timestamp

            continue


        gap = (
            timestamp
            - previous_time
        )


        if mode == "drive_thru":

            if gap > DRIVE_THRU_FINAL_GAP:

                current_segment += 1


        elif mode == "counter":

            if gap > COUNTER_FINAL_GAP:

                current_segment += 1


        segment_numbers.append(
            current_segment
        )


        previous_time = timestamp


    final_core_events.loc[
        group.index,
        "final_segment_id"
    ] = segment_numbers


final_core_events[
    "final_segment_id"
] = final_core_events[
    "final_segment_id"
].astype(int)


# =============================================================================
# 17.10 BUILD FINAL RATE-SAFE OBSERVATION WINDOWS
# =============================================================================

START_ALLOWANCE_FINAL = pd.Timedelta(
    minutes=8
)

END_ALLOWANCE_FINAL = pd.Timedelta(
    minutes=10
)


# Manual periods where drive-thru observation explicitly stopped.
#
# Counter observation during the August 30 bathroom break is not
# removed because the field note explicitly states that counter calls
# remained audible from inside.

manual_breaks = [
    {
        "source_row": 4,
        "run_id_clean": 13,
        "mode": "drive_thru",
        "start": pd.Timestamp(
            "2026-08-22T00:03:11-07:00"
        ),
        "end": pd.Timestamp(
            "2026-08-22T00:05:15-07:00"
        ),
        "reason": "Observer went inside; DTL not watched",
    },
    {
        "source_row": 4,
        "run_id_clean": 49,
        "mode": "drive_thru",
        "start": pd.Timestamp(
            "2026-08-30T13:18:42-07:00"
        ),
        "end": pd.Timestamp(
            "2026-08-30T13:20:34-07:00"
        ),
        "reason": "Bathroom break; DTL not watched",
    },
]


def calculate_break_overlap(
    source_row,
    run_id,
    mode,
    window_start,
    window_end,
):

    overlap_seconds = 0


    for break_item in manual_breaks:

        if (
            break_item["source_row"]
            != source_row
        ):

            continue


        if (
            break_item["run_id_clean"]
            != run_id
        ):

            continue


        if (
            break_item["mode"]
            != mode
        ):

            continue


        overlap_start = max(
            window_start,
            break_item["start"],
        )


        overlap_end = min(
            window_end,
            break_item["end"],
        )


        if overlap_end > overlap_start:

            overlap_seconds += (
                overlap_end
                - overlap_start
            ).total_seconds()


    return overlap_seconds / 60


# -----------------------------------------------------------------------------
# Determine whether each segment ends because another cleaned segment
# follows in the same run/mode group.
# -----------------------------------------------------------------------------

final_segment_summary = (
    final_core_events
    .groupby(
        [
            "source_row",
            "run_id_clean",
            "mode",
            "final_segment_id",
        ]
    )
    .agg(
        first_event=(
            "timestamp",
            "min",
        ),
        last_event=(
            "timestamp",
            "max",
        ),
        event_rows=(
            "timestamp",
            "size",
        ),
    )
    .reset_index()
)


final_segment_summary[
    "max_segment_id"
] = (
    final_segment_summary
    .groupby(
        [
            "source_row",
            "run_id_clean",
            "mode",
        ]
    )[
        "final_segment_id"
    ]
    .transform("max")
)


final_segment_summary[
    "ended_by_gap_split"
] = (
    final_segment_summary[
        "final_segment_id"
    ]
    < final_segment_summary[
        "max_segment_id"
    ]
)


final_windows_list = []


for _, segment in (
    final_segment_summary.iterrows()
):

    source_row = segment[
        "source_row"
    ]

    run_id = segment[
        "run_id_clean"
    ]

    mode = segment[
        "mode"
    ]

    segment_id = segment[
        "final_segment_id"
    ]

    first_event = segment[
        "first_event"
    ]

    last_event = segment[
        "last_event"
    ]

    ended_by_gap = segment[
        "ended_by_gap_split"
    ]


    effective_start = first_event
    effective_end = last_event

    start_source = "first_event"
    end_source = "last_event"


    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Timer lookup is by RUN, not by active-screen mode.
    #
    # This deliberately fixes the earlier timer-assignment flaw.
    # ---------------------------------------------------------

    run_timers = timer_events_final[
        (timer_events_final["source_row"] == source_row)
        & (
            timer_events_final["run_id_clean"]
            == run_id
        )
    ].copy()


    timer_starts = run_timers[
        run_timers["kind"]
        == "timer_start"
    ]["timestamp"]


    timer_ends = run_timers[
        run_timers["kind"]
        == "timer_end"
    ]["timestamp"]


    # ---------------------------------------------------------
    # Start boundary
    # ---------------------------------------------------------

    preceding_starts = timer_starts[
        timer_starts
        <= first_event
    ]


    if not preceding_starts.empty:

        closest_start = (
            preceding_starts.max()
        )


        if (
            first_event
            - closest_start
            <= START_ALLOWANCE_FINAL
        ):

            effective_start = (
                closest_start
            )

            start_source = "timer"


    # ---------------------------------------------------------
    # End boundary
    # ---------------------------------------------------------

    valid_ends_after_start = timer_ends[
        timer_ends
        >= effective_start
    ]


    # If a valid timer_end occurred BEFORE the final event,
    # do not allow stray later taps to stretch exposure.
    ends_before_last = (
        valid_ends_after_start[
            valid_ends_after_start
            < last_event
        ]
    )


    if not ends_before_last.empty:

        closest_prior_end = (
            ends_before_last.max()
        )

        effective_end = (
            closest_prior_end
        )

        end_source = (
            "timer_cap_before_last"
        )


    elif ended_by_gap:

        effective_end = last_event
        end_source = "gap_cap"


    else:

        following_ends = (
            valid_ends_after_start[
                valid_ends_after_start
                >= last_event
            ]
        )


        if not following_ends.empty:

            closest_end = (
                following_ends.min()
            )


            if (
                closest_end
                - last_event
                <= END_ALLOWANCE_FINAL
            ):

                effective_end = (
                    closest_end
                )

                end_source = "timer"


    raw_minutes = (
        effective_end
        - effective_start
    ).total_seconds() / 60


    # Negative or zero durations cannot support a rate estimate.
    if raw_minutes > 0:

        break_minutes = (
            calculate_break_overlap(
                source_row,
                run_id,
                mode,
                effective_start,
                effective_end,
            )
        )

    else:

        break_minutes = 0


    active_minutes = max(
        0,
        raw_minutes
        - break_minutes,
    )


    # Boundary-quality label for later sensitivity analysis.
    if (
        start_source == "timer"
        and end_source == "timer"
    ):

        boundary_quality = (
            "full_timer"
        )

    elif (
        "timer" in start_source
        or "timer" in end_source
    ):

        boundary_quality = (
            "partial_timer"
        )

    else:

        boundary_quality = (
            "event_bounded"
        )


    final_windows_list.append(
        {
            "source_row": source_row,
            "run_id_clean": run_id,
            "mode": mode,
            "final_segment_id": segment_id,
            "first_event": first_event,
            "last_event": last_event,
            "effective_start": effective_start,
            "effective_end": effective_end,
            "start_source": start_source,
            "end_source": end_source,
            "break_minutes": break_minutes,
            "raw_minutes": raw_minutes,
            "active_minutes": active_minutes,
            "boundary_quality": boundary_quality,
        }
    )


final_windows = pd.DataFrame(
    final_windows_list
)


# Give every analytical window a stable ID.
final_windows[
    "window_id"
] = range(
    1,
    len(final_windows) + 1,
)


# =============================================================================
# 17.11 ASSIGN EVENTS TO FINAL WINDOWS
# =============================================================================


final_event_window_check = (
    final_core_events
    .merge(
        final_windows[
            [
                "source_row",
                "run_id_clean",
                "mode",
                "final_segment_id",
                "window_id",
                "effective_start",
                "effective_end",
                "active_minutes",
                "boundary_quality",
            ]
        ],
        on=[
            "source_row",
            "run_id_clean",
            "mode",
            "final_segment_id",
        ],
        how="left",
        validate="many_to_one",
    )
)


# Event is rate-eligible only when:
#
#     - its cleaned window has positive active time
#     - timestamp falls inside the exposure interval
#     - it was not manually excluded from rates

final_event_window_check[
    "rate_eligible"
] = (
    (
        final_event_window_check[
            "active_minutes"
        ] > 0
    )
    & (
        final_event_window_check[
            "timestamp"
        ]
        >= final_event_window_check[
            "effective_start"
        ]
    )
    & (
        final_event_window_check[
            "timestamp"
        ]
        <= final_event_window_check[
            "effective_end"
        ]
    )
    & (
        ~final_event_window_check[
            "exclude_from_rate"
        ]
    )
)


# Only actual NEW transactions contribute to throughput rates.
final_rate_events = (
    final_event_window_check[
        final_event_window_check[
            "rate_eligible"
        ]
        & (
            final_event_window_check[
                "throughput_count"
            ] > 0
        )
    ]
    .copy()
)


# =============================================================================
# 17.12 ADD TRANSACTION COUNTS TO FINAL WINDOWS
# =============================================================================


window_rate_counts = (
    final_rate_events
    .groupby(
        "window_id"
    )[
        "throughput_count"
    ]
    .sum()
    .rename(
        "rate_transaction_count"
    )
)


final_windows = (
    final_windows
    .merge(
        window_rate_counts,
        on="window_id",
        how="left",
    )
)


final_windows[
    "rate_transaction_count"
] = (
    final_windows[
        "rate_transaction_count"
    ]
    .fillna(0)
)


# Rate is calculated only when there is positive active exposure.
final_windows[
    "transactions_per_hour"
] = pd.NA


positive_window_mask = (
    final_windows[
        "active_minutes"
    ] > 0
)


final_windows.loc[
    positive_window_mask,
    "transactions_per_hour"
] = (
    final_windows.loc[
        positive_window_mask,
        "rate_transaction_count",
    ]
    /
    (
        final_windows.loc[
            positive_window_mask,
            "active_minutes",
        ]
        / 60
    )
)


# =============================================================================
# 17.13 FINAL VALIDATION
# =============================================================================


final_drive_order_count = len(
    final_drive_events
)


final_counter_order_count = int(
    pd.to_numeric(
        final_counter_events[
            "final_corrected_counted"
        ],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)


zero_final_windows = final_windows[
    final_windows[
        "active_minutes"
    ] <= 0
]


print("\n========================================")
print("FINAL AUTHORITATIVE CLEAN DATA")
print("========================================")


print(
    "Drive-thru transactions retained:",
    final_drive_order_count,
)


print(
    "Corrected counter transactions retained:",
    final_counter_order_count,
)


print(
    "Final analytical windows:",
    len(final_windows),
)


print(
    "Zero-duration windows retained for descriptive "
    "purposes but excluded from rates:",
    len(zero_final_windows),
)


print(
    "Rate-eligible transaction rows:",
    len(final_rate_events),
)


print(
    "Automatic run reassignments:",
    len(auto_reassignments),
)


print(
    "Exact-timestamp duplicate car taps excluded:",
    len(duplicate_indexes),
)


print("\nFinal active channel-hours:")

print(
    (
        final_windows
        .groupby(
            "mode"
        )[
            "active_minutes"
        ]
        .sum()
        / 60
    ).round(2)
)


print("\nBoundary quality:")

print(
    final_windows[
        "boundary_quality"
    ].value_counts()
)


print("\nEnd boundary source:")

print(
    final_windows[
        "end_source"
    ].value_counts()
)


# =============================================================================
# 17.14 SAVE FINAL CLEAN DATASETS
# =============================================================================
#
# These are NEW files.
#
# The original JSON is never modified or overwritten.
# =============================================================================


final_events.to_csv(
    "innout_events_final_clean.csv",
    index=False,
)


final_core_events.to_csv(
    "innout_throughput_final_clean.csv",
    index=False,
)


final_windows.to_csv(
    "innout_windows_final_clean.csv",
    index=False,
)


final_rate_events.to_csv(
    "innout_rate_events_final_clean.csv",
    index=False,
)


print("\n========================================")
print("FINAL FILES SAVED")
print("========================================")

print(
    "innout_events_final_clean.csv"
)

print(
    "innout_throughput_final_clean.csv"
)

print(
    "innout_windows_final_clean.csv"
)

print(
    "innout_rate_events_final_clean.csv"
)
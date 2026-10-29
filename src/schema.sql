-- stop_events: one row per scheduled bus stop event, from either source.
--   source = 'historical'  MBTA Bus Arrival/Departure CSVs (timepoints only, has actual times)
--   source = 'api'         MBTA V3 /schedules daily pull (timepoints kept, no actual times yet)
--
-- Both sources are stored in the files' conventions: file-style route ids ('01', 'SL1'),
-- direction as the word, and wall-clock Boston timestamps with no UTC offset.

CREATE TABLE IF NOT EXISTS stop_events (
    id                BIGSERIAL    PRIMARY KEY,
    source            TEXT         NOT NULL CHECK (source IN ('historical', 'api')),

    -- Shared by both sources. These five columns are the natural key.
    service_date      DATE         NOT NULL,
    route_id          TEXT         NOT NULL,  -- file style: '01', '66', 'SL1'; pull_daily maps API ids back to this
    direction_id      TEXT         NOT NULL CHECK (direction_id IN ('Inbound', 'Outbound')),  -- API 1 = Inbound, 0 = Outbound
    stop_id           TEXT         NOT NULL,
    scheduled         TIMESTAMP    NOT NULL,  -- wall clock; after-midnight trips fall on the next calendar day

    -- Outcome. NULL for API rows and for historical rows with no recorded actual.
    actual            TIMESTAMP,
    delay_min         DOUBLE PRECISION GENERATED ALWAYS AS (EXTRACT(EPOCH FROM actual - scheduled) / 60) STORED,
    late              BOOLEAN          GENERATED ALWAYS AS (actual - scheduled > INTERVAL '3 minutes') STORED,

    -- Historical-only columns (NULL for API rows: the API has no answer for them).
    half_trip_id      TEXT,
    time_point_id     TEXT,
    time_point_order  SMALLINT,
    point_type        TEXT,      -- Startpoint / Midpoint / Endpoint
    standard_type     TEXT,      -- Schedule / Headway
    scheduled_headway INTEGER,   -- seconds
    headway           INTEGER,   -- seconds, observed during the trip: never a feature

    -- API-only columns (NULL for historical rows).
    trip_id           TEXT,
    stop_sequence     SMALLINT,

    loaded_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- Delays outside -30..+120 min are data errors; the loader drops them, this is the backstop.
    CONSTRAINT stop_events_delay_range CHECK (delay_min IS NULL OR delay_min BETWEEN -30 AND 120),
    CONSTRAINT stop_events_key UNIQUE (service_date, route_id, direction_id, stop_id, scheduled)
);

CREATE INDEX IF NOT EXISTS stop_events_source_date ON stop_events (source, service_date);

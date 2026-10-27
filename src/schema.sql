-- stop_events: one row per scheduled bus stop event, from either source.
--   source = 'historical'  MBTA Bus Arrival/Departure CSVs (timepoints only, has actual times)
--   source = 'api'         MBTA V3 /schedules daily pull (every stop, no actual times)

CREATE TABLE IF NOT EXISTS stop_events (
    id                BIGSERIAL    PRIMARY KEY,
    source            TEXT         NOT NULL CHECK (source IN ('historical', 'api')),

    -- Shared by both sources. These five columns are the natural key.
    service_date      DATE         NOT NULL,
    route_id          TEXT         NOT NULL,  -- leading zeros stripped ('01' -> '1'), as the API spells it
    direction_id      SMALLINT     NOT NULL CHECK (direction_id IN (0, 1)),  -- 0 = Outbound, 1 = Inbound
    stop_id           TEXT         NOT NULL,
    scheduled         TIMESTAMPTZ  NOT NULL,  -- real timestamp, Boston time; past-midnight trips land on the next day

    -- Outcome. NULL for API rows and for historical rows with no recorded actual.
    actual            TIMESTAMPTZ,
    delay_min         DOUBLE PRECISION GENERATED ALWAYS AS (EXTRACT(EPOCH FROM actual - scheduled) / 60) STORED,
    late              BOOLEAN          GENERATED ALWAYS AS (actual - scheduled > INTERVAL '3 minutes') STORED,

    -- Historical-only columns (NULL for API rows).
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
    timepoint         BOOLEAN,

    loaded_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- Delays outside -30..+120 min are data errors; the loader drops them, this is the backstop.
    CONSTRAINT stop_events_delay_range CHECK (delay_min IS NULL OR delay_min BETWEEN -30 AND 120),
    CONSTRAINT stop_events_key UNIQUE (service_date, route_id, direction_id, stop_id, scheduled)
);

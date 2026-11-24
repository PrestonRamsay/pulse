-- stop_events: one row per scheduled bus stop event, from any of three sources.
--   source = 'historical'  MBTA Bus Arrival/Departure CSVs (timepoints only, has actual times)
--   source = 'lamp'        MBTA LAMP bus events, daily (timepoints, has actual times; src/pull_lamp.py)
--   source = 'api'         MBTA V3 /schedules daily pull (timepoints kept, no actual times yet)
--
-- For the same key, historical beats lamp beats api: a monthly file replaces LAMP or API rows
-- for its days, and LAMP fills in actual times on API rows. Training splits use historical only.
--
-- All sources are stored in the files' conventions: file-style route ids ('01', 'SL1'),
-- direction as the word, and wall-clock Boston timestamps with no UTC offset.

CREATE TABLE IF NOT EXISTS stop_events (
    id                BIGSERIAL    PRIMARY KEY,
    source            TEXT         NOT NULL CONSTRAINT stop_events_source_check
                                   CHECK (source IN ('historical', 'lamp', 'api')),

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

-- Migration for tables created before 'lamp' existed (a no-op afterwards): drop every older
-- check on source, whatever Postgres named it, then make sure the current one exists.
DO $$
DECLARE c record;
BEGIN
    FOR c IN SELECT conname FROM pg_constraint
             WHERE conrelid = 'stop_events'::regclass AND contype = 'c'
               AND pg_get_constraintdef(oid) LIKE '%source%'
               AND pg_get_constraintdef(oid) NOT LIKE '%lamp%' LOOP
        EXECUTE format('ALTER TABLE stop_events DROP CONSTRAINT %I', c.conname);
    END LOOP;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conrelid = 'stop_events'::regclass AND conname = 'stop_events_source_check') THEN
        ALTER TABLE stop_events ADD CONSTRAINT stop_events_source_check
            CHECK (source IN ('historical', 'lamp', 'api'));
    END IF;
END $$;

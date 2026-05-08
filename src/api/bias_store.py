from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class BiasCell:
    key: str
    lat_key: float
    lon_key: float


def bias_cell(lat: float, lon: float, precision: int = 3) -> BiasCell:
    lat_key = round(float(lat), precision)
    lon_key = round(float(lon), precision)
    return BiasCell(key=f"{lat_key:.{precision}f},{lon_key:.{precision}f}", lat_key=lat_key, lon_key=lon_key)


class BiasStore:
    """SQLite-backed storage for observations and bias models.

    - Observations: raw truth sources (accuweather/user/cpcb, etc.)
    - Bias models: EMA residual per cell, including hour-of-day variant.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS observations (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  source TEXT NOT NULL,
                  lat REAL NOT NULL,
                  lon REAL NOT NULL,
                  cell_key TEXT NOT NULL,
                  aqi REAL NOT NULL,
                  aqi_scale TEXT NOT NULL,
                  confidence REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_observations_cell_time
                  ON observations (cell_key, created_at DESC);

                CREATE TABLE IF NOT EXISTS bias_ema (
                  cell_key TEXT NOT NULL,
                  hour_of_day INTEGER NOT NULL,
                  ema_bias REAL NOT NULL,
                  samples INTEGER NOT NULL,
                  updated_at TEXT NOT NULL,
                  last_residual REAL,
                  PRIMARY KEY (cell_key, hour_of_day)
                );
                """
            )

    def insert_observation(
        self,
        source: str,
        lat: float,
        lon: float,
        aqi: float,
        aqi_scale: str,
        confidence: float = 0.8,
        created_at: str | None = None,
    ) -> dict:
        created_at = created_at or utc_now_iso()
        confidence = clamp(float(confidence), 0.05, 0.99)
        cell = bias_cell(lat, lon)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observations (created_at, source, lat, lon, cell_key, aqi, aqi_scale, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, source, float(lat), float(lon), cell.key, float(aqi), aqi_scale, confidence),
            )
        return {"status": "ok", "cell": cell.key, "created_at": created_at}

    def latest_observations(self, cell_key: str, limit: int = 8) -> list[dict]:
        limit = max(1, min(int(limit), 50))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, source, lat, lon, aqi, aqi_scale, confidence
                FROM observations
                WHERE cell_key = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (cell_key, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_bias(self, cell_key: str, hour_of_day: int) -> dict | None:
        hour_of_day = int(hour_of_day) % 24
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT cell_key, hour_of_day, ema_bias, samples, updated_at, last_residual
                FROM bias_ema
                WHERE cell_key = ? AND hour_of_day = ?
                """,
                (cell_key, hour_of_day),
            ).fetchone()
        return dict(row) if row else None

    def update_bias_ema(
        self,
        cell_key: str,
        hour_of_day: int,
        observed_now: float,
        predicted_now: float,
        confidence: float = 0.8,
        alpha_base: float = 0.35,
    ) -> dict:
        hour_of_day = int(hour_of_day) % 24
        confidence = clamp(float(confidence), 0.05, 0.99)
        alpha = clamp(alpha_base * (0.4 + confidence), 0.08, 0.55)
        residual = float(observed_now) - float(predicted_now)

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                """
                SELECT ema_bias, samples
                FROM bias_ema
                WHERE cell_key = ? AND hour_of_day = ?
                """,
                (cell_key, hour_of_day),
            ).fetchone()

            if existing:
                prev = float(existing["ema_bias"])
                samples = int(existing["samples"]) + 1
            else:
                prev = 0.0
                samples = 1

            ema = (1.0 - alpha) * prev + alpha * residual
            conn.execute(
                """
                INSERT INTO bias_ema (cell_key, hour_of_day, ema_bias, samples, updated_at, last_residual)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cell_key, hour_of_day) DO UPDATE SET
                  ema_bias=excluded.ema_bias,
                  samples=excluded.samples,
                  updated_at=excluded.updated_at,
                  last_residual=excluded.last_residual
                """,
                (cell_key, hour_of_day, float(ema), samples, utc_now_iso(), float(residual)),
            )
        return {
            "cell_key": cell_key,
            "hour_of_day": hour_of_day,
            "ema_bias": round(float(ema), 3),
            "samples": samples,
            "last_residual": round(float(residual), 3),
            "alpha": round(float(alpha), 3),
        }


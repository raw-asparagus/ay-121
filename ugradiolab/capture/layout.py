"""Naming helpers for streaming capture sessions and cells."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..astronomy import equatorial_to_galactic


def _timestamp_token(ts: float) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S_%f")


def round_galactic_coordinates(ra_deg: float, dec_deg: float) -> tuple[int, int]:
    """Convert ICRS coordinates to integer galactic coordinates."""
    gl_deg, gb_deg = equatorial_to_galactic(ra_deg, dec_deg)
    gl = int(round(float(gl_deg))) % 360
    gb = int(round(float(gb_deg)))
    return gl, gb


def make_session_id(start_time: float, index: int) -> str:
    """Return a timestamped, sequential session identifier."""
    return f"session_{_timestamp_token(start_time)}_{index:02d}"


def make_cell_name(kind: str, gl_deg: int, gb_deg: int) -> str:
    """Return the canonical cell name for a capture."""
    return f"{kind}_{gl_deg}_{gb_deg}"


def make_capture_name(kind: str, gl_deg: int, gb_deg: int, ts: float) -> str:
    """Return the capture filename stem for a dump."""
    return f"{kind}_{gl_deg}_{gb_deg}_{_timestamp_token(ts)}"


def annotate_streaming_dump(dump: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Add session and cell layout fields to a dump record."""
    gl_deg, gb_deg = round_galactic_coordinates(
        float(dump["ra_deg"]),
        float(dump["dec_deg"]),
    )
    cell_kind = "cal" if bool(dump.get("noise_on")) else "obs"
    cell_name = make_cell_name(cell_kind, gl_deg, gb_deg)
    capture_name = make_capture_name(cell_kind, gl_deg, gb_deg, float(dump["time"]))

    dump["session_id"] = session_id
    dump["gl"] = gl_deg
    dump["gb"] = gb_deg
    dump["cell_kind"] = cell_kind
    dump["cell_name"] = cell_name
    dump["capture_name"] = capture_name
    return dump


def build_session_capture_path(outdir: str | Path, dump: Mapping[str, Any]) -> str:
    """Build the nested session/cell path for a streaming dump."""
    session_id = dump["session_id"]
    cell_name = dump["cell_name"]
    capture_name = dump["capture_name"]
    return str(Path(outdir) / str(session_id) / str(cell_name) / f"{capture_name}.npz")
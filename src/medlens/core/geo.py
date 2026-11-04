"""Privacy-safe locations. Raw coordinates are never stored (CLAUDE.md §1): a point is snapped
to the centre of a 0.01-degree cell (about 1.1 km) and labelled with its geohash (precision 6,
about 1.2 km x 0.6 km), both computed from the snapped point."""

import math
import re
import struct
from dataclasses import dataclass

CELL_DEGREES = 0.01
GEOHASH_PRECISION = 6
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"  # geohash alphabet  # pragma: allowlist secret
_EWKB_SRID_FLAG = 0x20000000
_WKT_POINT = re.compile(r"POINT\s*\(\s*(?P<lng>[-\d.eE]+)\s+(?P<lat>[-\d.eE]+)\s*\)", re.I)


@dataclass(frozen=True, slots=True)
class CoarseLocation:
    latitude: float
    longitude: float
    geocell: str

    @property
    def wkt(self) -> str:
        return point_wkt(self.latitude, self.longitude)


def coarsen(latitude: float, longitude: float) -> CoarseLocation:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("latitude must be within -90..90 and longitude within -180..180")
    lat = _snap(latitude, 90)
    lng = _snap(longitude, 180)
    return CoarseLocation(lat, lng, geohash(lat, lng))


def _snap(value: float, bound: float) -> float:
    centre = (math.floor(value / CELL_DEGREES) + 0.5) * CELL_DEGREES
    return round(max(-bound, min(bound, centre)), 3)


def geohash(latitude: float, longitude: float, precision: int = GEOHASH_PRECISION) -> str:
    lat_range, lng_range = [-90.0, 90.0], [-180.0, 180.0]
    bits, value, even = 0, 0, True
    result: list[str] = []
    while len(result) < precision:
        span, coordinate = (lng_range, longitude) if even else (lat_range, latitude)
        middle = (span[0] + span[1]) / 2
        value <<= 1
        if coordinate >= middle:
            value |= 1
            span[0] = middle
        else:
            span[1] = middle
        even = not even
        bits += 1
        if bits == 5:
            result.append(_BASE32[value])
            bits, value = 0, 0
    return "".join(result)


def point_wkt(latitude: float, longitude: float) -> str:
    """EWKT for a geography(Point, 4326) column (longitude first)."""
    return f"SRID=4326;POINT({longitude} {latitude})"


def point_coordinates(element: object) -> tuple[float, float] | None:
    """(latitude, longitude) of a point: EWKB as PostGIS returns it, or the (E)WKT we wrote."""
    if element is None:
        return None
    data = getattr(element, "data", element)
    if isinstance(data, str) and (match := _WKT_POINT.search(data)):
        return float(match["lat"]), float(match["lng"])
    if isinstance(data, str):
        raw = bytes.fromhex(data)
    elif isinstance(data, bytes | bytearray | memoryview):
        raw = bytes(data)
    else:
        raise TypeError(f"cannot read a point from {type(data).__name__}")
    order = "<" if raw[0] == 1 else ">"
    (geometry_type,) = struct.unpack(f"{order}I", raw[1:5])
    offset = 9 if geometry_type & _EWKB_SRID_FLAG else 5
    if geometry_type & 0xFFFF != 1:
        raise ValueError("not a point")
    longitude, latitude = struct.unpack(f"{order}dd", raw[offset : offset + 16])
    return latitude, longitude

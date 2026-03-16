# API: Pointing

Source: `ugradiolab/pointing.py`

Coordinate conversion between galactic (l, b), ICRS equatorial (RA, Dec), and horizontal (alt, az) frames.

---

## Coordinate Frames

| Frame | Coordinates | Units |
|---|---|---|
| Galactic | longitude `l`, latitude `b` | degrees |
| ICRS equatorial | right ascension `ra`, declination `dec` | degrees |
| Horizontal | altitude `alt`, azimuth `az` | degrees |

Azimuth follows the North-East-Up convention: 0° = North, 90° = East.

---

## Angle Convention Warning

> **`lst_rad` is in radians.** All other angles in the public API (`l_deg`, `b_deg`, `ra_deg`, `dec_deg`, `alt_deg`, `az_deg`, `lat_deg`) are in degrees. This inconsistency exists because `ugradio.timing.lst(jd, lon)` returns radians and that value is passed through without conversion.

Quick check before calling:
```python
import ugradio.timing as timing
import ugradio.nch as nch

jd = timing.julian_date(timing.unix_time())
lst_rad = timing.lst(jd, nch.lon)   # radians — use directly in the functions below
```

---

## Six Scalar Coordinate Functions

These use the internal IAU 1958 / Hipparcos 3×3 rotation matrices. They work offline (no network, no clock).

| Function | Inputs | Outputs | Extra args |
|---|---|---|---|
| `galactic_to_equatorial(l_deg, b_deg)` | galactic | `(ra_deg, dec_deg)` | — |
| `equatorial_to_galactic(ra_deg, dec_deg)` | equatorial | `(l_deg, b_deg)` | — |
| `equatorial_to_altaz(ra_deg, dec_deg, lst_rad, lat_deg)` | equatorial | `(alt_deg, az_deg)` | `lst_rad` **radians**, `lat_deg` degrees |
| `altaz_to_equatorial(alt_deg, az_deg, lst_rad, lat_deg)` | horizontal | `(ra_deg, dec_deg)` | `lst_rad` **radians**, `lat_deg` degrees |
| `galactic_to_altaz(l_deg, b_deg, lst_rad, lat_deg)` | galactic | `(alt_deg, az_deg)` | `lst_rad` **radians**, `lat_deg` degrees |
| `altaz_to_galactic(alt_deg, az_deg, lst_rad, lat_deg)` | horizontal | `(l_deg, b_deg)` | `lst_rad` **radians**, `lat_deg` degrees |

All six functions take and return scalar floats.

### Example

```python
from ugradiolab import galactic_to_altaz, galactic_to_equatorial
import ugradio.timing as timing
import ugradio.nch as nch

jd = timing.julian_date(timing.unix_time())
lst_rad = timing.lst(jd, nch.lon)

# Convert galactic centre to horizontal
alt, az = galactic_to_altaz(0.0, 0.0, lst_rad=lst_rad, lat_deg=nch.lat)

# Convert galactic to ICRS (no time needed)
ra, dec = galactic_to_equatorial(120.0, 0.0)
```

---

## Matrix Functions

For batch transformations, access the underlying rotation matrices directly.

#### `galactic_to_equatorial_matrix()`

Returns a `(3, 3)` float64 copy of the `_GAL_TO_EQ` matrix (IAU 1958 / Hipparcos). Multiply against a unit vector in galactic coordinates to get an equatorial unit vector.

#### `equatorial_to_altaz_matrix(lst_rad, lat_deg)`

Returns a `(3, 3)` float64 rotation matrix from ICRS equatorial to North-East-Up horizontal.

| Parameter | Type | Units |
|---|---|---|
| `lst_rad` | `float` | **radians** |
| `lat_deg` | `float` | degrees |

Pipeline: `M_lat @ Rz(-LST)`, where `Rz(-LST)` converts RA to hour angle and `M_lat` converts HA-Dec to North-East-Up.

---

## `compute_pointing`

Live pointing computation. Converts galactic coordinates to alt/az and ICRS for the current time and observer location.

```python
compute_pointing(gal_l, gal_b, lat=nch.lat, lon=nch.lon, obs_alt=nch.alt)
```

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `gal_l` | `float` | required | degrees | Galactic longitude |
| `gal_b` | `float` | required | degrees | Galactic latitude |
| `lat` | `float` | `nch.lat` | degrees | Observer latitude |
| `lon` | `float` | `nch.lon` | degrees | Observer longitude |
| `obs_alt` | `float` | `nch.alt` | metres | Observer altitude |

**Returns** `(alt_deg, az_deg, ra_deg, dec_deg, jd)` — all floats.

**Side effects**: calls `get_unix_time()` (which may make an NTP request) and `ugradio.timing.julian_date`.

**Important**: `compute_pointing` uses `astropy.coordinates.SkyCoord` for the galactic→ICRS conversion, then `ugradio.coord.get_altaz` for ICRS→alt/az. It does **not** use the internal `_GAL_TO_EQ` Hipparcos matrix. The six scalar functions and `compute_pointing` may give slightly different RA/Dec values due to the different transformation implementations.

Default observer is NCH (Nancay–Campbell Hall, UC Berkeley), defined in `ugradio.nch`.

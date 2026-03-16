import ntplib
import os
import time
import ugradio.timing as timing


def make_path(outdir: str, prefix: str, tag: str) -> str:
    """Return a timestamped output filepath and create the directory."""
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(outdir, f'{prefix}_{tag}_{ts}.npz')


def get_unix_time(timeout: float = 2.0) -> float:
    """Return the current Unix time.

    Tries to fetch time from an NTP server first. If that fails for any reason,
    falls back to the local system clock.
    """
    client = ntplib.NTPClient()

    try:
        response = client.request("pool.ntp.org", version=3, timeout=timeout)
        return response.tx_time
    except (ntplib.NTPException, OSError):
        print("Unable to connect to NTP! Using system time.")
        return timing.unix_time()

import ntplib
import ugradio.timing as timing


def get_unix_time() -> float:
    """Return current Unix time from NTP, falling back to system time."""
    try:
        c = ntplib.NTPClient()
        return c.request('pool.ntp.org', version=3).tx_time
    except ntplib.NTPException:
        return timing.unix_time()

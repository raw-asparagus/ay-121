import ntplib
import ugradio.timing as timing


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

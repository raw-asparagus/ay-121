import ntplib
import ugradio.timing as timing


def get_unix_time(timeout: float = 2.0, local: bool = False) -> float:
    """Return the current Unix time.

    Parameters
    ----------
    timeout : float, optional
        Network timeout in seconds for the NTP request.
    local : bool, optional
        If ``True``, bypass NTP and use the local system clock immediately.

    Returns
    -------
    unix_time : float
        Current Unix timestamp in seconds.

    Notes
    -----
    ``ntplib.NTPException`` and ``OSError`` raised by the network lookup are
    caught and handled by falling back to the local system clock.
    """
    if local:
        return timing.unix_time()

    client = ntplib.NTPClient()

    try:
        response = client.request("pool.ntp.org", version=3, timeout=timeout)
        return response.tx_time
    except (ntplib.NTPException, OSError):
        print("Unable to connect to NTP! Using system time.")
        return timing.unix_time()

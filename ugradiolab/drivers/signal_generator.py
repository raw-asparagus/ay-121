"""Agilent/Keysight N9310A signal generator control via USBTMC (SCPI)."""

import time

DEVICE = '/dev/usbtmc0'
WAIT   = 0.25  # seconds between writes


class SignalGenerator:
    """Direct USBTMC interface to an Agilent/Keysight N9310A signal generator.

    Raises
    ------
    OSError
        If the USBTMC device cannot be opened.
    AssertionError
        If the connected instrument does not identify as an N9310A.
    """

    def __init__(self):
        self._dev = open(DEVICE, 'rb+')
        self._validate()

    def _validate(self):
        """Verify that the connected instrument is an N9310A.

        Raises
        ------
        AssertionError
            If the instrument identification string does not contain
            ``'N9310A'``.
        OSError
            If the underlying query fails.
        """
        resp = self._query('*IDN?')
        assert 'N9310A' in resp, f'Unexpected instrument: {resp}'

    def _write(self, cmd):
        """Send a SCPI command to the instrument.

        Raises
        ------
        OSError
            If the device write or flush fails.
        """
        cmd = cmd.encode()
        self._dev.write(cmd)
        self._dev.flush()
        time.sleep(WAIT)

    def _read(self):
        """Read a SCPI response from the instrument.

        Notes
        -----
        ``TimeoutError`` from the device read loop is handled internally and
        treated as the end-of-response marker.

        Raises
        ------
        OSError
            If the device read fails for reasons other than the terminating
            timeout.
        """
        chunks = []
        while True:
            try:
                chunks.append(self._dev.read(1))
            except TimeoutError:
                break
        return b''.join(chunks).decode().strip()

    def _query(self, cmd):
        """Send a SCPI query and return the response string.

        Raises
        ------
        OSError
            If the device write or read fails.
        """
        self._write(cmd)
        return self._read()

    # ---- Frequency --------------------------------------------------------

    def set_freq_mhz(self, freq_mhz):
        """Set the CW frequency in MHz.

        Parameters
        ----------
        freq_mhz : float
            Frequency in MHz.

        Returns
        -------
        None
            The command is sent to the instrument.

        Raises
        ------
        OSError
            If the SCPI write fails.
        """
        self._write(f'FREQ:CW {freq_mhz} MHz')

    def get_freq(self):
        """Return the current CW frequency.

        Returns
        -------
        freq_hz : float
            Frequency in Hz.

        Raises
        ------
        ValueError
            If the instrument response cannot be parsed as a frequency.
        OSError
            If the SCPI query fails.
        """
        resp  = self._query('FREQ:CW?')
        parts = resp.split()

        multiplier = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1.0}
        if len(parts) >= 2 and parts[1] in multiplier:
            return float(parts[0]) * multiplier[parts[1]]
        return float(parts[0])

    # ---- Amplitude --------------------------------------------------------

    def set_ampl_dbm(self, amp_dbm):
        """Set the CW amplitude in dBm.

        Parameters
        ----------
        amp_dbm : float
            Amplitude in dBm.

        Returns
        -------
        None
            The command is sent to the instrument.

        Raises
        ------
        OSError
            If the SCPI write fails.
        """
        self._write(f'AMPL:CW {amp_dbm} dBm')

    def get_ampl(self):
        """Return the current CW amplitude.

        Returns
        -------
        amp_dbm : float
            Amplitude in dBm.

        Raises
        ------
        ValueError
            If the instrument response cannot be parsed as a float.
        OSError
            If the SCPI query fails.
        """
        resp  = self._query('AMPL:CW?')
        parts = resp.split()
        return float(parts[0])

    # ---- RF output --------------------------------------------------------

    def rf_on(self):
        """Enable RF output.

        Returns
        -------
        None
            The command is sent to the instrument.

        Raises
        ------
        OSError
            If the SCPI write fails.
        """
        self._write('RFO:STAT ON')

    def rf_off(self):
        """Disable RF output.

        Returns
        -------
        None
            The command is sent to the instrument.

        Raises
        ------
        OSError
            If the SCPI write fails.
        """
        self._write('RFO:STAT OFF')

    def rf_state(self):
        """Return the RF output state.

        Returns
        -------
        rf_on : bool
            ``True`` if RF output is on.

        Raises
        ------
        ValueError
            If the instrument response cannot be parsed as ``0`` or ``1``.
        OSError
            If the SCPI query fails.
        """
        resp = self._query('RFO:STAT?')
        return bool(int(resp.strip()[0]))

    # ---- Lifecycle --------------------------------------------------------

    def close(self):
        """Perform a best-effort shutdown of the USBTMC session.

        Returns
        -------
        None
            RF output is disabled if possible and the device handle is closed.

        Notes
        -----
        Exceptions raised while turning RF off or closing the handle are caught
        and suppressed so that shutdown remains idempotent and best effort.
        """
        dev = self._dev
        if dev is None:
            return
        try:
            self.rf_off()
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass
        self._dev = None

"""Stateful queue runner for experiment execution."""

import time

from .experiment import CalExperiment, ObsExperiment


class QueueRunner:
    """Manage and execute an ordered sequence of experiments."""

    def __init__(
        self,
        experiments,
        sdr,
        synth=None,
        confirm=True,
        cadence_sec=None,
        input_fn=input,
        output_fn=print,
    ):
        self.experiments = list(experiments)
        self.sdr = sdr
        self.synth = synth
        self.confirm = confirm
        self.cadence_sec = cadence_sec
        self.input_fn = input_fn
        self.output_fn = output_fn

    @staticmethod
    def _format_experiment(exp, index, total):
        """Format experiment details for display."""
        tag = type(exp).__name__
        lines = [f'[{index}/{total}] {exp.prefix} ({tag})']
        lines.append(f'  alt={exp.alt_deg}  az={exp.az_deg}')
        lines.append(
            f'  nsamples={exp.nsamples}  nblocks={exp.nblocks}  '
            f'sample_rate={exp.sample_rate / 1e6:.2f} MHz'
        )
        if isinstance(exp, CalExperiment):
            lines.append(f'  siggen: {exp.siggen_freq_mhz} MHz, {exp.siggen_amp_dbm} dBm')
        else:
            lines.append('  siggen: OFF')
        return '\n'.join(lines)

    def run(self):
        """Execute the queue and return output file paths."""
        n = len(self.experiments)
        paths = []
        obs_start = None
        for i, exp in enumerate(self.experiments):
            if isinstance(exp, CalExperiment) and self.synth is None:
                raise ValueError(
                    f'Experiment {i + 1}/{n} ({exp.prefix}) is CalExperiment, '
                    'but synth=None. Provide a connected signal generator.'
                )

            # Sleep until next cadence boundary (before starting an ObsExperiment)
            if self.cadence_sec and isinstance(exp, ObsExperiment) and obs_start is not None:
                elapsed = time.time() - obs_start
                wait = self.cadence_sec - elapsed
                if wait > 0:
                    self.output_fn(f'  sleeping {wait:.1f}s until next cadence...')
                    time.sleep(wait)

            self.output_fn(self._format_experiment(exp, i + 1, n))
            if self.confirm:
                resp = self.input_fn('  [Enter]=run  s=skip  q=quit: ').strip().lower()
                if resp == 'q':
                    self.output_fn('Queue aborted.')
                    break
                if resp == 's':
                    self.output_fn('  skipped.')
                    continue

            if isinstance(exp, ObsExperiment):
                obs_start = time.time()

            path = exp.run(self.sdr, synth=self.synth)
            paths.append(path)
            self.output_fn(f'  -> {path}')
        return paths

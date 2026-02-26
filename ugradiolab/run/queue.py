"""Stateful queue runner for experiment execution."""

import time


def _format_experiment(exp, index, total):
    """Formats experiment details for display."""
    tag   = type(exp).__name__
    lines = [
        f'[{index}/{total}] {exp.prefix} ({tag})',
        f'  alt={exp.alt_deg:.2f}  az={exp.az_deg:.2f}',
        f'  nsamples={exp.nsamples}  nblocks={exp.nblocks}'
        f'  sample_rate={exp.sample_rate / 1e6:.2f} MHz',
        f'  siggen: {exp.siggen_summary()}'
    ]
    return '\n'.join(lines)


class QueueRunner:
    """Manages and executes an ordered sequence of experiments."""

    def __init__(
        self,
        experiments,
        sdr,
        synth       = None,
        confirm     = True,
        cadence_sec = None,
    ):
        self.experiments = list(experiments)
        self.sdr         = sdr
        self.synth       = synth
        self.confirm     = confirm
        self.cadence_sec = cadence_sec

    def run(self):
        """Executes the queue and return output file paths."""
        n = len(self.experiments)

        paths     = []
        obs_start = None
        for i, exp in enumerate(self.experiments):
            # Sleep until next cadence boundary
            # (before starting an ObsExperiment)
            if (
                    self.cadence_sec
                    and exp.counts_for_cadence
                    and obs_start is not None
            ):
                elapsed = time.time() - obs_start
                wait    = self.cadence_sec - elapsed
                if wait > 0:
                    print(f'  sleeping {wait:.1f}s until next cadence...')
                    time.sleep(wait)

            print(_format_experiment(exp, i + 1, n))
            if self.confirm:
                resp = (
                    input('  [Enter]=run  s=skip  q=quit: ')
                    .strip()
                    .lower()
                )
                if resp == 'q':
                    print('Queue aborted.')
                    break
                if resp == 's':
                    print('  skipped.')
                    continue

            if exp.counts_for_cadence:
                obs_start = time.time()

            path = exp.run(self.sdr, synth=self.synth)
            paths.append(path)
            print(f'  -> {path}')

        return paths

"""Stateful queue runner for experiment execution."""


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
        synth   = None,
        confirm = True,
    ):
        self.experiments = list(experiments)
        self.sdr         = sdr
        self.synth       = synth
        self.confirm     = confirm

    def run(self):
        """Executes the queue and return output file paths."""
        n = len(self.experiments)

        paths = []
        for i, exp in enumerate(self.experiments):
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

            path = exp.run(self.sdr, synth=self.synth)
            paths.append(path)
            print(f'  -> {path}')

        return paths

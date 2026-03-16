"""Stateful queue runner for experiment execution."""


def _format_experiment(exp, index, total):
    """Formats experiment details for display."""
    lines = [
        f'[{index}/{total}] {exp.prefix} ({type(exp).__name__})',
        f'  alt={exp.alt_deg:.2f}  az={exp.az_deg:.2f}',
        *exp._run_summary(),
    ]
    return '\n'.join(lines)


class QueueRunner:
    """Manages and executes an ordered sequence of experiments."""

    def __init__(
        self,
        experiments,
        confirm = True,
    ):
        self.experiments = list(experiments)
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

            path = exp.run()
            paths.append(path)
            print(f'  -> {path}')

        return paths

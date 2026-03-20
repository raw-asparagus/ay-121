"""Stateful sequential runner for experiment execution."""


def _format_experiment(exp, index, total):
    """Format one experiment summary block for terminal display.

    Notes
    -----
    This helper does not perform validation. Attribute errors from malformed
    experiment objects propagate to the caller unchanged.
    """
    lines = [
        f'[{index}/{total}] {exp.prefix} ({type(exp).__name__})',
        f'  alt={exp.alt_deg:.2f}  az={exp.az_deg:.2f}',
        *exp._run_summary(),
    ]
    return '\n'.join(lines)


class SequentialRunner:
    """Execute a finite ordered sequence of experiments.

    Parameters
    ----------
    experiments : iterable
        Experiment objects exposing ``prefix``, ``alt_deg``, ``az_deg``,
        ``_run_summary()``, and ``run()``.
    confirm : bool, optional
        If ``True``, prompt before each experiment is executed.

    Attributes
    ----------
    experiments : list
        Ordered experiment list to execute.
    confirm : bool
        Whether interactive confirmation is enabled.
    """

    def __init__(
        self,
        experiments,
        confirm = True,
    ):
        self.experiments = list(experiments)
        self.confirm     = confirm

    def run(self):
        """Execute the queued experiments in order.

        Returns
        -------
        paths : list[str]
            Saved output paths for experiments that were executed and not
            skipped.

        Raises
        ------
        Exception
            Propagates any exception raised by an individual experiment's
            ``run()`` method.
        """
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

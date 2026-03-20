import os
import time


def make_path(outdir: str, prefix: str, tag: str) -> str:
    """Return a timestamped output filepath and create the destination directory.

    Parameters
    ----------
    outdir : str
        Output directory to create if needed.
    prefix : str
        Filename prefix.
    tag : str
        Filename tag inserted before the timestamp.

    Returns
    -------
    path : str
        Timestamped ``.npz`` output path under ``outdir``.

    Raises
    ------
    OSError
        If the directory cannot be created.
    """
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(outdir, f"{prefix}_{tag}_{ts}.npz")

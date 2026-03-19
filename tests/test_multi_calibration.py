import importlib.util
import sys
from pathlib import Path


def _load_multi_calibration_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / 'labs/03/scripts'
    module_path = script_dir / 'multi_calibration.py'
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            'lab3_multi_calibration',
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_select_target_falls_back_to_moon(monkeypatch):
    mod = _load_multi_calibration_module()

    monkeypatch.setattr(
        mod,
        'compute_sun_pointing',
        lambda: (0.0, 10.0, 20.0, 30.0, 40.0),
    )
    monkeypatch.setattr(
        mod,
        'compute_moon_pointing',
        lambda: (12.0, 34.0, 0.0, 0.0, 0.0),
    )

    assert mod.select_target() == ('moon', 12.0, 34.0, 10.0)


def test_select_target_falls_back_to_m17(monkeypatch):
    mod = _load_multi_calibration_module()

    monkeypatch.setattr(
        mod,
        'compute_sun_pointing',
        lambda: (0.0, 10.0, 20.0, 30.0, 40.0),
    )
    monkeypatch.setattr(
        mod,
        'compute_moon_pointing',
        lambda: (0.0, 34.0, 0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        mod,
        'compute_radec_pointing',
        lambda ra, dec: (15.0, 25.0, 50.0)
        if ra == mod.M17_RA_DEG else (0.0, 0.0, 60.0),
    )
    monkeypatch.setattr(mod, 'lst_deg', lambda jd: 100.0)
    monkeypatch.setattr(
        mod,
        'optimal_duration',
        lambda ha, dec, baseline, phase: 7.5,
    )

    assert mod.select_target() == ('m17', 15.0, 25.0, 7.5)

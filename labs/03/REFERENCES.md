# Lab 03 — References

A curated bibliography for every magic number, physical claim, and analysis
technique used in this lab. Citations in the notebooks and source code use
the short keys in the left column.

Where multiple references exist for the same fact, the first entry is the
one I would point a reader at first; the others are listed for completeness.
Numerical values quoted in the lab text are the *typical* values from these
references, not always reproduced to the original paper's precision —
the lab's own error budget (notebook 05 §8.1) is broader than the dispersion
between published values.

---

## Textbooks (the workhorses)

| Key | Reference |
|---|---|
| **TMS3** | Thompson, A. R., Moran, J. M., & Swenson, G. W. Jr. 2017, *Interferometry and Synthesis in Radio Astronomy*, 3rd ed., Springer (open access). The standard reference for everything interferometric: van Cittert–Zernike (Ch. 14), $(u,v,w)$ coordinates and geometric delay (Ch. 4), bandwidth and time-average decorrelation (§6.3), the radiometer equation (§6.2), and the suite of baseline-determination techniques used in nb 04 (Ch. 12). |
| **R&W** | Rohlfs, K., & Wilson, T. L. 2004, *Tools of Radio Astronomy*, 4th ed., Springer. Radiometer equation, $T_{\rm sys}$, antenna temperature conversion. |
| **R&L** | Rybicki, G. B., & Lightman, A. P. 1979, *Radiative Processes in Astrophysics*, Wiley. Free–free emission §5; cosmic radiative transfer; brightness temperature definition. |
| **B&W** | Born, M., & Wolf, E. 1999, *Principles of Optics*, 7th ed., Cambridge University Press. Airy / jinc derivation §8.5; Fourier-optics treatment of the visibility. |
| **Stix** | Stix, M. 2002, *The Sun: An Introduction*, 2nd ed., Springer. Photospheric, chromospheric, and coronal structure; the standard textbook value of the optical solar radius and the radio "extension" of the Sun. |
| **A&S** | Abramowitz, M., & Stegun, I. A. 1965, *Handbook of Mathematical Functions*, National Bureau of Standards. Bessel functions §9; the table of zeros $j_{1,k}$ used in the diameter measurement. |
| **DLMF** | NIST Digital Library of Mathematical Functions, https://dlmf.nist.gov, Release 1.2 (2024). Modern, online, machine-checked replacement for A&S; Bessel-function chapter §10. |

## Foundational interferometry papers

| Key | Reference |
|---|---|
| **vC34** | van Cittert, P. H. 1934, *Physica*, **1**, 201. The original derivation of the spatial coherence function. |
| **Z38** | Zernike, F. 1938, *Physica*, **5**, 785. The radio-relevant statement of the same theorem. |
| **Ryle52** | Ryle, M. 1952, *Proc. R. Soc. Lond. A*, **211**, 351. The adding interferometer (relevant for the §1.2 derivation of the NCH instrument response). |
| **Golub73** | Golub, G. H., & Pereyra, V. 1973, *SIAM Journal on Numerical Analysis*, **10**, 413, "The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate". The variable-projection (separable NLS) algorithm used in Methods 4 and 4a: the nonlinear parameters $(b_{\rm ew}, b_{\rm ns})$ are iterated while the linear parameters $(A, B, C, D)$ are solved analytically at each step, reducing a 6-parameter NLS to a 2-parameter one. |
| **Harris78** | Harris, F. J. 1978, *Proceedings of the IEEE*, **66**, 51, "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform". The canonical reference for windowing in the DFT/FFT; the Hann window used in the STFT fringe-frequency method (Method 1a) and in the FFT x-space method (Method 1) is discussed in §III-B. |
| **Oppenheim** | Oppenheim, A. V., & Schafer, R. W. 1989, *Discrete-Time Signal Processing*, Prentice Hall. §8.6 for the Short-Time Fourier Transform (STFT) framework used in Method 1a; §7 for the DFT. |
| **Virtanen20** | Virtanen, P. et al. (SciPy 1.0 Contributors). 2020, *Nature Methods*, **17**, 261, "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python". The `scipy.optimize.least_squares` Levenberg-Marquardt back-end used in Methods 4 and 4a. |

## Solar physics — quiet Sun, chromosphere, free–free

| Key | Reference |
|---|---|
| **Dulk85** | Dulk, G. A. 1985, *Annual Review of Astronomy & Astrophysics*, **23**, 169, "Radio Emission from the Sun and Stars". The textbook review of cm-wavelength solar emission, free–free vs gyroresonance, brightness temperatures of the quiet Sun. |
| **VAL81** | Vernazza, J. E., Avrett, E. H., & Loeser, R. 1981, *ApJS*, **45**, 635, "Structure of the Solar Chromosphere III". The semi-empirical chromospheric model from which the $T_e(h)$ profile cited in §1.5 derives; subsequent updates are in **FAL93** (Fontenla, Avrett, Loeser 1993, *ApJ*, 406, 319) and **FAL2009** (Fontenla et al. 2009, *ApJ*, 707, 482). |
| **Zirin91** | Zirin, H., Baumert, B. M., & Hurford, G. J. 1991, *ApJ*, **370**, 779, "The microwave brightness temperature spectrum of the quiet sun". The canonical millimetre/centimetre $T_b$ measurements; quotes ~10 000 K at 10 GHz. |
| **BBG98** | Bastian, T. S., Benz, A. O., & Gary, D. E. 1998, *Annual Review of Astronomy & Astrophysics*, **36**, 131, "Radio Emission from Solar Flares". Background on active-region radio emission, gyroresonance vs thermal free–free, brightness-temperature ranges over sunspots. |

## Solar diameter — optical and radio

> **Provenance of the numerical comparison values.** The optical
> `SOLAR_DIAMETER_ARCMIN_NOMINAL = 31.6'` predates this analysis (most
> likely from the AY 121 lab manual itself; appropriate for an observation
> near aphelion). The *physical* mean optical diameter is $\sim 31.99'$
> from `[BCD98]`. The radio comparison value used in nb 05 is now anchored
> on `[Selhorst04]`'s `[NoRH]` 17 GHz measurement (the closest published
> *measured* value to our 10 GHz observation), and the K-band measurements
> of `[Marongiu24]` are used as a sanity-check on the wavelength
> dependence. **There is no recent definitive published radius at 10 GHz
> in the literature visible to me from the present search**; the
> Toyokawa-polarimeter / RATAN-600 historical data exist but I have not
> located a clean recent number. The student should treat the 17 GHz value
> as the closest reference and note that the 10 GHz radius is expected to
> be *slightly larger* (longer wavelength → higher in the chromosphere).

| Key | Reference |
|---|---|
| **AY121-Lab3** | (See "Lab manual" section.) The operational source for `SOLAR_DIAMETER_ARCMIN_NOMINAL = 31.6'` in `constants.py`. The student should verify this against the *current* lab manual; the value is appropriate for an observation near solar aphelion (July). |
| **BCD98** | Brown, T. M., & Christensen-Dalsgaard, J. 1998, *ApJ Letters*, **500**, L195, "Accurate determination of the solar photospheric radius". Helioseismic determination of $R_\odot$ used as the underlying physical optical reference. Mean optical $R_\odot \approx 959''$ at 1 AU, i.e. mean diameter $\sim 31.97'$; orbital eccentricity modulates this by $\pm 1.7\,\%$ across the year. |
| **Meftah18** | Meftah, M., Hauchecorne, A., Urbain, M., et al. 2018, *A&A*, **616**, A64. PICARD-mission solar-radius determination from SODISM, $R_\odot = 959.78 \pm 0.19''$ in the visible; cited as a direct-imaging cross-check on **BCD98**. |
| **Stix-radio-rule** | The qualitative claim "the radio Sun at centimetre wavelengths is a few percent larger than the optical Sun due to chromospheric/low-coronal $\tau=1$ being a few thousand km above the photosphere" is textbook-level — see [Stix] §10.3 and [Dulk85] §III. |
| **Selhorst04** | **Selhorst, C. L., Silva, A. V. R., & Costa, J. E. R. 2004, *Astronomy & Astrophysics*, **420**, 1117**, "Solar atmospheric model over a highly polarized 17 GHz active region" / "Radius variations over a solar cycle". Reports the mean 17 GHz solar radius from > 3800 NoRH (Nobeyama Radioheliograph) maps spanning the 1992–2003 solar cycle: $$R_\odot(17\,\mathrm{GHz}) = 976.6 \pm 1.5'' \;\;\;\Longleftrightarrow\;\;\; \theta_\odot(17\,\mathrm{GHz}) = 32.55 \pm 0.05'.$$ This is the **anchor reference** used as the radio comparison value in this lab. The Selhorst+04 value is from the closest frequency to our 10 GHz observation that has a definitive published radius; lower frequencies (i.e. 10 GHz) probe slightly higher in the chromosphere and are expected to give a *slightly larger* radius. arXiv: astro-ph/0312427. |
| **Selhorst-modelling** | Generic pointer to the C. L. Selhorst, J. E. R. Costa, A. V. R. Silva and collaborators *modelling* series of papers (multiple, ~2003–present) on the cm-wavelength solar atmosphere — limb brightening, the radius vs frequency relation, the role of magnetic structures and active regions in the brightness profile. Used wherever the lab text needs a "the modelling literature shows…" citation without committing to a specific volume/page that I have not verified. The student should look up the most recent paper in the series for a real lab writeup. |
| **Marongiu24** | **Marongiu, M., Pellizzoni, A., Mulas, S., et al. 2024, *Astronomy & Astrophysics*, **684**, A122**, "Study of solar brightness profiles in the 18–26 GHz frequency range with INAF radio telescopes — I. Solar radius". K-band measurements with the Grueff and Sardinia Radio Telescopes: $R_\odot(18.3\,\mathrm{GHz}) = 982.0 \pm 2.5''$, $R_\odot(25.8\,\mathrm{GHz}) = 978.4 \pm 2.2''$, etc. (HP-method values; the IP method gives $\sim 5''$ smaller). Used as a sanity-check on the cm-wavelength solar-radius compilation. The paper also gives the wavelength dependence of the radius across 18–26 GHz. |
| **Menezes-Valio18** | Menezes, F., & Valio, A. 2018 (Solar Phys., online preprint arXiv:1712.06771), "Solar Radius at Sub-Terahertz Frequencies and its Relation to Solar Activity". Subterahertz measurements: $R_\odot(212\,\mathrm{GHz}) = 966.5 \pm 2.8''$, $R_\odot(405\,\mathrm{GHz}) = 966.5 \pm 2.7''$. Useful as the high-frequency end of the radius-vs-frequency curve. |
| **Alissandrakis-ALMA** | C. E. Alissandrakis and collaborators have published a series of ALMA-era papers (from ~2017 onward) on the chromospheric/coronal solar radius and centre-to-limb brightness profiles at mm/submm wavelengths, generally consistent with the picture of mild limb brightening at cm wavelengths and marginal limb brightening / no limb darkening at mm wavelengths. Used as background for the limb-brightening discussion in nb 01 §1.5. |

## SDO / HMI / NOAA cross-check

| Key | Reference |
|---|---|
| **Pesnell12** | Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C. 2012, *Solar Physics*, **275**, 3, "The Solar Dynamics Observatory (SDO)". The mission paper; SDO provides full-disk continuum/magnetogram data via HMI for cross-checking spot positions. |
| **Scherrer12** | Scherrer, P. H. et al. 2012, *Solar Physics*, **275**, 207, "The Helioseismic and Magnetic Imager (HMI) Investigation for the SDO". The HMI instrument paper; data archived at JSOC, https://jsoc.stanford.edu. |
| **NOAA-SWPC** | NOAA Space Weather Prediction Center, *Solar Region Summary*, https://www.swpc.noaa.gov/products/solar-region-summary. Daily catalogue of NOAA-numbered active regions, used for cross-checking the EW offset of any sunspot detected at the Bessel nulls. |

## Geodetic / metrological constants

| Key | Reference |
|---|---|
| **BIPM-SI** | Bureau International des Poids et Mesures, *The International System of Units (SI)*, 9th edition, 2019. Defines $c = 299\,792\,458$ m s$^{-1}$ exactly. |
| **IERS2010** | IERS Conventions (2010), G. Petit and B. Luzum (eds.), IERS Technical Note 36. Defines the sidereal day $T_{\rm sid} = 86\,164.0905$ s and Earth's rotation rate $\omega_\oplus = 7.2921159 \times 10^{-5}$ rad s$^{-1}$. |
| **AA** | *The Astronomical Almanac for the Year 2026*, USNO/HMNAO. Sun ephemeris (declination, RA, hour angle); the values used in the chip metadata `sun_dec_deg`, `sun_ra_deg` are derived from `astropy` calls that ultimately resolve to JPL DE430/DE440. |
| **NCH-coords** | Berkeley Astronomy Department, *AY 121 Lab Manual* (current version). Source for the New Campbell Hall observatory coordinates `NCH_LAT_DEG = 37.873199°`, `NCH_LON_DEG = -122.257063°` and the nominal antenna baseline orientation. |

## Lab manual and previous AY 121 lab notes

| Key | Reference |
|---|---|
| **AY121-Lab3** | Berkeley Astronomy Department, *AY 121 Lab 3 Manual: Interferometric Observations of the Sun* (current version, 2025/26). The procedural source for the observing strategy, the IF chain configuration (`F_S_HZ`, `F_RF0_HZ`, `PLOT_BAND_GHZ`), the bad-channel list (`BAD_CHANNELS = (0, 256, 512, 768)` — DC and FPGA harmonics), and the baseline-fitting techniques formalised in nb 04. |
| **AY121-FitNotes** | Berkeley Astronomy Department, *AY 121 Fitting Notes / Bevington-style $\chi^2$ procedure*. Source of the curvature-matrix $[\alpha]$ / covariance $[\alpha]^{-1}$ procedure used in the brute-force grid-search method (nb 04 §4.6). The textbook reference is Bevington & Robinson 2003, *Data Reduction and Error Analysis for the Physical Sciences*, 3rd ed., McGraw-Hill. |

## Lidar prior

The lidar baseline measurement was made on the NCH roof with an
**iPhone 13 Pro** using the built-in time-of-flight ARKit depth API. Apple
does not publish a metric 1 σ accuracy for the iPhone lidar at multi-metre
ranges; published independent evaluations of the iPhone 12/13 Pro lidar at
3–10 m range typically report few-cm random error growing with distance,
plus systematic offsets at the dm level on poorly-textured surfaces (see
e.g. Spreafico et al. 2021, *Geosciences*, **11**, 478, for an iPhone 12 Pro
benchmark). The 30 cm 1 σ adopted in `constants.py::NOMINAL_B_*_ERR_M` is a
deliberately conservative floor combining single-shot ranging accuracy at
~15 m and the unmodelled contribution from identifying the antenna phase
centre by eye on a real dish; **it is not a manufacturer specification**,
and the fringe data themselves (notebook 04 summary table) are internally
consistent at the sub-cm level, so the 30 cm should be read as "the
lidar prior is the bottleneck, not the radio data."

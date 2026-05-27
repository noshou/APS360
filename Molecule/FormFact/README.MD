Form Factor Library 
A Fortran library for X-ray scattering calculations providing atomic form factors (f₀) and anomalous 
scattering corrections (f₁, f₂) for crystallographic structure factor computations.

Overview
This library provides:

Atomic form factors (f₀): Scattering amplitude as a function of Q = (sin θ)/λ
Anomalous scattering factors (f₁, f₂): Energy-dependent corrections for accurate X-ray diffraction calculations
Fast lookup tables: Pre-computed values for rapid calculations

Physical Background
In X-ray crystallography, the structure factor calculation requires:

f₀(Q): Decreases with scattering angle due to destructive interference from electron cloud
f₁: Real part of anomalous scattering (usually small and negative)
f₂: Imaginary part of anomalous scattering (usually small and positive)

Energy: Anomalous scattering factors provided at 12.4128 keV (1.0 Å wavelength)
Data Source: International Tables for Crystallography Vol. C (DOI: 10.1107/97809553602060000600)

Dependencies
Required

gfortran (or compatible Fortran compiler)
GNU Make

For Regenerating Data (Optional)
Only needed if modifying source data:

OCaml (≥ 4.08)
opam (OCaml package manager)
OCaml packages: str, csv, yojson
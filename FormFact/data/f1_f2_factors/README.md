source for data:

https://henke.lbl.gov/optical_constants/asf.html

		Low-Energy X-ray Interaction Coefficients:
		Photoabsorption, Scattering, and Reflection
			E = 30-30,000 eV, Z = 1-92

	        B. L. Henke, E. M. Gullikson, and J. C. Davis
			Center for X-Ray Optics, 2-400
			Lawrence Berkeley Laboratory
			Berkeley, California 94720

These files were used to generate the tables published in reference [1].
The files contain three columns of data: Energy(eV), f_1, f_2,
where f_1 and f_2 are the atomic (forward) scattering factors.
There are 500+ points on a uniform logarithmic mesh with points
added 0.1 eV above and below "sharp" absorption edges.
(Note: below 29 eV f_1 is set equal to -9999.)
The tabulated values of f_1 contain a relativistic, energy independent,
correction given by, Z* = Z - (Z/82.5)^(2.37).

The atomic photoabsorption cross section, mu_a, may be readily obtained
from the values of f_2 using the relation,

			mu_a = 2*r_0*lambda*f_2

where r_0 is the classical electron radius, and lambda is the wavelength.
The index of refraction for a material with N atoms per unit volume
is calculated by,

		n = 1 - N*r_0*(lambda)^2*(f_1+if_2)/(2*pi).

These (semi-empirical) atomic scattering factors are based upon
photoabsorption measurements of elements in their elemental state.
The basic assumption is that condensed matter may be modeled as a
collection of non-interacting atoms.  This assumption is in general
a good one for energies sufficiently far from absorption thresholds.
In the threshold regions, the specific chemical state is important
and direct experimental measurements must be made.

These tables are based on a compilation of the available experimental
measurements and theoretical calculations.  For many elements there is
little or no published data and in such cases it was necessary to
rely on theoretical calculations and interpolations across Z.
In order to improve the accuracy in the future considerably more
experimental measurements are needed.

Please send any comments to EMGullikson@lbl.gov.

Note that the following elements have been updated since the publication
of Ref. [1] in July 1993.

Element		Updated		Energy Range
Mg		1/15/94		30-50 eV
Al		1/15/94		30-73 eV
Si		1/15/94		30-100 eV
Au		11/7/94		2000-6500 eV
Li		11/15/94	2000-30000 eV

[1]  B. L. Henke, E. M. Gullikson, and J. C. Davis,
Atomic Data and Nuclear Data Tables Vol. 54 No. 2 (July 1993).


HOW TO USE FACTOR_LOOKUP

It is a nested dictionary with the following structure:
{
	Energy : float (electron volt) : {
		element_acronym : string : [f1 : float, f2 : float ]
	}
}

---

## Files in this directory

- `<element>.tsv` — Raw Henke tables, one per element. Columns: `E_ev`, `f1_<elm>`, `f2_<elm>`.
- `(12412.8)eV_f1_f2.json` — f1/f2 values at 12412.8 eV (1.0 Å) for all elements.
  This is the file consumed by `F1F2Parser.ml` to generate `F1F2Factors.f90`.
- `lookup_gen.R` — R script that joins all TSV files into a full energy-indexed JSON.

---

## Using lookup_gen.R

Requires R with `tidyverse`, `stringr`, and `jsonlite`. Run from this directory:

```bash
Rscript lookup_gen.R
```

Reads every `.tsv` in the current directory, joins on `E_ev`, and writes `factor_lookup.json`.
To update `(12412.8)eV_f1_f2.json` after adding a new element, extract the 12412.8 eV entry
from the generated `factor_lookup.json`.

---

## Adding a New Element

1. Add `<element>.tsv` with columns `E_ev`, `f1_<elm>`, `f2_<elm>`
2. Run `Rscript lookup_gen.R` to regenerate `factor_lookup.json`
3. Add the element's entry at 12412.8 eV to `(12412.8)eV_f1_f2.json`
4. Rebuild: `cd build && make`

---

## Energy Normalization Note

The Henke TSV tables use non-uniform, element-specific energy grids — not every element has
a tabulated point at exactly 12412.8 eV. Where no exact match exists, the nearest available
energy was used. The deviation is small (typically < 100 eV) and does not meaningfully affect
the anomalous scattering corrections at this energy.

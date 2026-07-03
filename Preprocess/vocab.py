import xraydb
from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class _IonVocab:

    """Static xraydb ion vocabulary, shared across all modules via the VOCAB singleton.

    Private; import VOCAB, not this class.

    Attributes
    ----------
    ions : tuple of str
        All ion symbols known to xraydb, in xraydb's original order.
    index : dict of str to int
        Mapping from lowercase ion symbol to its 1-based VOCAB index.
    """

    # AS OF 2026: 98 neutral elements + 111 ionic forms + 2 special cases (sival, cval)
    SPECIAL_CASES: ClassVar[dict[str, str]] = {"siva": "si", "cval": "c"}
    
    # transuranics have f0 but no f1/f2
    TRANSURANICS:  ClassVar[frozenset[str]] = frozenset({
        "np","np3+","np4+","np6+","pu","pu3+","pu4+","pu6+","am","cm","bk","cf"
    })

    ions:   tuple[str, ...]
    index:  dict[str, int] 
	
    @classmethod
    def load(cls) -> "_IonVocab":
        """Build the vocabulary from the xraydb f0 ion list.

        Indices are 1-based; 0 is reserved as the padding sentinel.

        Returns
        -------
        _IonVocab
            A new instance populated with all f0 ion symbols and their
            1-based index mapping.
        """
        ions = tuple(xraydb.get_xraydb().f0_ions())
        return cls(ions=ions, index={ion.lower(): idx+1 for idx, ion in enumerate(ions)})

    def __len__(self) -> int:
        """Return the number of ions in the vocabulary.

        Returns
        -------
        int
            Number of ions stored in `ions`.
        """
        return len(self.ions)

    def __contains__(self, ion: str) -> bool:
        """Check whether an ion string is present in the vocabulary.

        Parameters
        ----------
        ion : str
            Ion symbol to look up (e.g. 'c', 'fe2+').

        Returns
        -------
        bool
            True if `ion` is a key in `index`, False otherwise.
        """
        return ion in self.index

    def __getitem__(self, ion: str | int) -> str | int:
        """Translate between ion symbol and 1-based VOCAB index.

        Parameters
        ----------
        ion : str or int
            If a string, the ion symbol to look up. If an int, the
            1-based VOCAB index to resolve back to an ion symbol.

        Returns
        -------
        str or int
            The 1-based index for a string input, or the ion symbol
            for an int input.
        """
        if   isinstance(ion, str):
            return self.index[ion]
        elif isinstance(ion, int):
            return self.ions[ion-1]

    @property
    def size(self) -> int:
        """Return the number of ions in the vocabulary.

        Returns
        -------
        int
            Number of ions stored in `ions`.
        """
        return len(self.ions)

    def transuranic_indices(self) -> frozenset[int]:
        """Return VOCAB indices of all transuranic ions present in the vocabulary.

        Transuranics have f0 scattering factors but no f1/f2, so they may need
        special handling in the form factor computation.

        Returns
        -------
        frozenset of int
            VOCAB indices of all transuranic ions found in `index`.
        """
        return frozenset(self.index[ion] for ion in self.TRANSURANICS if ion in self.index)

    def special_case_indices(self) -> dict[str, int]:
        """Return a mapping from special-case alias to its resolved VOCAB index.

        Special cases are non-standard ion strings ('sival', 'cval') that
        are remapped to a canonical element before lookup. The returned index is
        that of the resolved ion, not the alias itself.

        Returns
        -------
        dict of str to int
            Mapping such as ``{'sival': index_of_si, 'cval': index_of_c}``.
        """
        return {alias: self.index[resolved]
                for alias, resolved in self.SPECIAL_CASES.items()
                if resolved in self.index}

VOCAB: _IonVocab = _IonVocab.load()


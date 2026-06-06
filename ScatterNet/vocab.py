import xraydb
from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class _IonVocab:
    """Static xraydb ion vocabulary. Shared across all modules via the VOCAB singleton.

    Private; import VOCAB, not this class.
    """

    # AS OF 2026: 98 neutral elements + 111 ionic forms + 2 special cases (sival, cval)
    SPECIAL_CASES: ClassVar[dict[str, str]] = {"sival": "si", "cval": "c"}
    
    # transuranics have f0 but no f1/f2
    TRANSURANICS:  ClassVar[frozenset[str]] = frozenset({
        "np","np3+","np4+","np6+","pu","pu3+","pu4+","pu6+","am","cm","bk","cf"
    })

    ions:   tuple[str, ...]
    index:  dict[str, int] 
	
    @classmethod
    def load(cls) -> "_IonVocab":
        """Build vocabulary from xraydb f0 ion list."""
        ions = tuple(xraydb.get_xraydb().f0_ions())
        return cls(ions=ions, index={ion.lower(): idx for idx, ion in enumerate(ions)})

    def __len__(self) -> int:
        return len(self.ions)

    def __contains__(self, ion: str) -> bool:
        return ion in self.index

    def __getitem__(self, ion: str) -> int:
        return self.index[ion]

    @property
    def size(self) -> int:
        return len(self.ions)

VOCAB: _IonVocab = _IonVocab.load()


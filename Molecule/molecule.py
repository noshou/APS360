import numpy as np
from beartype import beartype
import numpy.typing as npt
from dataclasses import dataclass, field

@beartype
@dataclass(frozen=True, slots=True)
class Molecule:
    xCoords: npt.NDArray[np.float64]
    yCoords: npt.NDArray[np.float64]
    zCoords: npt.NDArray[np.float64]
    f:       npt.NDArray[np.complex128]  
    
    _r:     npt.NDArray[np.float64] = field(init=False)
    _theta: npt.NDArray[np.float64] = field(init=False)
    _phi:   npt.NDArray[np.float64] = field(init=False)
    
    def __post_init__(self):
        x, y, z = self.xCoords, self.yCoords, self.zCoords
        if not (x.shape == y.shape == z.shape == self.f.shape) or x.size == 0:
            raise ValueError("Molecule: arrays must share shape and be non-empty")
        
        r = np.sqrt(x**2 + y**2 + z**2)
        if np.any(r == 0):
            raise ValueError("Molecule: atom at origin (r = 0) not allowed")
        
        object.__setattr__(self, "_r",     r)
        object.__setattr__(self, "_theta", np.arctan2(y, x))
        object.__setattr__(self, "_phi",   np.arccos(z / r))
    
    @property
    def r(self):     return self._r
    
    @property
    def theta(self): return self._theta
    
    @property
    def phi(self):   return self._phi
import numpy as np
from numpy.typing import NDArray

class Frame:
    pts: int | None
    time: float | None

    def to_ndarray(self, format: str = "rgb24") -> NDArray[np.uint8]: ...

import gc
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional


class Memory_Manager:
    def __init__(self, base_dir: Optional[str] = None):
        self.temp_dir = (
            Path(base_dir) if base_dir
            else Path(tempfile.mkdtemp(prefix='entrap_'))
        )
        if base_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._files = []
        self._arrays = []

    def create(
        self,
        shape: tuple,
        dtype=np.float64,
        name: str = None
    ) -> np.memmap:
        fname = self.temp_dir / (
            f'{name}.dat' if name else f'memmap_{len(self._files)}.dat'
        )
        self._files.append(fname)
        mmap = np.memmap(str(fname), dtype=dtype, mode='w+', shape=shape)
        self._arrays.append(mmap)
        return mmap

    def cleanup(self):
        for arr in self._arrays:
            try:
                arr.flush()
                del arr
            except Exception:
                pass
        self._arrays.clear()
        gc.collect()

        for fname in self._files:
            try:
                if fname.exists():
                    fname.unlink()
            except Exception:
                pass
        self._files.clear()

        try:
            if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                self.temp_dir.rmdir()
        except Exception:
            pass

    def __del__(self):
        self.cleanup()

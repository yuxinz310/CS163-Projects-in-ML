import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Locate instrument calibration and stellar-spectrum data.
os.environ["STPSF_PATH"] = str(
    PROJECT_DIR / "calibration" / "stpsf-data"
)
os.environ["PYSYN_CDBS"] = str(
    PROJECT_DIR / "calibration" / "synphot-data"
)

import stpsf


# function for generate spectrum
def create_stellar_spectrum(spectral_type, catalog="phoenix"):
    """Return a model stellar spectrum."""
    return stpsf.specFromSpectralType(
        spectral_type,
        catalog=catalog,
    )
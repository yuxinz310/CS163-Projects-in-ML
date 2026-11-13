"""Download and verify pinned calibration inputs; no STPSF import required."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen
from miri_paths import opd_selection_path

PROJECT_DIR = Path(__file__).resolve().parent


def matches(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def fetch(url, destination, checksum, offline=False):
    if matches(destination, checksum):
        return
    if offline:
        raise FileNotFoundError(f"Offline: missing or mismatched calibration file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading:", destination.name, flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, output)
        if not matches(temporary, checksum):
            raise ValueError(f"Checksum mismatch for {url}; existing data was not replaced.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def unpack_archive(archive, destination, expected_files):
    """Only accept regular files/directories within the expected archive root."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as folder:
        scratch = Path(folder)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                path = PurePosixPath(member.name)
                if (path.is_absolute() or ".." in path.parts or not path.parts
                        or path.parts[0] != "stpsf-data"
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                target = scratch.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        extracted = scratch / "stpsf-data"
        for name, checksum in expected_files.items():
            if not matches(extracted / name, checksum):
                raise ValueError(f"Extracted calibration failed verification: {name}")
        shutil.copytree(extracted, destination / "stpsf-data", dirs_exist_ok=True)


def ensure_calibration(project_dir=PROJECT_DIR, offline=False):
    project_dir = Path(project_dir)
    sources = json.loads((project_dir / "calibration_sources.json").read_text())
    calibration = project_dir / "calibration"
    package = sources["stpsf"]
    if not all(matches(calibration / "stpsf-data" / name, checksum)
               for name, checksum in package["files"].items()):
        archive = calibration / "downloads" / package["archive"]
        fetch(package["url"], archive, package["sha256"], offline)
        print("Extracting verified STPSF calibration...", flush=True)
        unpack_archive(archive, calibration, package["files"])
    for record in sources["downloads"]:
        fetch(record["url"], calibration / record["path"], record["sha256"], offline)
    selections = sources.get("opd_selections")
    if selections is None:  # Original single-filter manifest.
        selections = {"F1140C": sources["opd_selection"]}
    for filter_name, record in selections.items():
        selection = opd_selection_path(filter_name, project_dir)
        if selection.exists():
            if json.loads(selection.read_text()) != record:
                raise ValueError("Existing OPD selection differs from calibration_sources.json. "
                                 "Reconcile the selection explicitly before running this demo.")
        else:
            selection.parent.mkdir(parents=True, exist_ok=True)
            selection.write_text(json.dumps(record, indent=2) + "\n")
    print("Calibration verified:", calibration, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use verified local files/cache only.")
    args = parser.parse_args()
    ensure_calibration(offline=args.offline)


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Union

DEFAULT_QR_SAVE_DIR = Path(__file__).parent / "qrcodes"
DEFAULT_QR_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_qr_image(image, filename: str, output_dir: Union[Path, str] = DEFAULT_QR_SAVE_DIR) -> Path:
    """Save a QR code image to disk and return the saved path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path / filename
    image.save(destination)
    return destination


def get_qr_save_dir() -> Path:
    """Return the default directory where QR codes are saved."""
    return DEFAULT_QR_SAVE_DIR

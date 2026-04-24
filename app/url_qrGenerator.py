from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path

from streamlit import config


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QR_PATH = ROOT_DIR / "output" / "streamlit_network_url_qr.png"


@dataclass(frozen=True)
class QRGenerationResult:
    url: str
    qr_path: Path


def _get_lan_ip() -> str:
    """Return the LAN IP address Streamlit's Network URL should use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidate = sock.getsockname()[0]
            if not ip_address(candidate).is_loopback:
                return candidate
    except OSError:
        pass

    try:
        output = subprocess.check_output(["ifconfig"], text=True, stderr=subprocess.DEVNULL)
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "inet":
                candidate = parts[1]
                address = ip_address(candidate)
                if not address.is_loopback and not address.is_link_local:
                    return candidate
    except Exception:
        pass

    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _get_streamlit_port() -> int:
    try:
        return int(config.get_option("server.port"))
    except Exception:
        return 8501


def _get_base_url_path() -> str:
    try:
        base_path = str(config.get_option("server.baseUrlPath") or "").strip("/")
    except Exception:
        base_path = ""
    return f"/{base_path}" if base_path else ""


def build_streamlit_network_url() -> str:
    protocol = "http"
    try:
        if config.get_option("server.sslCertFile"):
            protocol = "https"
    except Exception:
        pass

    return f"{protocol}://{_get_lan_ip()}:{_get_streamlit_port()}{_get_base_url_path()}"


def generate_streamlit_network_qr(output_path: str | Path = DEFAULT_QR_PATH) -> QRGenerationResult:
    """Generate a QR code PNG for the current Streamlit Network URL."""
    try:
        import qrcode
    except ImportError as exc:
        raise RuntimeError(
            "The qrcode package is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    url = build_streamlit_network_url()
    qr_path = Path(output_path)
    qr_path.parent.mkdir(parents=True, exist_ok=True)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(qr_path)

    return QRGenerationResult(url=url, qr_path=qr_path)

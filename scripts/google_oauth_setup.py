"""
Autoriza Gmail + Google Calendar (OAuth Desktop).
Uso (una sola vez):
  1. Creá un proyecto en Google Cloud Console
  2. Habilitá Gmail API y Google Calendar API
  3. OAuth consent screen (External / Testing) + usuario de prueba
  4. Credenciales → OAuth client ID → Desktop → descargar JSON
  5. Guardalo como credentials/credentials.json
  6. python scripts/google_oauth_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from integrations.google_workspace import get_credentials, oauth_configured
from config.settings import GOOGLE_OAUTH_CLIENT_SECRETS, GOOGLE_OAUTH_TOKEN


def main() -> int:
    print("=== Setup OAuth Google (Gmail + Calendar) ===")
    print(f"Credentials: {GOOGLE_OAUTH_CLIENT_SECRETS}")
    print(f"Token:       {GOOGLE_OAUTH_TOKEN}")
    if not oauth_configured():
        print(
            f"\nERROR: no existe {GOOGLE_OAUTH_CLIENT_SECRETS}\n"
            "Descargá el JSON de cliente OAuth (Desktop) desde Google Cloud Console."
        )
        return 1
    creds = get_credentials(interactive=True)
    print("\nOK: autorización guardada.")
    print(f"Token válido: {bool(creds and creds.valid)}")
    print("Ya podés usar tool_enviar_email y tool_crear_eventos_calendario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

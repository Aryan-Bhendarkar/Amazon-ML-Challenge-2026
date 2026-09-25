"""Shared helpers for hooks (stdlib only; must work with any Python 3.8+ on Windows/Linux)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# External lookup services (fair-play rule: using them = disqualification)
FORBIDDEN_HOSTS = [
    "maps.googleapis", "geocode", "geocoding", "nominatim", "openstreetmap.org", "overpass-api",
    "opencagedata", "api.mapbox", "positionstack", "geoapify", "locationiq", "here.com/geocod",
    "opencorporates", "mca.gov.in", "api.insee.fr", "entreprise.data.gouv", "recherche-entreprises",
    "data.gouv.fr", "geonames", "zippopotam", "postalpincode", "api.postalpincode", "smartystreets",
    "usps.com", "census.gov/geocoder",
]
FORBIDDEN_PKGS = ["geopy", "googlemaps", "geocoder", "opencage", "pgeocode", "uszipcode", "zipcodes",
                  "pyzipcode", "postal", "pypostal", "libpostal"]


def read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def deny(reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    sys.exit(0)


def ask(reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                             "permissionDecision": "ask",
                                             "permissionDecisionReason": reason}}))
    sys.exit(0)

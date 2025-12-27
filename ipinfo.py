#!/usr/bin/env python3
"""
GeoIP lookup tool using MaxMind GeoLite2 databases (ASN + City).

Outputs:
  - pretty (default): human friendly blocks
  - table: compact aligned columns
  - json: JSON array
  - jsonl: newline-delimited JSON
  - csv: CSV with fixed columns
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, TextIO

import geoip2.database
import geoip2.errors


DEFAULT_GEOIP_DIR = Path("/var/lib/GeoIP")
DEFAULT_ASN_DB = DEFAULT_GEOIP_DIR / "GeoLite2-ASN.mmdb"
DEFAULT_CITY_DB = DEFAULT_GEOIP_DIR / "GeoLite2-City.mmdb"


@dataclass
class GeoIPResult:
    ip: str

    status: str = "ok"
    error: Optional[str] = None

    asn_number: Optional[int] = None
    asn_org: Optional[str] = None
    asn_network: Optional[str] = None

    continent_code: Optional[str] = None
    continent_name: Optional[str] = None

    country_iso: Optional[str] = None
    country_name: Optional[str] = None

    region_iso: Optional[str] = None
    region_name: Optional[str] = None

    city_name: Optional[str] = None
    postal_code: Optional[str] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_radius_km: Optional[int] = None
    time_zone: Optional[str] = None

    traits_network: Optional[str] = None
    is_anycast: Optional[bool] = None

    subdivisions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asn"] = f"AS{self.asn_number}" if self.asn_number is not None else None
        d["subdivisions"] = list(self.subdivisions)
        return d


CSV_FIELDS: Sequence[str] = (
    "ip",
    "status",
    "error",
    "asn",
    "asn_number",
    "asn_org",
    "asn_network",
    "continent_code",
    "continent_name",
    "country_iso",
    "country_name",
    "region_iso",
    "region_name",
    "city_name",
    "postal_code",
    "latitude",
    "longitude",
    "accuracy_radius_km",
    "time_zone",
    "traits_network",
    "is_anycast",
    "subdivisions",
)


class GeoIPService:
    def __init__(self, asn_db: Path, city_db: Path) -> None:
        self.asn_db = asn_db
        self.city_db = city_db
        self._asn_reader: Optional[geoip2.database.Reader] = None
        self._city_reader: Optional[geoip2.database.Reader] = None

    def __enter__(self) -> "GeoIPService":
        self._asn_reader = geoip2.database.Reader(str(self.asn_db))
        self._city_reader = geoip2.database.Reader(str(self.city_db))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._asn_reader is not None:
            self._asn_reader.close()
        if self._city_reader is not None:
            self._city_reader.close()

    def lookup(self, ip: str, verbose: bool = False) -> GeoIPResult:
        result = GeoIPResult(ip=ip)

        try:
            ipaddress.ip_address(ip)
        except ValueError as e:
            result.status = "invalid_ip"
            result.error = str(e)
            return result

        if self._asn_reader is None or self._city_reader is None:
            result.status = "error"
            result.error = "GeoIPService not initialized (missing database readers)"
            return result

        try:
            asn_resp = self._asn_reader.asn(ip)
            result.asn_number = asn_resp.autonomous_system_number
            result.asn_org = asn_resp.autonomous_system_organization
            result.asn_network = str(asn_resp.network) if asn_resp.network else None
        except geoip2.errors.AddressNotFoundError:
            pass
        except ValueError as e:
            result.status = "error"
            result.error = f"ASN lookup failed: {e}"

        try:
            city_resp = self._city_reader.city(ip)

            result.continent_name = getattr(city_resp.continent, "name", None)
            result.continent_code = getattr(city_resp.continent, "code", None)

            result.country_name = getattr(city_resp.country, "name", None)
            result.country_iso = getattr(city_resp.country, "iso_code", None)

            subs = list(city_resp.subdivisions) if city_resp.subdivisions else []
            if subs:
                first = subs[0]
                result.region_name = getattr(first, "name", None)
                result.region_iso = getattr(first, "iso_code", None)
                result.subdivisions = [
                    f"{getattr(s, 'name', None) or ''} ({getattr(s, 'iso_code', None) or ''})".strip()
                    for s in subs
                    if getattr(s, "name", None) or getattr(s, "iso_code", None)
                ]

            result.city_name = getattr(city_resp.city, "name", None)
            result.postal_code = getattr(city_resp.postal, "code", None)

            loc = city_resp.location
            result.latitude = getattr(loc, "latitude", None)
            result.longitude = getattr(loc, "longitude", None)
            result.accuracy_radius_km = getattr(loc, "accuracy_radius", None)
            result.time_zone = getattr(loc, "time_zone", None)

            traits = city_resp.traits
            result.traits_network = str(getattr(traits, "network", None)) if getattr(traits, "network", None) else None
            result.is_anycast = getattr(traits, "is_anycast", None)

        except geoip2.errors.AddressNotFoundError:
            if result.asn_number is None:
                result.status = "not_found"
            else:
                result.status = "partial"
        except ValueError as e:
            result.status = "error"
            result.error = f"Geo lookup failed: {e}"

        if not verbose:
            pass

        return result


def iter_ips_from_sources(ips: Sequence[str], file_path: Optional[Path]) -> List[str]:
    out: List[str] = []

    for ip in ips:
        ip = ip.strip()
        if ip:
            out.append(ip)

    if file_path is None:
        return out

    if str(file_path) == "-":
        lines = sys.stdin.read().splitlines()
    else:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        token = raw.split()[0]
        if token and not token.startswith("#"):
            out.append(token)

    return out


def open_output(path: Optional[Path]) -> TextIO:
    if path is None or str(path) == "-":
        return sys.stdout
    return path.open("w", encoding="utf-8", newline="")


def s(v: Optional[object]) -> str:
    if v is None:
        return ""
    return str(v)


def render_pretty(results: Sequence[GeoIPResult], out: TextIO, verbose: bool) -> None:
    for idx, r in enumerate(results):
        if idx:
            out.write("\n")

        out.write(f"{r.ip}\n")
        if r.status != "ok" and r.status != "partial":
            out.write(f"  Status: {r.status}\n")
            if r.error:
                out.write(f"  Error:  {r.error}\n")
            continue

        if r.asn_number is not None:
            if r.asn_org:
                out.write(f"  ASN:    AS{r.asn_number} ({r.asn_org})\n")
            else:
                out.write(f"  ASN:    AS{r.asn_number}\n")
        else:
            out.write("  ASN:    (not found)\n")

        location_bits: List[str] = []
        if r.city_name:
            location_bits.append(r.city_name)
        if r.region_name:
            if r.region_iso:
                location_bits.append(f"{r.region_name} ({r.region_iso})")
            else:
                location_bits.append(r.region_name)
        if r.country_name:
            if r.country_iso:
                location_bits.append(f"{r.country_name} ({r.country_iso})")
            else:
                location_bits.append(r.country_name)

        if location_bits:
            out.write(f"  Place:  {', '.join(location_bits)}\n")
        elif r.country_iso or r.country_name:
            out.write(f"  Place:  {s(r.country_name)} ({s(r.country_iso)})\n")
        else:
            out.write("  Place:  (not found)\n")

        if r.postal_code:
            out.write(f"  Postal: {r.postal_code}\n")

        if r.latitude is not None and r.longitude is not None:
            coord = f"{r.latitude:.6f}, {r.longitude:.6f}"
            if r.accuracy_radius_km is not None:
                out.write(f"  Coord:  {coord} (accuracy ~{r.accuracy_radius_km} km)\n")
            else:
                out.write(f"  Coord:  {coord}\n")

        if r.time_zone:
            out.write(f"  TZ:     {r.time_zone}\n")

        if verbose:
            if r.asn_network:
                out.write(f"  ASNNet: {r.asn_network}\n")
            if r.traits_network:
                out.write(f"  Net:    {r.traits_network}\n")
            if r.is_anycast is not None:
                out.write(f"  Anycast:{' yes' if r.is_anycast else ' no'}\n")
            if r.continent_name or r.continent_code:
                out.write(f"  Cont:   {s(r.continent_name)} ({s(r.continent_code)})\n")
            if r.subdivisions:
                out.write(f"  Subs:   {', '.join(r.subdivisions)}\n")


def render_table(results: Sequence[GeoIPResult], out: TextIO, verbose: bool, header: bool = True) -> None:
    if verbose:
        headers = ["IP", "ASN", "Org", "Country", "Region", "City", "TZ", "Net", "Anycast"]
    else:
        headers = ["IP", "ASN", "Org", "Country", "Region", "City", "TZ"]

    rows: List[List[str]] = []
    for r in results:
        asn = f"AS{r.asn_number}" if r.asn_number is not None else ""
        org = r.asn_org or ""
        country = r.country_iso or r.country_name or ""
        region = r.region_iso or r.region_name or ""
        city = r.city_name or ""
        tz = r.time_zone or ""

        if verbose:
            net = r.traits_network or r.asn_network or ""
            anycast = ""
            if r.is_anycast is True:
                anycast = "yes"
            elif r.is_anycast is False:
                anycast = "no"
            rows.append([r.ip, asn, org, country, region, city, tz, net, anycast])
        else:
            rows.append([r.ip, asn, org, country, region, city, tz])

    def trunc(x: str, n: int) -> str:
        if len(x) <= n:
            return x
        return x[: n - 3] + "..."

    max_org = 38 if verbose else 42
    for row in rows:
        row[2] = trunc(row[2], max_org)
        if verbose:
            row[7] = trunc(row[7], 24)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(values[i].ljust(widths[i]) for i in range(len(values))).rstrip()

    if header:
        out.write(fmt_row(headers) + "\n")
        out.write(fmt_row(["-" * w for w in widths]) + "\n")

    for row in rows:
        out.write(fmt_row(row) + "\n")


def render_json(results: Sequence[GeoIPResult], out: TextIO, pretty: bool) -> None:
    payload = [r.to_dict() for r in results]
    if pretty:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")
    else:
        json.dump(payload, out, separators=(",", ":"), sort_keys=True)
        out.write("\n")


def render_jsonl(results: Sequence[GeoIPResult], out: TextIO) -> None:
    for r in results:
        out.write(json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def render_csv(results: Sequence[GeoIPResult], out: TextIO, header: bool = True) -> None:
    writer = csv.DictWriter(out, fieldnames=list(CSV_FIELDS))
    if header:
        writer.writeheader()

    for r in results:
        d = r.to_dict()
        row = {k: d.get(k) for k in CSV_FIELDS}
        row["subdivisions"] = ", ".join(r.subdivisions) if r.subdivisions else ""
        writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GeoIP lookup for IP addresses using MaxMind GeoLite2 ASN and City databases."
    )
    parser.add_argument("ips", nargs="*", help="IP address(es) to look up")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Read IPs from file (one per line). Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--asn-db",
        type=Path,
        default=DEFAULT_ASN_DB,
        help=f"Path to GeoLite2 ASN database (default: {DEFAULT_ASN_DB})",
    )
    parser.add_argument(
        "--city-db",
        type=Path,
        default=DEFAULT_CITY_DB,
        help=f"Path to GeoLite2 City database (default: {DEFAULT_CITY_DB})",
    )
    parser.add_argument(
        "--format",
        choices=("pretty", "table", "json", "jsonl", "csv"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write output to file (default: stdout). Use '-' for stdout.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include additional fields in pretty/table output",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print header row (csv/table)",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="For --format json, do not pretty-print",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any IP is invalid or a lookup errors out",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        ip_list = iter_ips_from_sources(args.ips, args.file)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not ip_list:
        print("error: no IP addresses provided (use positional args or -f FILE)", file=sys.stderr)
        return 2

    for db_path, label in ((args.asn_db, "ASN"), (args.city_db, "City")):
        if not db_path.exists():
            print(f"error: {label} database not found: {db_path}", file=sys.stderr)
            return 2

    results: List[GeoIPResult] = []
    strict_fail = False

    try:
        with GeoIPService(args.asn_db, args.city_db) as svc:
            for ip in ip_list:
                r = svc.lookup(ip, verbose=args.verbose)
                results.append(r)
                if args.strict and r.status not in ("ok", "partial", "not_found"):
                    strict_fail = True
                if args.strict and r.status == "error":
                    strict_fail = True
    except Exception as e:
        print(f"error: failed to open databases or perform lookups: {e}", file=sys.stderr)
        return 2

    with open_output(args.output) as out:
        header = not args.no_header

        if args.format == "pretty":
            render_pretty(results, out=out, verbose=args.verbose)
            out.write("\n")
        elif args.format == "table":
            render_table(results, out=out, verbose=args.verbose, header=header)
        elif args.format == "json":
            render_json(results, out=out, pretty=(not args.compact_json))
        elif args.format == "jsonl":
            render_jsonl(results, out=out)
        elif args.format == "csv":
            render_csv(results, out=out, header=header)

    if strict_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

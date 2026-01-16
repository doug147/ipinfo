#!/usr/bin/env python3
"""
GeoIP lookup tool using MaxMind GeoLite2 databases (ASN + City).

Outputs:
  - pretty (default): human friendly blocks
  - table: compact aligned columns
  - json: JSON array
  - jsonl: newline-delimited JSON
  - csv: CSV with fixed columns

Search mode:
  - Search for IP ranges matching ASN, country, region, city criteria

Sorting:
  - Lookup default: sort by IP
  - Search default: sort by network
  - Use --sort to choose fields, e.g. --sort asn,ip or --sort -asn,ip
  - Use --sort none (or input/preserve) to keep original order
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import sys
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

import geoip2.database
import geoip2.errors
import maxminddb


DEFAULT_GEOIP_DIR = Path("/usr/share/GeoIP")
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


@dataclass
class SearchResult:
    """Result from a search query - represents a network range."""
    network: str
    asn_number: Optional[int] = None
    asn_org: Optional[str] = None
    country_iso: Optional[str] = None
    country_name: Optional[str] = None
    region_iso: Optional[str] = None
    region_name: Optional[str] = None
    city_name: Optional[str] = None
    postal_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asn"] = f"AS{self.asn_number}" if self.asn_number is not None else None
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

SEARCH_CSV_FIELDS: Sequence[str] = (
    "network",
    "asn",
    "asn_number",
    "asn_org",
    "country_iso",
    "country_name",
    "region_iso",
    "region_name",
    "city_name",
    "postal_code",
    "latitude",
    "longitude",
)


def _ip_value(ip_str: str) -> Optional[Tuple[int, int]]:
    try:
        addr = ipaddress.ip_address(ip_str)
        return (addr.version, int(addr))
    except ValueError:
        return None


def _network_value(net_str: str) -> Optional[Tuple[int, int, int]]:
    try:
        net = ipaddress.ip_network(net_str, strict=False)
        return (net.version, int(net.network_address), int(net.prefixlen))
    except ValueError:
        return None


def _casefold(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    return v.casefold()


_LOOKUP_SORT_ALIASES: Dict[str, str] = {
    "asn": "asn_number",
    "asnnum": "asn_number",
    "org": "asn_org",
    "country": "country_iso",
    "cc": "country_iso",
    "region": "region_iso",
    "state": "region_iso",
    "city": "city_name",
    "postal": "postal_code",
    "zip": "postal_code",
    "tz": "time_zone",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "anycast": "is_anycast",
    "net": "traits_network",
    "network": "traits_network",
}

_LOOKUP_SORT_GETTERS: Dict[str, Callable[[GeoIPResult], Any]] = {
    "ip": lambda r: _ip_value(r.ip),
    "status": lambda r: _casefold(r.status),
    "error": lambda r: _casefold(r.error),
    "asn_number": lambda r: r.asn_number,
    "asn_org": lambda r: _casefold(r.asn_org),
    "asn_network": lambda r: _network_value(r.asn_network) if r.asn_network else None,
    "continent_code": lambda r: _casefold(r.continent_code),
    "continent_name": lambda r: _casefold(r.continent_name),
    "country_iso": lambda r: _casefold(r.country_iso),
    "country_name": lambda r: _casefold(r.country_name),
    "region_iso": lambda r: _casefold(r.region_iso),
    "region_name": lambda r: _casefold(r.region_name),
    "city_name": lambda r: _casefold(r.city_name),
    "postal_code": lambda r: _casefold(r.postal_code),
    "latitude": lambda r: r.latitude,
    "longitude": lambda r: r.longitude,
    "accuracy_radius_km": lambda r: r.accuracy_radius_km,
    "time_zone": lambda r: _casefold(r.time_zone),
    "traits_network": lambda r: _network_value(r.traits_network) if r.traits_network else None,
    "is_anycast": lambda r: r.is_anycast,
}

_SEARCH_SORT_ALIASES: Dict[str, str] = {
    "net": "network",
    "cidr": "network",
    "asn": "asn_number",
    "asnnum": "asn_number",
    "org": "asn_org",
    "country": "country_iso",
    "cc": "country_iso",
    "region": "region_iso",
    "state": "region_iso",
    "city": "city_name",
    "postal": "postal_code",
    "zip": "postal_code",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
}

_SEARCH_SORT_GETTERS: Dict[str, Callable[[SearchResult], Any]] = {
    "network": lambda r: _network_value(r.network),
    "asn_number": lambda r: r.asn_number,
    "asn_org": lambda r: _casefold(r.asn_org),
    "country_iso": lambda r: _casefold(r.country_iso),
    "country_name": lambda r: _casefold(r.country_name),
    "region_iso": lambda r: _casefold(r.region_iso),
    "region_name": lambda r: _casefold(r.region_name),
    "city_name": lambda r: _casefold(r.city_name),
    "postal_code": lambda r: _casefold(r.postal_code),
    "latitude": lambda r: r.latitude,
    "longitude": lambda r: r.longitude,
}


def _parse_sort_fields(
    sort_arg: Optional[str],
    default_sort: Optional[str],
    default_desc: bool,
    aliases: Dict[str, str],
    getters: Dict[str, Callable],
) -> List[Tuple[str, bool]]:
    raw = sort_arg if sort_arg is not None else default_sort
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    if raw.lower() in ("none", "input", "preserve"):
        return []

    specs: List[Tuple[str, bool]] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue

        desc = default_desc
        if token[0] in "+-":
            desc = token[0] == "-"
            token = token[1:].strip()
        if not token:
            continue

        field = aliases.get(token.lower(), token.lower())
        if field not in getters:
            allowed = ", ".join(sorted(getters.keys()))
            raise ValueError(f"Unknown sort field '{token}'. Allowed fields: {allowed}")

        specs.append((field, desc))

    return specs


def _ensure_tiebreak(specs: List[Tuple[str, bool]], field: str) -> List[Tuple[str, bool]]:
    if any(f == field for f, _ in specs):
        return specs
    return specs + [(field, False)]


def _cmp_values(a: Any, b: Any, desc: bool) -> int:
    if a is None and b is None:
        return 0
    if a is None:
        return 1
    if b is None:
        return -1

    if a < b:
        return 1 if desc else -1
    if a > b:
        return -1 if desc else 1
    return 0


def _make_comparator(specs: List[Tuple[str, bool]], getters: Dict[str, Callable]) -> Callable[[Any, Any], int]:
    def cmp(x: Any, y: Any) -> int:
        for field, desc in specs:
            vx = getters[field](x)
            vy = getters[field](y)
            c = _cmp_values(vx, vy, desc)
            if c != 0:
                return c
        return 0

    return cmp


def sort_geoip_results(results: Sequence[GeoIPResult], sort_arg: Optional[str], sort_desc: bool) -> List[GeoIPResult]:
    specs = _parse_sort_fields(
        sort_arg=sort_arg,
        default_sort="ip",
        default_desc=sort_desc,
        aliases=_LOOKUP_SORT_ALIASES,
        getters=_LOOKUP_SORT_GETTERS,
    )
    if not specs:
        return list(results)

    specs = _ensure_tiebreak(specs, "ip")
    cmp = _make_comparator(specs, _LOOKUP_SORT_GETTERS)
    return sorted(results, key=cmp_to_key(cmp))


def sort_search_results(results: Sequence[SearchResult], sort_arg: Optional[str], sort_desc: bool) -> List[SearchResult]:
    specs = _parse_sort_fields(
        sort_arg=sort_arg,
        default_sort="network",
        default_desc=sort_desc,
        aliases=_SEARCH_SORT_ALIASES,
        getters=_SEARCH_SORT_GETTERS,
    )
    if not specs:
        return list(results)

    specs = _ensure_tiebreak(specs, "network")
    cmp = _make_comparator(specs, _SEARCH_SORT_GETTERS)
    return sorted(results, key=cmp_to_key(cmp))


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


@dataclass
class SearchCriteria:
    """Criteria for searching IP ranges."""
    asn: Optional[int] = None
    asn_org_pattern: Optional[str] = None
    country_iso: Optional[str] = None
    region_iso: Optional[str] = None
    city_pattern: Optional[str] = None
    ipv4_only: bool = True
    ipv6_only: bool = False
    limit: Optional[int] = None

    def matches_asn(self, asn_num: Optional[int], asn_org: Optional[str]) -> bool:
        if self.asn is not None and asn_num != self.asn:
            return False
        if self.asn_org_pattern is not None:
            if asn_org is None:
                return False
            if self.asn_org_pattern.lower() not in asn_org.lower():
                return False
        return True

    def matches_geo(
        self,
        country_iso: Optional[str],
        region_iso: Optional[str],
        city_name: Optional[str],
    ) -> bool:
        if self.country_iso is not None:
            if country_iso is None or country_iso.upper() != self.country_iso.upper():
                return False
        if self.region_iso is not None:
            if region_iso is None or region_iso.upper() != self.region_iso.upper():
                return False
        if self.city_pattern is not None:
            if city_name is None:
                return False
            if self.city_pattern.lower() not in city_name.lower():
                return False
        return True


def extract_geo_from_city_data(data: dict) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[float],
    Optional[float],
]:
    """Extract geographic info from city database record."""
    country = data.get("country", {})
    country_iso = country.get("iso_code")
    country_names = country.get("names", {})
    country_name = country_names.get("en")

    subdivisions = data.get("subdivisions", [])
    region_iso = None
    region_name = None
    if subdivisions:
        first_sub = subdivisions[0]
        region_iso = first_sub.get("iso_code")
        region_names = first_sub.get("names", {})
        region_name = region_names.get("en")

    city_data = data.get("city", {})
    city_names = city_data.get("names", {})
    city_name = city_names.get("en")

    postal = data.get("postal", {})
    postal_code = postal.get("code")

    location = data.get("location", {})
    latitude = location.get("latitude")
    longitude = location.get("longitude")

    return country_iso, country_name, region_iso, region_name, city_name, postal_code, latitude, longitude


def network_to_int_range(network_str: str) -> Optional[Tuple[int, int, int]]:
    """Convert network string to (start_int, end_int, version) tuple."""
    try:
        net = ipaddress.ip_network(network_str, strict=False)
        start = int(net.network_address)
        end = int(net.broadcast_address)
        return (start, end, net.version)
    except ValueError:
        return None


class NetworkIntervalIndex:
    """
    Fast interval index for checking if an IP range overlaps with any indexed networks.
    Uses sorted lists + binary search for O(log n) lookups.
    """

    def __init__(self):
        self._v4_intervals: List[Tuple[int, int, int, str]] = []
        self._v6_intervals: List[Tuple[int, int, int, str]] = []
        self._v4_starts: List[int] = []
        self._v6_starts: List[int] = []
        self._built = False

    def add(self, network_str: str, asn_num: int, asn_org: str) -> None:
        result = network_to_int_range(network_str)
        if result is None:
            return
        start, end, version = result

        if version == 4:
            self._v4_intervals.append((start, end, asn_num, asn_org))
        else:
            self._v6_intervals.append((start, end, asn_num, asn_org))

    def build(self) -> None:
        """Sort intervals and build search indices. Must call after all adds."""
        self._v4_intervals.sort(key=lambda x: x[0])
        self._v6_intervals.sort(key=lambda x: x[0])
        self._v4_starts = [iv[0] for iv in self._v4_intervals]
        self._v6_starts = [iv[0] for iv in self._v6_intervals]
        self._built = True

    def find_overlapping(self, network_str: str) -> Optional[Tuple[int, str]]:
        """
        Find if the given network overlaps with any indexed network.
        Returns (asn_num, asn_org) if found, None otherwise.
        """
        if not self._built:
            raise RuntimeError("Must call build() before queries")

        result = network_to_int_range(network_str)
        if result is None:
            return None

        query_start, query_end, version = result

        if version == 4:
            intervals = self._v4_intervals
            starts = self._v4_starts
        else:
            intervals = self._v6_intervals
            starts = self._v6_starts

        if not intervals:
            return None

        right_idx = bisect_right(starts, query_end)

        for i in range(right_idx - 1, -1, -1):
            iv_start, iv_end, asn_num, asn_org = intervals[i]

            if iv_end < query_start:
                if iv_start < query_start - (1 << 24):
                    break
                continue

            if iv_start <= query_end and iv_end >= query_start:
                return (asn_num, asn_org)

        return None

    def __len__(self) -> int:
        return len(self._v4_intervals) + len(self._v6_intervals)


def search_networks(
    asn_db_path: Path,
    city_db_path: Path,
    criteria: SearchCriteria,
    progress_callback=None,
) -> Iterator[SearchResult]:
    """
    Search for networks matching the given criteria.
    Uses interval indexing for fast ASN overlap lookups.
    """
    count = 0

    def should_include_version(version: int) -> bool:
        if criteria.ipv4_only and version != 4:
            return False
        if criteria.ipv6_only and version != 6:
            return False
        return True

    def get_network_version(network_str: str) -> Optional[int]:
        try:
            net = ipaddress.ip_network(network_str, strict=False)
            return net.version
        except ValueError:
            return None

    if criteria.asn is not None or criteria.asn_org_pattern is not None:
        if progress_callback:
            progress_callback("Building ASN network index...")

        asn_index = NetworkIntervalIndex()

        with maxminddb.open_database(str(asn_db_path)) as asn_reader:
            for network_str, data in asn_reader:
                if not data:
                    continue

                version = get_network_version(network_str)
                if version is None or not should_include_version(version):
                    continue

                asn_num = data.get("autonomous_system_number")
                asn_org = data.get("autonomous_system_organization", "")

                if criteria.matches_asn(asn_num, asn_org):
                    asn_index.add(network_str, asn_num, asn_org)

        asn_index.build()

        if progress_callback:
            progress_callback(f"Indexed {len(asn_index)} ASN networks, scanning City database...")

        if len(asn_index) == 0:
            if progress_callback:
                progress_callback("No matching ASN networks found")
            return

        if criteria.country_iso or criteria.region_iso or criteria.city_pattern:
            with maxminddb.open_database(str(city_db_path)) as city_reader:
                for network_str, data in city_reader:
                    if criteria.limit is not None and count >= criteria.limit:
                        return

                    if not data:
                        continue

                    version = get_network_version(network_str)
                    if version is None or not should_include_version(version):
                        continue

                    asn_match = asn_index.find_overlapping(network_str)
                    if asn_match is None:
                        continue

                    asn_num, asn_org = asn_match

                    (
                        country_iso,
                        country_name,
                        region_iso,
                        region_name,
                        city_name,
                        postal_code,
                        latitude,
                        longitude,
                    ) = extract_geo_from_city_data(data)

                    if not criteria.matches_geo(country_iso, region_iso, city_name):
                        continue

                    count += 1
                    yield SearchResult(
                        network=network_str,
                        asn_number=asn_num,
                        asn_org=asn_org,
                        country_iso=country_iso,
                        country_name=country_name,
                        region_iso=region_iso,
                        region_name=region_name,
                        city_name=city_name,
                        postal_code=postal_code,
                        latitude=latitude,
                        longitude=longitude,
                    )
        else:
            with maxminddb.open_database(str(asn_db_path)) as asn_reader:
                with maxminddb.open_database(str(city_db_path)) as city_reader:
                    for network_str, data in asn_reader:
                        if criteria.limit is not None and count >= criteria.limit:
                            return

                        if not data:
                            continue

                        version = get_network_version(network_str)
                        if version is None or not should_include_version(version):
                            continue

                        asn_num = data.get("autonomous_system_number")
                        asn_org = data.get("autonomous_system_organization")

                        if not criteria.matches_asn(asn_num, asn_org):
                            continue

                        country_iso = country_name = region_iso = region_name = None
                        city_name = postal_code = None
                        latitude = longitude = None

                        try:
                            net = ipaddress.ip_network(network_str, strict=False)
                            rep_ip = str(net.network_address)
                            geo_data = city_reader.get(rep_ip)
                            if geo_data:
                                (
                                    country_iso,
                                    country_name,
                                    region_iso,
                                    region_name,
                                    city_name,
                                    postal_code,
                                    latitude,
                                    longitude,
                                ) = extract_geo_from_city_data(geo_data)
                        except Exception:
                            pass

                        count += 1
                        yield SearchResult(
                            network=network_str,
                            asn_number=asn_num,
                            asn_org=asn_org,
                            country_iso=country_iso,
                            country_name=country_name,
                            region_iso=region_iso,
                            region_name=region_name,
                            city_name=city_name,
                            postal_code=postal_code,
                            latitude=latitude,
                            longitude=longitude,
                        )

    else:
        if progress_callback:
            progress_callback("Scanning City database...")

        with maxminddb.open_database(str(asn_db_path)) as asn_reader:
            with maxminddb.open_database(str(city_db_path)) as city_reader:
                for network_str, data in city_reader:
                    if criteria.limit is not None and count >= criteria.limit:
                        return

                    if not data:
                        continue

                    version = get_network_version(network_str)
                    if version is None or not should_include_version(version):
                        continue

                    (
                        country_iso,
                        country_name,
                        region_iso,
                        region_name,
                        city_name,
                        postal_code,
                        latitude,
                        longitude,
                    ) = extract_geo_from_city_data(data)

                    if not criteria.matches_geo(country_iso, region_iso, city_name):
                        continue

                    asn_num = None
                    asn_org = None
                    try:
                        net = ipaddress.ip_network(network_str, strict=False)
                        rep_ip = str(net.network_address)
                        asn_data = asn_reader.get(rep_ip)
                        if asn_data:
                            asn_num = asn_data.get("autonomous_system_number")
                            asn_org = asn_data.get("autonomous_system_organization")
                    except Exception:
                        pass

                    count += 1
                    yield SearchResult(
                        network=network_str,
                        asn_number=asn_num,
                        asn_org=asn_org,
                        country_iso=country_iso,
                        country_name=country_name,
                        region_iso=region_iso,
                        region_name=region_name,
                        city_name=city_name,
                        postal_code=postal_code,
                        latitude=latitude,
                        longitude=longitude,
                    )


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


def render_search_pretty(results: Sequence[SearchResult], out: TextIO) -> None:
    for idx, r in enumerate(results):
        if idx:
            out.write("\n")

        out.write(f"{r.network}\n")
        if r.asn_number is not None:
            if r.asn_org:
                out.write(f"  ASN:    AS{r.asn_number} ({r.asn_org})\n")
            else:
                out.write(f"  ASN:    AS{r.asn_number}\n")

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

        if r.postal_code:
            out.write(f"  Postal: {r.postal_code}\n")

        if r.latitude is not None and r.longitude is not None:
            out.write(f"  Coord:  {r.latitude:.6f}, {r.longitude:.6f}\n")


def render_search_table(results: Sequence[SearchResult], out: TextIO, header: bool = True) -> None:
    headers = ["Network", "ASN", "Org", "Country", "Region", "City"]

    rows: List[List[str]] = []
    for r in results:
        asn = f"AS{r.asn_number}" if r.asn_number is not None else ""
        org = r.asn_org or ""
        country = r.country_iso or ""
        region = r.region_iso or ""
        city = r.city_name or ""
        rows.append([r.network, asn, org, country, region, city])

    def trunc(x: str, n: int) -> str:
        if len(x) <= n:
            return x
        return x[: n - 3] + "..."

    for row in rows:
        row[2] = trunc(row[2], 35)
        row[5] = trunc(row[5], 20)

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


def render_search_json(results: Sequence[SearchResult], out: TextIO, pretty: bool) -> None:
    payload = [r.to_dict() for r in results]
    if pretty:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")
    else:
        json.dump(payload, out, separators=(",", ":"), sort_keys=True)
        out.write("\n")


def render_search_jsonl(results: Sequence[SearchResult], out: TextIO) -> None:
    for r in results:
        out.write(json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def render_search_csv(results: Sequence[SearchResult], out: TextIO, header: bool = True) -> None:
    writer = csv.DictWriter(out, fieldnames=list(SEARCH_CSV_FIELDS))
    if header:
        writer.writeheader()

    for r in results:
        d = r.to_dict()
        row = {k: d.get(k) for k in SEARCH_CSV_FIELDS}
        writer.writerow(row)


def render_search_networks_only(results: Sequence[SearchResult], out: TextIO) -> None:
    """Output just the network CIDRs, one per line."""
    for r in results:
        out.write(f"{r.network}\n")


def parse_asn(value: str) -> int:
    """Parse ASN from string like 'AS63949' or '63949'."""
    value = value.strip().upper()
    if value.startswith("AS"):
        value = value[2:]
    return int(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GeoIP lookup and search tool using MaxMind GeoLite2 ASN and City databases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Look up single IP
  %(prog)s 8.8.8.8

  # Look up multiple IPs
  %(prog)s 8.8.8.8 1.1.1.1 9.9.9.9

  # Look up multiple IPs, sort by ASN then IP
  %(prog)s 8.8.8.8 1.1.1.1 9.9.9.9 --sort asn,ip

  # Look up multiple IPs, preserve input order
  %(prog)s 8.8.8.8 1.1.1.1 9.9.9.9 --sort input

  # Search for networks in Georgia (GA) belonging to AS63949
  %(prog)s --search --asn AS63949 --region GA

  # Search for all networks in Atlanta
  %(prog)s --search --city Atlanta --country US

  # Search and output just network CIDRs
  %(prog)s --search --asn 63949 --region GA --format networks

  # Search with limit, sort by ASN descending then network ascending
  %(prog)s --search --asn 63949 --limit 100 --sort -asn,network
""",
    )

    mode_group = parser.add_argument_group("mode")
    mode_group.add_argument(
        "--search",
        action="store_true",
        help="Search mode: find networks matching criteria instead of looking up IPs",
    )

    lookup_group = parser.add_argument_group("lookup options")
    lookup_group.add_argument("ips", nargs="*", help="IP address(es) to look up")
    lookup_group.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Read IPs from file (one per line). Use '-' to read from stdin.",
    )

    search_group = parser.add_argument_group("search options")
    search_group.add_argument(
        "--asn",
        type=str,
        help="Filter by ASN number (e.g., AS63949 or 63949)",
    )
    search_group.add_argument(
        "--asn-org",
        type=str,
        help="Filter by ASN organization name (substring match)",
    )
    search_group.add_argument(
        "--country",
        type=str,
        help="Filter by country ISO code (e.g., US)",
    )
    search_group.add_argument(
        "--region",
        type=str,
        help="Filter by region/state ISO code (e.g., GA, CA, TX)",
    )
    search_group.add_argument(
        "--city",
        type=str,
        help="Filter by city name (substring match)",
    )
    search_group.add_argument(
        "--ipv6",
        action="store_true",
        help="Include IPv6 networks (default: IPv4 only)",
    )
    search_group.add_argument(
        "--ipv6-only",
        action="store_true",
        help="Only include IPv6 networks",
    )
    search_group.add_argument(
        "--limit",
        type=int,
        help="Maximum number of results to return",
    )

    db_group = parser.add_argument_group("database options")
    db_group.add_argument(
        "--asn-db",
        type=Path,
        default=DEFAULT_ASN_DB,
        help=f"Path to GeoLite2 ASN database (default: {DEFAULT_ASN_DB})",
    )
    db_group.add_argument(
        "--city-db",
        type=Path,
        default=DEFAULT_CITY_DB,
        help=f"Path to GeoLite2 City database (default: {DEFAULT_CITY_DB})",
    )

    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--format",
        choices=("pretty", "table", "json", "jsonl", "csv", "networks"),
        default="pretty",
        help="Output format (default: pretty). 'networks' outputs just CIDRs (search mode only)",
    )
    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write output to file (default: stdout). Use '-' for stdout.",
    )
    output_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include additional fields in pretty/table output",
    )
    output_group.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print header row (csv/table)",
    )
    output_group.add_argument(
        "--compact-json",
        action="store_true",
        help="For --format json, do not pretty-print",
    )
    output_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress messages (search mode)",
    )
    output_group.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any IP is invalid or a lookup errors out",
    )
    output_group.add_argument(
        "--sort",
        type=str,
        default=None,
        help=(
            "Sort results by field(s). Comma-separated; prefix with '-' for descending. "
            "Lookup default: ip. Search default: network. "
            "Use --sort none or --sort input to disable sorting. "
            "Examples: --sort ip ; --sort asn,ip ; --sort -country_iso,city_name"
        ),
    )
    output_group.add_argument(
        "--sort-desc",
        action="store_true",
        help="Sort descending by default for fields without a leading +/-.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    for db_path, label in ((args.asn_db, "ASN"), (args.city_db, "City")):
        if not db_path.exists():
            print(f"error: {label} database not found: {db_path}", file=sys.stderr)
            return 2

    if args.search:
        if not any([args.asn, args.asn_org, args.country, args.region, args.city]):
            print(
                "error: search mode requires at least one filter (--asn, --asn-org, --country, --region, --city)",
                file=sys.stderr,
            )
            return 2

        asn_num = None
        if args.asn:
            try:
                asn_num = parse_asn(args.asn)
            except ValueError:
                print(f"error: invalid ASN format: {args.asn}", file=sys.stderr)
                return 2

        criteria = SearchCriteria(
            asn=asn_num,
            asn_org_pattern=args.asn_org,
            country_iso=args.country,
            region_iso=args.region,
            city_pattern=args.city,
            ipv4_only=not args.ipv6 and not args.ipv6_only,
            ipv6_only=args.ipv6_only,
            limit=args.limit,
        )

        def progress(msg: str) -> None:
            if not args.quiet:
                print(msg, file=sys.stderr)

        try:
            results = list(search_networks(args.asn_db, args.city_db, criteria, progress_callback=progress))
        except Exception as e:
            print(f"error: search failed: {e}", file=sys.stderr)
            return 2

        try:
            results = sort_search_results(results, args.sort, args.sort_desc)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        if not args.quiet:
            print(f"Found {len(results)} matching networks", file=sys.stderr)

        with open_output(args.output) as out:
            header = not args.no_header

            if args.format == "pretty":
                render_search_pretty(results, out=out)
                if results:
                    out.write("\n")
            elif args.format == "table":
                render_search_table(results, out=out, header=header)
            elif args.format == "json":
                render_search_json(results, out=out, pretty=(not args.compact_json))
            elif args.format == "jsonl":
                render_search_jsonl(results, out=out)
            elif args.format == "csv":
                render_search_csv(results, out=out, header=header)
            elif args.format == "networks":
                render_search_networks_only(results, out=out)

        return 0

    try:
        ip_list = iter_ips_from_sources(args.ips, args.file)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not ip_list:
        print("error: no IP addresses provided (use positional args or -f FILE)", file=sys.stderr)
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

    try:
        results = sort_geoip_results(results, args.sort, args.sort_desc)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
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
        elif args.format == "networks":
            print("error: --format networks is only available in search mode", file=sys.stderr)
            return 2

    if strict_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

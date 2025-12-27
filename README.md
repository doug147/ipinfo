# ipinfo - GeoIP Lookup Tool

A command-line tool for looking up IP address geolocation and ASN information using MaxMind GeoLite2 databases.

## Features

- Look up geolocation data (country, region, city, coordinates, timezone)
- Look up ASN (Autonomous System Number) information
- Multiple output formats: pretty, table, JSON, JSONL, CSV
- Batch lookups from file or stdin
- IPv4 and IPv6 support

## Prerequisites

- Python 3.8+
- MaxMind GeoLite2 ASN and City databases
- MaxMind account (free)

## Installation

### 1. Install Python Dependencies

```bash
pip install geoip2
```

### 2. Set Up MaxMind GeoLite2 Databases

#### Create a MaxMind Account and License Key

1. Sign up for a free MaxMind account at [maxmind.com](https://www.maxmind.com/en/geolite2/signup)

2. Create a license key:
   - Go to `https://www.maxmind.com/en/accounts/<your-account-id>/license-key/create`
   - Or navigate to: **Account** → **Manage License Keys** → **Generate New License Key**
   - Save your license key securely

#### Install and Configure GeoIP Update Tool

Follow the [official MaxMind documentation](https://dev.maxmind.com/geoip/updating-databases/) to set up automatic database updates.

**On Debian/Ubuntu:**

```bash
sudo add-apt-repository ppa:maxmind/ppa
sudo apt update
sudo apt install geoipupdate
```

**On macOS (Homebrew):**

```bash
brew install geoipupdate
```

**On other systems:**

Download from [GitHub releases](https://github.com/maxmind/geoipupdate/releases).

#### Configure GeoIP Update

Edit `/etc/GeoIP.conf` (or `~/.config/GeoIP.conf`):

```conf
AccountID YOUR_ACCOUNT_ID
LicenseKey YOUR_LICENSE_KEY
EditionIDs GeoLite2-ASN GeoLite2-City
DatabaseDirectory /var/lib/GeoIP
```

#### Download the Databases

```bash
sudo mkdir -p /var/lib/GeoIP
sudo geoipupdate
```

To set up automatic updates, add a cron job or systemd timer:

```bash
# Example cron entry (updates weekly)
0 0 * * 0 /usr/bin/geoipupdate
```

### 3. Install the Script

#### Option A: Install to PATH (Recommended)

```bash
# Copy script to a location in your PATH
sudo cp ipinfo.py /usr/local/bin/ipinfo
sudo chmod +x /usr/local/bin/ipinfo
```

#### Option B: Create a Symlink

```bash
# Make the script executable
chmod +x ipinfo.py

# Create symlink (adjust path to where you saved the script)
sudo ln -s /path/to/ipinfo.py /usr/local/bin/ipinfo
```

#### Option C: Shell Alias

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias ipinfo='python3 /path/to/ipinfo.py'
```

Then reload your shell:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Usage

### Basic Lookups

```bash
# Single IP
ipinfo 8.8.8.8

# Multiple IPs
ipinfo 8.8.8.8 1.1.1.1 208.67.222.222

# IPv6
ipinfo 2001:4860:4860::8888
```

### Output Formats

```bash
# Pretty print (default)
ipinfo 8.8.8.8

# Compact table
ipinfo --format table 8.8.8.8 1.1.1.1

# JSON array
ipinfo --format json 8.8.8.8

# Newline-delimited JSON
ipinfo --format jsonl 8.8.8.8 1.1.1.1

# CSV
ipinfo --format csv 8.8.8.8 1.1.1.1
```

### Batch Lookups

```bash
# From file
ipinfo -f ip_list.txt

# From stdin
cat ip_list.txt | ipinfo -f -

# Pipe from other commands
grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' access.log | sort -u | ipinfo -f -
```

### Additional Options

```bash
# Verbose output (includes network info, anycast status, etc.)
ipinfo -v 8.8.8.8

# Write to file
ipinfo -o results.json --format json 8.8.8.8

# Custom database paths
ipinfo --asn-db /path/to/GeoLite2-ASN.mmdb --city-db /path/to/GeoLite2-City.mmdb 8.8.8.8

# No header row (for CSV/table)
ipinfo --format csv --no-header 8.8.8.8

# Compact JSON (no pretty-printing)
ipinfo --format json --compact-json 8.8.8.8

# Strict mode (exit non-zero on errors)
ipinfo --strict 8.8.8.8
```

## Example Output

### Pretty Format (Default)

```
8.8.8.8
  ASN:    AS15169 (GOOGLE)
  Place:  Mountain View, California (CA), United States (US)
  Coord:  37.405991, -122.078514 (accuracy ~1000 km)
  TZ:     America/Los_Angeles
```

### Table Format

```
IP        ASN      Org     Country  Region  City            TZ
--------  -------  ------  -------  ------  --------------  ------------------
8.8.8.8   AS15169  GOOGLE  US       CA      Mountain View   America/Los_Angeles
1.1.1.1   AS13335  CLOUDF  AU       NSW     Sydney          Australia/Sydney
```

### JSON Format

```json
[
  {
    "asn": "AS15169",
    "asn_network": "8.8.8.0/24",
    "asn_number": 15169,
    "asn_org": "GOOGLE",
    "city_name": "Mountain View",
    "country_iso": "US",
    "country_name": "United States",
    "ip": "8.8.8.8",
    "latitude": 37.405991,
    "longitude": -122.078514,
    "status": "ok",
    ...
  }
]
```

## Troubleshooting

### Database Not Found

If you see `error: ASN database not found`, ensure:

1. The databases are downloaded: `sudo geoipupdate`
2. They're in the expected location: `/var/lib/GeoIP/`
3. Or specify custom paths with `--asn-db` and `--city-db`

### Permission Denied

```bash
sudo chmod 644 /var/lib/GeoIP/*.mmdb
```

### License Key Issues

Verify your `/etc/GeoIP.conf` has the correct AccountID and LicenseKey from your [MaxMind account](https://www.maxmind.com/en/accounts).

## License

This tool uses MaxMind GeoLite2 data. GeoLite2 databases are subject to the [MaxMind GeoLite2 End User License Agreement](https://www.maxmind.com/en/geolite2/eula).

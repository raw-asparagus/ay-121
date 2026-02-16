# Quickstart: RTL-SDR & ugradio Setup (macOS + Linux)

Guide for installing `librtlsdr`, `pyrtlsdr`, and `ugradio` into a local virtual environment.

Based on: [Setting Up Your Raspberry Pi – UC Berkeley Radio Lab](https://casper.astro.berkeley.edu/astrobaki/index.php/Setting_Up_Your_Raspberry_Pi)

## Prerequisites

- macOS with [Homebrew](https://brew.sh) installed
- A Python virtual environment at `.venv/` in your project root

## Install Build Dependencies

```bash
brew install autoconf automake libtool libusb pkg-config
```

## Setup

Run the following from your **project root** (the directory containing `.venv/`):

```bash
VENV_DIR=".venv"
SRC_DIR="src"
PREFIX="$(realpath $VENV_DIR)"

mkdir -p "$SRC_DIR"

# Build and install librtlsdr into the venv
cd "$SRC_DIR"
git clone https://github.com/AaronParsons/librtlsdr.git
cd librtlsdr
autoreconf -i
./configure --prefix="$PREFIX"
make
make install
cd ../..

export DYLD_LIBRARY_PATH="$PREFIX/lib:$DYLD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# Install pyrtlsdr
cd "$SRC_DIR"
git clone https://github.com/AaronParsons/pyrtlsdr.git
cd pyrtlsdr
../../${VENV_DIR}/bin/pip install .
cd ../..

# Install ugradio
cd "$SRC_DIR"
git clone https://github.com/AaronParsons/ugradio.git
cd ugradio/ugradio_code
../../../${VENV_DIR}/bin/pip install .
cd ../../..

# Persist library path in venv activation
echo 'export DYLD_LIBRARY_PATH="'"$PREFIX"'/lib:$DYLD_LIBRARY_PATH"' >> .venv/bin/activate
```

## Verify

```bash
source .venv/bin/activate
python -c "import rtlsdr; print('rtlsdr OK')"
python -c "import ugradio; print('ugradio OK')"
```

## Notes

- All C libraries and Python packages are installed into `.venv/` and `src/` — nothing is installed system-wide (aside from Homebrew build tools).
- To uninstall, simply delete the `src/` directory and recreate your virtual environment.

---

## Linux Setup (Ubuntu/Debian/WSL)

### Prerequisites

- Ubuntu/Debian (or WSL) with `apt`
- A Python virtual environment at `.venv/` in your project root

### Install Build Dependencies

```bash
sudo apt update
sudo apt install -y build-essential git autoconf automake libtool pkg-config libusb-1.0-0-dev
```

### Setup

Run the following from your **project root** (the directory containing `.venv/`):

```bash
VENV_DIR=".venv"
SRC_DIR="src"
PREFIX="$(realpath "$VENV_DIR")"
LIBUSB_LIBDIR="$(pkg-config --variable=libdir libusb-1.0)"

mkdir -p "$SRC_DIR"

# Build and install librtlsdr into the venv
cd "$SRC_DIR"
git clone https://github.com/AaronParsons/librtlsdr.git
cd librtlsdr
autoreconf -i
./configure --prefix="$PREFIX"
make
make install
cd ../..

export LD_LIBRARY_PATH="$PREFIX/lib:$LIBUSB_LIBDIR:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# Install pyrtlsdr
cd "$SRC_DIR"
git clone https://github.com/AaronParsons/pyrtlsdr.git
cd pyrtlsdr
../../${VENV_DIR}/bin/pip install .
cd ../..

# Install ugradio
cd "$SRC_DIR"
git clone https://github.com/AaronParsons/ugradio.git
cd ugradio/ugradio_code
../../../${VENV_DIR}/bin/pip install .
cd ../../..

# Persist library path in venv activation
echo 'export LD_LIBRARY_PATH="'"$PREFIX"'/lib:'"$LIBUSB_LIBDIR"':${LD_LIBRARY_PATH:-}"' >> .venv/bin/activate
```

### Verify

```bash
source .venv/bin/activate
python -c "import rtlsdr; print('rtlsdr OK')"
python -c "import importlib; importlib.import_module('ugradio.sdr'); print('ugradio.sdr OK')"
```

### Notes

- On Linux, `DYLD_LIBRARY_PATH` is ignored. Use `LD_LIBRARY_PATH`.
- `librtlsdr` is a C library (not a pip package), so `pip show librtlsdr` returning "not found" is expected.

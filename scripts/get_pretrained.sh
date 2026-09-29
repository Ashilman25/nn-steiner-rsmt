#!/usr/bin/env bash
# Download the authors' pretrained model (m=15, kb=4) to work/models/pretrained/.
# It is ~770 MB and stored with Git LFS upstream, so we fetch the file directly
# (no git-lfs needed). On Colab work/models lives on Drive, so this runs once.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DEST="$WORK/models/pretrained"
# Saved without "=" in the name: Hydra can't parse "=" inside command-line values.
MODEL="$DEST/m15_kb4.pt"
URL="https://media.githubusercontent.com/media/ABKGroup/NN-Steiner/$UPSTREAM_SHA/models/m=15_kb=4.tar.gz"
SHA256="2b5f2b6b24c651aa6aedc90788b8fe3bbc70bb83ea7a21a528984e6b910ef215"

if [ -f "$MODEL" ]; then
    echo "Pretrained model already at $MODEL"
    exit 0
fi

mkdir -p "$DEST"
TARBALL="$DEST/m=15_kb=4.tar.gz"
echo "==> Downloading pretrained model (~770 MB)"
curl -fL --progress-bar -o "$TARBALL" "$URL"

echo "==> Verifying checksum"
if command -v sha256sum >/dev/null; then
    ACTUAL="$(sha256sum "$TARBALL" | cut -d' ' -f1)"
else
    ACTUAL="$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)"
fi
[ "$ACTUAL" = "$SHA256" ] || { echo "!! Checksum mismatch; delete $TARBALL and retry"; exit 1; }

echo "==> Extracting"
# The archive holds models/m=15_kb=4.pt; drop the models/ folder and rename.
tar -xzf "$TARBALL" -C "$DEST" --strip-components=1
mv "$DEST/m=15_kb=4.pt" "$MODEL"
rm "$TARBALL"
[ -f "$MODEL" ] || { echo "!! Expected $MODEL after extracting; found:"; ls -la "$DEST"; exit 1; }
echo "Pretrained model ready: $MODEL"

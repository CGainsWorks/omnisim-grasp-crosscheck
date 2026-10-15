#!/usr/bin/env bash
# Build the OmniSim runtime image from source, at the commit that was audited.
#
# The prebuilt Windows installer and the published GHCR image are binaries
# built elsewhere, which a source read cannot vouch for. This builds the
# upstream Dockerfile from a clone checked against a pinned commit hash, so
# what runs is what was read. See docs/security-review.md.
#
#   bash scripts/build_omnisim_image.sh [workdir]
set -euo pipefail

OMNISIM_TAG="v9.2.4"
OMNISIM_COMMIT="28e00e1f1a2d7932743911a697dc8f63ed5eb81a"
IMAGE="${OMNISIM_IMAGE:-omnisim-local:${OMNISIM_TAG}}"
WORK="${1:-${TMPDIR:-/tmp}/omnisim-src}"

if [ ! -d "$WORK/.git" ]; then
    # autocrlf off: a Windows checkout otherwise hands the Linux build CRLF
    # makefiles and shell scripts.
    git -c core.autocrlf=false clone --quiet --depth 1 --branch "$OMNISIM_TAG" \
        https://github.com/omnilink-tech/omnisim.git "$WORK"
fi

actual="$(git -C "$WORK" rev-parse HEAD)"
if [ "$actual" != "$OMNISIM_COMMIT" ]; then
    echo "refusing to build: $WORK is at $actual, expected $OMNISIM_COMMIT" >&2
    echo "the tag has moved since the audit; review the difference before changing the pin" >&2
    exit 1
fi

docker build -f "$WORK/docker/Dockerfile.runtime" -t "$IMAGE" \
    --label "org.opencontainers.image.revision=$OMNISIM_COMMIT" "$WORK"
echo "built $IMAGE from $OMNISIM_COMMIT"

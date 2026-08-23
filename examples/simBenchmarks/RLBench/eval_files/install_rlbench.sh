#!/usr/bin/env bash
# RLBench (CoppiuliaSim) environment setup for starVLA.
# Requires: ~/.local mamba env `starvla`, and a working Linux X / GL.
set -euo pipefail

echo "== RLBench install =="
echo "1) CoppeliaSim MUST be 4.1.0 (RLBench 1.2.0 / PyRep 4.1.0.3 are built for it)."
echo "   Do NOT use the 4.10.0 build — the ABI crashes RLBench's Scene init."
COPPELIA_URL="https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
COPPELIA_ROOT="${COPPELIA_ROOT:-$HOME/CoppeliaSim_4_1}"
if [ ! -x "${COPPELIA_ROOT}/coppeliaSim" ]; then
  echo "[install] downloading/extracting CoppeliaSim 4.1.0 -> ${COPPELIA_ROOT}"
  TAR=/tmp/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
  curl -L --fail -o "${TAR}" "${COPPELIA_URL}"
  mkdir -p "${COPPELIA_ROOT}"
  tar -xf "${TAR}" -C "${COPPELIA_ROOT}" --strip-components 1
  rm -f "${TAR}"
fi

echo "2) pip deps into the starvla env"
pip install PyRep==4.1.0.3 rlbench==1.2.0 av imageio-ffmpeg 2>/dev/null || \
  pip install av imageio-ffmpeg

cat <<EOF
3) Set these env vars in every shell that launches RLBench (exactly ONE
   CoppeliaSim version on LD_LIBRARY_PATH — mixing 4.10 + 4.1 breaks Qt ABI):

    export COPPELIASIM_ROOT=${COPPELIA_ROOT}
    export LD_LIBRARY_PATH=\$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT/platforms
    export DISPLAY=:0        # or a GLX-capable virtual X (xvfb)

4) Headless machines need a GPU/GL provider. Without a real GL context the
   camera renderer segfaults (signal 11, QOpenGLContext). Use xvfb + Mesa or
   a GPU node.
EOF
echo "[install] done."
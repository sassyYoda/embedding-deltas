#!/usr/bin/env bash
# run_all.sh — process all in-scope sample videos with frozen parameters
# (Phase 6 / RUN-01..03)
#
# Frozen tuning parameters (locked in Phase 5 on tiger_woods, the longest in-scope video):
#   --height 1.5    --min-gap-sec 15.0   --merge-gap-sec 3.0
# These are the argparse defaults in pipeline.py — no flags needed.
#
# Usage: ./run_all.sh
# Output: output/reels/{video}_highlight.mp4 + output/timestamps/{video}.json per video
#
# IMPORTANT: this script uses tools/pipeline_from_cache.py (the cache-bypass dev tool)
# rather than pipeline.py directly. pipeline.py's CLIP-extraction step deadlocks on
# MPS for videos > ~3000 frames (see .planning/phases/05-orchestration-first-video/
# 05-01-SUMMARY.md "Known Limitation: MPS Deadlock"). The standalone extract.py harness
# DOES succeed (different process boundary), so embeddings are pre-extracted into
# output/cache/{video}_embeddings.npy and the cache-bypass loads them instead.
#
# To regenerate embeddings from scratch on a different machine where MPS works
# reliably, replace the loop body with:
#   .venv/bin/python3 pipeline.py "$VIDEO"
# AND first run `.venv/bin/python3 extract.py "$VIDEO"` to populate the cache.

set -euo pipefail
cd "$(dirname "$0")"

# In-scope videos (marcus_jordan dropped — see PROJECT.md / Phase 6 README).
VIDEOS=(
  videos/justin_timberlake.mp4
  videos/tiger_woods.mp4
  videos/test_assault_theft.mp4
  videos/test_missing_person.mp4
)

echo "═══════════════════════════════════════════════════════════════"
echo " Body Cam Highlight Reel — batch run on ${#VIDEOS[@]} videos"
echo " Frozen params: --height 1.5 --min-gap-sec 15.0 --merge-gap-sec 3.0"
echo "═══════════════════════════════════════════════════════════════"

for VIDEO in "${VIDEOS[@]}"; do
  echo ""
  echo "▶ $VIDEO"
  .venv/bin/python3 tools/pipeline_from_cache.py "$VIDEO"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Done. ${#VIDEOS[@]} reels in output/reels/, ${#VIDEOS[@]} JSONs in output/timestamps/."
echo "═══════════════════════════════════════════════════════════════"
ls -la output/reels/ output/timestamps/

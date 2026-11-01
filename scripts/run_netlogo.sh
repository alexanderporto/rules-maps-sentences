#!/usr/bin/env sh
# Runs the three NetLogo designs headless with BehaviorSpace, using the setup files in
# netlogo/experiments/. The designs match the Python scripts (run_abm.py, run_negation.py,
# run_quantifiers.py). Each run sets `random-seed behaviorspace-run-number` before `setup`, so the
# output is the same for any number of threads.
#
# Usage: sh scripts/run_netlogo.sh "<NetLogo installation directory>" <threads> [model ...]
#   e.g. sh scripts/run_netlogo.sh "/c/Program Files/NetLogo 7.0.4" 11 negation iconographers quantifiers
# Written for and run with NetLogo 7.0.4 (its bundled Java runtime).
# Outputs: results/netlogo_<model>_table.csv.gz (BehaviorSpace table format, gzip-compressed)

set -eu

if [ "$#" -lt 3 ]; then
  echo "Usage: sh scripts/run_netlogo.sh \"<NetLogo directory>\" <threads> <model> [model ...]" >&2
  exit 2
fi

NETLOGO="$1"
THREADS="$2"
shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAVA="$NETLOGO/runtime/bin/java.exe"
[ -x "$JAVA" ] || JAVA="$NETLOGO/runtime/bin/java"
JAR="$(ls "$NETLOGO"/app/netlogo-*.jar | head -n 1)"

if [ ! -f "$JAR" ]; then
  echo "No netlogo-*.jar found in $NETLOGO/app." >&2
  exit 1
fi

for MODEL in "$@"; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') start $MODEL"
  "$JAVA" -XX:MaxRAMPercentage=50 -Dfile.encoding=UTF-8 \
    "-Dnetlogo.extensions.dir=$NETLOGO/extensions" \
    -classpath "$JAR" org.nlogo.headless.Main \
    --model "$ROOT/netlogo/$MODEL.nlogo" \
    --setup-file "$ROOT/netlogo/experiments/$MODEL.xml" \
    --table "$ROOT/results/netlogo_${MODEL}_table.csv" \
    --threads "$THREADS"
  gzip -n -f "$ROOT/results/netlogo_${MODEL}_table.csv"
  echo "$(date '+%Y-%m-%d %H:%M:%S') done $MODEL"
done

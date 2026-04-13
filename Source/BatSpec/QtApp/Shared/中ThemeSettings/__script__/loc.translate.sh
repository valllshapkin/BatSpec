#!/bin/sh

# # # = = = = = = = # # #
.     GolonSystemSH     .
# # # = = = = = = = # # #

set -e

ScriptDir="$(getScriptDir)"

python "$ScriptDir/loc.translate.py"
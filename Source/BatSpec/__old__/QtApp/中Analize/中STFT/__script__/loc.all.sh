#!/bin/sh

# # # = = = = = = = # # #
.     GolonSystemSH     .
# # # = = = = = = = # # #

set -e

cd "$ScriptDir"
./loc.find.sh
./loc.translate.sh
./loc.comp.sh
./loc.finderr.sh

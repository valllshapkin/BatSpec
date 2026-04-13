#!/bin/sh

    # # # = = = = = = = # # #
    .     GolonSystemSH     .
    # # # = = = = = = = # # #

set -e

ScriptDir="$(getScriptDir)"
Root="$(dirname "$(dirname "$ScriptDir")")"

echo '
eval "$('$ScriptDir/Pixi/import.sh')"
eval "$('$ScriptDir/VSCode/import.sh')"
eval "$('$ScriptDir/Qt/import.sh')"
export PYTHONPATH="'$Root/Source':/MainData/Repo/valllshapkin/AnimalSpecis/Source"
'


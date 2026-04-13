#!/bin/sh

    # # # = = = = = = = # # #
    .     GolonSystemSH     .
    # # # = = = = = = = # # #

set -e

ScriptDir="$(getScriptDir)"

echo '
. GolonSystemSH
require Pixi Main

pushd "'$ScriptDir'"
    eval "$(pixi shell-hook)"
popd
'




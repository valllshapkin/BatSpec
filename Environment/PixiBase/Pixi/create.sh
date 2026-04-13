#!/bin/sh

    # # # = = = = = = = # # #
    .     GolonSystemSH     .
    # # # = = = = = = = # # #

set -e

require Pixi Main

cd "$(getScriptDir)"

pixi install


#!/bin/sh

    # # # = = = = = = = # # #
    .     GolonSystemSH     .
    # # # = = = = = = = # # #

set -e

cd "$ScriptDir/.."
mkdir -p __assets__/translations
for ts_file in __locale__/this_*.ts; do
    qm_file="__assets__/translations/$(basename "$ts_file" .ts).qm"
    lrelease "$ts_file" -qm "$qm_file"
done

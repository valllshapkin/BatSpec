#!/bin/sh

    # # # = = = = = = = # # #
    .     GolonSystemSH     .
    # # # = = = = = = = # # #

set -e

cd "$ScriptDir/.."
python setup.py build_ext --inplace
rm -rf build
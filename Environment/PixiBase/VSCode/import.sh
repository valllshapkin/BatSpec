#!/bin/sh

    # # # = = = = = = = # # #
    .     GolonSystemSH     .
    # # # = = = = = = = # # #

set -e

echo '
. GolonSystemSH
require VSCode Main

editProject(){
    wrapper_code "'$(getScriptDir)/BatSpec.code-workspace'"
}

echo [Add] editProject
'



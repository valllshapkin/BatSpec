ScriptDir="$(cd "$(dirname "$0")" && pwd)"
cd "$ScriptDir/.."
mkdir -p __locale__
lupdate __init__.py -ts \
    __locale__/this_ru_RU.ts \
    __locale__/this_en_US.ts \
    __locale__/this_de_DE.ts \
    __locale__/this_fr_FR.ts \
    __locale__/this_es_ES.ts \
    __locale__/this_zh_CN.ts

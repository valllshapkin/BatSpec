#!/bin/sh

# # # = = = = = = = # # #
.     GolonSystemSH     .
# # # = = = = = = = # # #

set -e

cd "$ScriptDir/.."
mkdir -p __locale__

lupdate Widget.py -ts \
    __locale__/this_ru_RU.ts \
    __locale__/this_en_US.ts \
    __locale__/this_de_DE.ts \
    __locale__/this_fr_FR.ts \
    __locale__/this_es_ES.ts \
    __locale__/this_zh_CN.ts \
    __locale__/this_zh_TW.ts \
    __locale__/this_ja_JP.ts \
    __locale__/this_ko_KR.ts \
    __locale__/this_ar_SA.ts \
    __locale__/this_pt_PT.ts \
    __locale__/this_pt_BR.ts \
    __locale__/this_it_IT.ts \
    __locale__/this_nl_NL.ts \
    __locale__/this_pl_PL.ts \
    __locale__/this_tr_TR.ts \
    __locale__/this_vi_VN.ts \
    __locale__/this_hi_IN.ts \
    __locale__/this_he_IL.ts \
    __locale__/this_el_GR.ts \
    __locale__/this_sv_SE.ts \
    __locale__/this_da_DK.ts \
    __locale__/this_fi_FI.ts \
    __locale__/this_nb_NO.ts \
    __locale__/this_cs_CZ.ts \
    __locale__/this_sk_SK.ts \
    __locale__/this_hu_HU.ts \
    __locale__/this_ro_RO.ts \
    __locale__/this_bg_BG.ts \
    __locale__/this_uk_UA.ts \
    __locale__/this_ca_ES.ts
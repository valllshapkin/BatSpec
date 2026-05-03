from pathlib import Path
from PySide6.QtCore import QLocale
from W.PySide6.QtLocales import Locales

ScriptDir = Path(__file__).parent

AppLocales = Locales(
    SETTING_PATH = ScriptDir / "__assets__" / "LocalesSettings.ini"
)

SUPPORTED_LANGUAGES = [
    QLocale(QLocale.Language.English, QLocale.Country.UnitedStates),
    QLocale(QLocale.Language.Russian, QLocale.Country.Russia),
    QLocale(QLocale.Language.German, QLocale.Country.Germany),
    QLocale(QLocale.Language.French, QLocale.Country.France),
    QLocale(QLocale.Language.Spanish, QLocale.Country.Spain),
    QLocale(QLocale.Language.Chinese, QLocale.Country.China),
    QLocale(QLocale.Language.Chinese, QLocale.Country.Taiwan),
    QLocale(QLocale.Language.Japanese, QLocale.Country.Japan),
    QLocale(QLocale.Language.Korean, QLocale.Country.SouthKorea),
    QLocale(QLocale.Language.Arabic, QLocale.Country.SaudiArabia),
    QLocale(QLocale.Language.Portuguese, QLocale.Country.Portugal),
    QLocale(QLocale.Language.Portuguese, QLocale.Country.Brazil),
    QLocale(QLocale.Language.Italian, QLocale.Country.Italy),
    QLocale(QLocale.Language.Dutch, QLocale.Country.Netherlands),
    QLocale(QLocale.Language.Polish, QLocale.Country.Poland),
    QLocale(QLocale.Language.Turkish, QLocale.Country.Turkey),
    QLocale(QLocale.Language.Vietnamese, QLocale.Country.Vietnam),
    QLocale(QLocale.Language.Hindi, QLocale.Country.India),
    QLocale(QLocale.Language.Hebrew, QLocale.Country.Israel),
    QLocale(QLocale.Language.Greek, QLocale.Country.Greece),
    QLocale(QLocale.Language.Swedish, QLocale.Country.Sweden),
    QLocale(QLocale.Language.Danish, QLocale.Country.Denmark),
    QLocale(QLocale.Language.Finnish, QLocale.Country.Finland),
    QLocale(QLocale.Language.NorwegianBokmal, QLocale.Country.Norway),
    QLocale(QLocale.Language.Czech, QLocale.Country.Czechia),
    QLocale(QLocale.Language.Slovak, QLocale.Country.Slovakia),
    QLocale(QLocale.Language.Hungarian, QLocale.Country.Hungary),
    QLocale(QLocale.Language.Romanian, QLocale.Country.Romania),
    QLocale(QLocale.Language.Bulgarian, QLocale.Country.Bulgaria),
    QLocale(QLocale.Language.Ukrainian, QLocale.Country.Ukraine),
    QLocale(QLocale.Language.Catalan, QLocale.Country.Spain),
]

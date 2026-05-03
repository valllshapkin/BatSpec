# 1. С строковыми ключами
from typing import Any

from BatSpec.QtUp.Reactive import ReactiveDict


d1: ObservableDict[str, Any] = ObservableDict(name="Алексей", age=30)

# 2. С целочисленными ключами
d2: ObservableDict[int, str] = ObservableDict()
d2[1] = "Первый"
d2[42] = "Ответ"

# 3. С кортежами как ключами
d3: ObservableDict[tuple[int, int], str] = ObservableDict()
d3[(10, 20)] = "Точка"

# 4. Смешанный тип
d4 = ObservableDict[Any, Any]()
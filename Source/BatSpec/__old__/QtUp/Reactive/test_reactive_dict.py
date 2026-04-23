from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QObject

# Предполагается, что код ReactiveDict лежит в файле reactive_dict.py
from BatSpec.QtUp.Reactive.reactive_dict import ReactiveDict, make_reactive


# ==================== Фикстуры (Fixtures) ====================

@pytest.fixture
def rdict() -> ReactiveDict[str, int]:
    """Предоставляет свежий заполненный словарь для каждого теста."""
    return ReactiveDict({"a": 1, "b": 2})

@pytest.fixture
def empty_rdict() -> ReactiveDict[str, int]:
    """Предоставляет пустой словарь."""
    return ReactiveDict()

@pytest.fixture
def mocks(rdict: ReactiveDict) -> dict[str, MagicMock]:
    """
    Подключает Mock-объекты ко всем сигналам словаря.
    Позволяет проверять, сколько раз и с какими аргументами вызывался сигнал.
    """
    m = {
        "itemSet": MagicMock(),
        "itemRemoved": MagicMock(),
        "cleared": MagicMock(),
        "updated": MagicMock(),
        "changed": MagicMock(),
    }
    rdict.signals.itemSet.connect(m["itemSet"])
    rdict.signals.itemRemoved.connect(m["itemRemoved"])
    rdict.signals.cleared.connect(m["cleared"])
    rdict.signals.updated.connect(m["updated"])
    rdict.signals.changed.connect(m["changed"])
    return m


# ==================== Тесты базового поведения ====================

def test_initialization(rdict):
    """Проверка того, что ReactiveDict ведет себя как обычный dict."""
    assert isinstance(rdict, dict)
    assert len(rdict) == 2
    assert rdict["a"] == 1
    assert rdict.get("c", 100) == 100
    assert isinstance(rdict.signals, QObject)


# ==================== Одиночные мутации и No-op ====================

def test_setitem_new_key(rdict, mocks):
    """Добавление нового ключа испускает правильные сигналы."""
    rdict["c"] = 3
    
    mocks["itemSet"].assert_called_once_with("c", 3, None)
    mocks["changed"].assert_called_once()
    mocks["updated"].assert_not_called()  # updated только для массовых операций

def test_setitem_existing_key(rdict, mocks):
    """Изменение существующего ключа передает old_value."""
    rdict["a"] = 10
    
    mocks["itemSet"].assert_called_once_with("a", 10, 1)
    mocks["changed"].assert_called_once()

def test_setitem_noop_optimization(rdict, mocks):
    """ВАЖНО: Если значение не изменилось, сигналы не должны испускаться (No-op)."""
    rdict["a"] = 1  # Там уже лежит 1
    
    mocks["itemSet"].assert_not_called()
    mocks["changed"].assert_not_called()

def test_delitem_success(rdict, mocks):
    """Удаление существующего ключа."""
    del rdict["a"]
    
    mocks["itemRemoved"].assert_called_once_with("a", 1)
    mocks["changed"].assert_called_once()
    assert "a" not in rdict


# ==================== Exception Safety (Безопасность исключений) ====================

def test_delitem_exception_safety(rdict, mocks):
    """При удалении несуществующего ключа вылетает KeyError, сигналы МОЛЧАТ."""
    with pytest.raises(KeyError):
        del rdict["missing_key"]
    
    mocks["itemRemoved"].assert_not_called()
    mocks["changed"].assert_not_called()

def test_pop_exception_safety(rdict, mocks):
    """Аналогично для pop без default-значения."""
    with pytest.raises(KeyError):
        rdict.pop("missing_key")
    
    mocks["itemRemoved"].assert_not_called()
    mocks["changed"].assert_not_called()


# ==================== Методы pop, popitem, setdefault ====================

def test_pop_success(rdict, mocks):
    value = rdict.pop("b")
    assert value == 2
    mocks["itemRemoved"].assert_called_once_with("b", 2)
    mocks["changed"].assert_called_once()

def test_pop_with_default(rdict, mocks):
    """pop с default возвращает default и НЕ испускает сигналы, если ключа нет."""
    value = rdict.pop("missing", 99)
    assert value == 99
    mocks["itemRemoved"].assert_not_called()
    mocks["changed"].assert_not_called()

def test_popitem(rdict, mocks):
    key, value = rdict.popitem()
    mocks["itemRemoved"].assert_called_once_with(key, value)
    mocks["changed"].assert_called_once()

def test_setdefault(rdict, mocks):
    # Существующий ключ -> No-op
    assert rdict.setdefault("a", 99) == 1
    mocks["changed"].assert_not_called()

    # Новый ключ -> trigger
    assert rdict.setdefault("c", 3) == 3
    mocks["itemSet"].assert_called_once_with("c", 3, None)
    mocks["changed"].assert_called_once()


# ==================== Массовые операции (Atomicity) ====================

def test_clear(rdict, mocks):
    rdict.clear()
    mocks["cleared"].assert_called_once()
    mocks["changed"].assert_called_once()
    assert len(rdict) == 0

def test_clear_noop(empty_rdict):
    """Очистка пустого словаря не должна спамить сигналами."""
    m_changed = MagicMock()
    empty_rdict.signals.changed.connect(m_changed)
    
    empty_rdict.clear()
    m_changed.assert_not_called()

def test_update_atomicity(rdict, mocks):
    """
    Массовое обновление. itemSet срабатывает для каждого, 
    а changed и updated — строго ОДИН раз.
    """
    rdict.update({"a": 10, "c": 3})  # 'a' изменено, 'c' добавлено, 'b' не тронуто
    
    assert rdict["a"] == 10
    assert rdict["c"] == 3
    
    assert mocks["itemSet"].call_count == 2
    mocks["updated"].assert_called_once_with(["a", "c"])
    mocks["changed"].assert_called_once()  # Атомарность!

def test_update_noop(rdict, mocks):
    """Обновление словаря теми же данными не вызывает сигналы."""
    rdict.update({"a": 1, "b": 2})
    
    mocks["itemSet"].assert_not_called()
    mocks["updated"].assert_not_called()
    mocks["changed"].assert_not_called()

def test_ior_operator(rdict, mocks):
    """Проверка оператора |= (Python 3.9+)"""
    rdict |= {"c": 3}
    assert "c" in rdict
    mocks["itemSet"].assert_called_once_with("c", 3, None)
    mocks["changed"].assert_called_once()


# ==================== Утилиты и Изоляция ====================

def test_copy_isolation(rdict, mocks):
    """Копия словаря должна иметь СВОИ сигналы, никак не связанные с оригиналом."""
    d_copy = rdict.copy()
    assert isinstance(d_copy, ReactiveDict)
    assert d_copy is not rdict
    assert d_copy.signals is not rdict.signals

    # Изменяем копию
    d_copy["x"] = 99
    
    # Сигналы оригинала должны молчать
    mocks["changed"].assert_not_called()

def test_make_reactive():
    """Тест фабрики (idempotency)."""
    d_normal = {"a": 1}
    r_dict1 = make_reactive(d_normal)
    assert isinstance(r_dict1, ReactiveDict)
    
    # Повторный вызов должен вернуть тот же самый объект
    r_dict2 = make_reactive(r_dict1)
    assert r_dict1 is r_dict2
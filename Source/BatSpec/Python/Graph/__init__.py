from typing import Any, Dict, Callable, cast


MISSING = object()


class GraphCache:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._nodes: dict[str, 'GraphNode[Any]'] = {}

    def get_value[T](self, key: str) -> T | Any:
        return self._values.get(key, MISSING)
    
    def set_value[T](self, key: str, val: T) -> None:
        self._values[key] = val

    def get_or_create_node[T](self, key: str, func: Callable[[], T]) -> 'GraphNode[T]':
        if key in self._nodes:
            return cast('GraphNode[T]', self._nodes[key])
        
        # Если узла нет, создаем его
        node = GraphNode(key=key, func=func, cache=self)
        self._nodes[key] = node
        return node
    
class GraphNode[T]:
    def __init__(self, key: str, func: Callable[[], T], cache: GraphCache) -> None:
        self.key = key
        self.func = func
        self.cache = cache

    def __mul__(self, other: 'GraphNode'):
        return get_shared_cache(self, other).get_or_create_node(
            f"mul({self.key}, {other.key})",
            lambda: self.value * other.value
        )
    

    def __add__(self, other: 'GraphNode'):
        return get_shared_cache(self, other).get_or_create_node(
            f"add({self.key}, {other.key})",
            lambda: self.value + other.value
        )
        
    
    @property
    def value(self) -> T:
        val = self.cache.get_value(self.key)
        
        if val is not MISSING:
            return val
        
        val = self.func()
        self.cache.set_value(self.key, val)
        return val


def get_shared_cache(*nodes: GraphNode[Any]) -> GraphCache:
    """Проверяет, что все узлы делят один кэш, и возвращает его."""
    if not nodes:
        raise ValueError("Не передано ни одного узла.")
    
    # Берем кэш первого узла за эталон
    first_cache = nodes[0].cache
    
    # Проверяем остальные
    for node in nodes[1:]:
        if node.cache is not first_cache:
            raise ValueError(f"Конфликт контекстов: узел {node.key} использует другой GraphCache.")
            
    return first_cache


import textwrap
from typing import Any


def delete_init[T: type](cls: T) -> T:
    
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        raise SyntaxError("MRO")
    
    cls.__init__ = __init__
    return cls

def show_mro[T: type](cls: T) -> T:
    print(textwrap.indent("\n".join(map(str, cls.__mro__)), cls.__name__ + ": "))
    return cls



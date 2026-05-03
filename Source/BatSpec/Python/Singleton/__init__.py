from typing import Type


class SingletonCollection:
    def __init__(self):
        self.collection = {}

    def __getitem__[T](self, key: Type[T]) -> T:
        if key not in self.collection:
            raise Exception(f"SingletonCollection does not have {key}")
        return self.collection[key]
    
    def __setitem__[T](self, key: Type[T], value: T):
        self.collection[key] = value




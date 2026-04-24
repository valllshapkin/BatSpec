from contextlib import contextmanager
from typing import Generator, Any
from PySide6.QtWidgets import QWidget, QLayout

@contextmanager
def build_node[T](parent: QWidget | QLayout | Any, item: T) -> Generator[T, None, None]:
    """
    Декларативный билдер UI. 
    Поддерживает любую вложенность:
    - Widget -> Widget (QSplitter, QStackedWidget и т.д.)
    - Widget -> Layout (Устанавливает слой для виджета)
    - Layout -> Widget (Добавляет виджет в слой)
    - Layout -> Layout (Вкладывает слой в слой)
    """
    yield item 
    
    # 1. Если родитель — это слой (QLayout)
    if isinstance(parent, QLayout):
        if isinstance(item, QWidget):
            parent.addWidget(item)
        elif isinstance(item, QLayout):
            parent.addLayout(item)
        else:
            raise TypeError(f"Cannot add {type(item)} to QLayout.")
            
    # 2. Если родитель — это виджет (QWidget)
    elif isinstance(parent, QWidget):
        if isinstance(item, QLayout):
            # Вкладываем Layout внутрь QWidget
            parent.setLayout(item)
        elif isinstance(item, QWidget):
            # Если это контейнер типа QSplitter или QStackedWidget
            if hasattr(parent, 'addWidget'):
                parent.addWidget(item)
            else:
                # Обычный виджет просто делает другой виджет своим ребенком
                item.setParent(parent)
        else:
            raise TypeError(f"Cannot add {type(item)} to QWidget.")
            
    else:
        raise TypeError(f"Invalid parent type: {type(parent)}. Must be QWidget or QLayout.")
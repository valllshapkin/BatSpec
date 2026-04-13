from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QEvent, QObject, QRect, Signal
from typing import Self
from Python import delete_init


# @delete_init
# class SizeTrigger(QWidget):
#     sizeChanged = Signal()

#     def __init_subclass__(cls):

#         old_resizeEvent = cls.resizeEvent

#         def new_resizeEvent(self: Self, event: QEvent):
#             old_resizeEvent(self, event)
#             self.sizeChanged.emit()

#         cls.resizeEvent = new_resizeEvent

#         return super().__init_subclass__()

# @delete_init
# class SizeReciver(QWidget):
#     def sizeSyncing(self):
#         raise NotImplementedError()
    

# class SizeReciverPadding(SizeReciver):
#     size_syncing_padding = 20

#     def sizeSyncing(self):
#         # Получаем локальный прямоугольник
#         local_rect = self.rect()
        
#         # Преобразуем углы в глобальные координаты
#         global_top_left = self.mapToGlobal(local_rect.topLeft())
#         global_bottom_right = self.mapToGlobal(local_rect.bottomRight())
        
#         # Создаем глобальный прямоугольник
#         global_rect = QRect(global_top_left, global_bottom_right)
        
#         # Применяем изменения с учетом отступов
#         self.resize(global_rect.width() - self.size_syncing_padding * 2,
#                     global_rect.height() - self.size_syncing_padding * 2)
#         self.move(global_rect.x() + self.size_syncing_padding,
#                 global_rect.y() + self.size_syncing_padding)
        
#     def setSizeTarget(self, target: SizeTrigger):
#         target.sizeChanged.connect(self.sizeSyncing)
#         return self
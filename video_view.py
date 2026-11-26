from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel

# Минимальный размер выделяемой области, px превью. Короче — считаем промахом
# (случайный клик мышью) и молча игнорируем.
MIN_ROI_PX = 8


class VideoView(QLabel):
    """Превью кадра с выделением области интереса мышью.

    Подкласс QLabel, а не отдельный QWidget: main.py уже красит сюда
    pixmap/текст через setPixmap/setText — это продолжает работать,
    мы только дорисовываем рамку выделения поверх картинки.

    Сигнал отдаёт нормализованные доли отображаемого кадра.
    """

    # x0, y0, x1, y1 в долях кадра (0..1)
    roi_selected = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_enabled = False
        self._anchor = None
        self._rubber = None

    def set_selection_enabled(self, enabled: bool):
        self.selection_enabled = bool(enabled)
        self._anchor = None
        self._rubber = None
        self.update()

    def _image_rect(self):
        pixmap = self.pixmap()
        if pixmap.isNull():
            return QRect()
        return self.style().itemPixmapRect(
            self.contentsRect(), int(self.alignment()), pixmap)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._rubber is None:
            return
        painter = QPainter(self)
        pen = QPen(QColor(255, 214, 0))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.fillRect(self._rubber, QColor(255, 214, 0, 40))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._rubber)

    def mousePressEvent(self, event):
        if not self.selection_enabled or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = event.position().toPoint()
        if not self._image_rect().contains(point):
            return
        self._anchor = point
        self._rubber = QRect(self._anchor, self._anchor)
        self.update()

    def mouseMoveEvent(self, event):
        if self._anchor is None:
            super().mouseMoveEvent(event)
            return
        self._rubber = QRect(self._anchor, event.position().toPoint()).normalized().intersected(
            self._image_rect())
        self.update()

    def mouseReleaseEvent(self, event):
        if self._anchor is None:
            super().mouseReleaseEvent(event)
            return
        rect = QRect(self._anchor, event.position().toPoint()).normalized()
        self._anchor = None
        self._rubber = None
        self.update()
        image_rect = self._image_rect()
        rect = rect.intersected(image_rect)
        if rect.width() < MIN_ROI_PX or rect.height() < MIN_ROI_PX:
            return
        rect.translate(-image_rect.x(), -image_rect.y())
        width, height = image_rect.width(), image_rect.height()
        self.roi_selected.emit(
            rect.left() / width,
            rect.top() / height,
            (rect.right() + 1) / width,
            (rect.bottom() + 1) / height,
        )

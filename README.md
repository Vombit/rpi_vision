# rpi-vision

Система обнаружения изменений в кадре с камеры. Разработка ведётся на Windows
(USB-камера через OpenCV), целевая платформа — Raspberry Pi (CSI-камера через Picamera2).

## Как это работает

```mermaid
flowchart TB
    CAM[Камера] --> CT[CaptureThread<br>capture.py]
    CT -- frame_captured --> P[Processor<br>processing.py, QThread]
    P -- result_ready --> GUI[GUI<br>main.py]
    P --> PRE[preprocessing.py]
    P --> DET[detector.py]
    P --> REF[reference.py]
```

```mermaid
flowchart TB
    F[кадр] --> R[resize 512×512] --> G[gray] --> B[blur] --> N[нормализация яркости]
    N --> D[absdiff с эталоном] --> T[threshold] --> M[morph open]
    M --> C[фильтр мелких областей] --> Q{change_ratio ≥ порога<br>N кадров подряд?}
    Q -- да --> A[ALARM + рамки зон]
    Q -- нет --> OK[NORMAL]
```

- Эталон — медиана `REFERENCE_FRAMES` кадров, задаётся кнопкой **Set reference**.
- **Detect change** — ручная проверка изменений по последним кадрам.
- Все параметры алгоритма — в `config.py`.

## Запуск (Windows / dev)

Требуется Python 3.14 и [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run main.py
```

Камера выбирается в выпадающем списке в окне. Лог — `app.log` (очищается при каждом запуске).

## Запуск на Raspberry Pi

```
sudo apt install python3-picamera2 python3-opencv python3-pyside6.qtwidgets
uv venv --system-site-packages   # чтобы venv видел apt-пакеты
uv sync
uv run main.py
```

- `picamera2` и `opencv` ставим через apt: pip-сборки на ARM тяжёлые и конфликтуют с libcamera.
- В `config.py`: `CAMERA_BACKEND = "picamera2"` (или оставить `auto`) и `LOCK_CAMERA_PARAMS = True` — фиксация экспозиции/баланса белого (на Windows выключена: роняет FPS).
- Калибровка `PIXEL_THRESHOLD` / `MIN_AREA`: включить `DEBUG = True` — снимки этапов пайплайна сохраняются в `debug/`.

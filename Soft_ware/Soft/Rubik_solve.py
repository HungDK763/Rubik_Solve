import sys
import threading
import time
import re

import cv2
import numpy as np
import kociemba

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QComboBox, QGroupBox,
    QTextEdit, QSplitter, QFrame, QMessageBox, QScrollArea,
    QSizePolicy, QSpacerItem, QToolButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPainter, QFont, QFontDatabase, QPalette, QIcon

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FACES = list("URFDLB")

FACE_LABEL = {
    "U": "U · Trắng (Up)",
    "R": "R · Đỏ (Right)",
    "F": "F · Xanh lá (Front)",
    "D": "D · Vàng (Down)",
    "L": "L · Cam (Left)",
    "B": "B · Xanh dương (Back)",
}

FACE_COLOR = {
    "U": "#FFFFFF",
    "R": "#C41E3A",
    "F": "#009B48",
    "D": "#FFD500",
    "L": "#FF5800",
    "B": "#0046AD",
}

FACE_TEXT_COLOR = {
    "U": "#000000",
    "R": "#FFFFFF",
    "F": "#FFFFFF",
    "D": "#000000",
    "L": "#FFFFFF",
    "B": "#FFFFFF",
}

# Màu nền tối
BG_DARK    = "#1A1A2E"
BG_CARD    = "#16213E"
BG_INPUT   = "#0F3460"
ACCENT     = "#E94560"
ACCENT2    = "#F0C040"
TEXT_MAIN  = "#E0E0E0"
TEXT_DIM   = "#888888"
GREEN_OK   = "#2E7D32"
GREEN_LITE = "#4CAF50"
BORDER     = "#333355"

SUPPORTED_CHIPS = ["CH340", "CP2102", "CH341", "CP210x"]
BAUD_RATES = ["9600", "19200", "38400", "57600", "115200", "230400"]


def build_data_frame(solution_str: str) -> str:
    moves = solution_str.strip().split()
    return "start: " + "; ".join(moves) + ":end"


def build_control_frame(command: str) -> str:
    return f"control: {command}"


# ─────────────────────────────────────────────────────────────────────────────
# Camera worker thread
# ─────────────────────────────────────────────────────────────────────────────

# Thu tu lay mau tu bang mau hardware 2x3
# Row0: R(Do)  F(XanhLa)  D(Vang)
# Row1: B(XanhDuong)  L(Cam)  U(Trang)
SAMPLE_GRID_ORDER = ["R", "F", "D", "B", "L", "U"]

MODE_IDLE    = "idle"
MODE_SAMPLE  = "sample"   # Buoc 1: lay mau tu bang mau 2x3
MODE_SCAN    = "scan"     # Buoc 2: nhan dang mat rubik


def _median_bgr(roi):
    b = int(np.median(roi[:, :, 0]))
    g = int(np.median(roi[:, :, 1]))
    r = int(np.median(roi[:, :, 2]))
    return (b, g, r)


def _bgr_to_lab(bgr):
    b, g, r = bgr
    img = np.uint8([[[b, g, r]]])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[0][0].astype(float)
    return lab


def _color_dist_lab(bgr1, bgr2):
    lab1 = _bgr_to_lab(bgr1)
    lab2 = _bgr_to_lab(bgr2)
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def _classify_bgr(bgr, samples):
    best, best_d = "U", float("inf")
    for face, ref_bgr in samples.items():
        d = _color_dist_lab(bgr, ref_bgr)
        if d < best_d:
            best_d, best = d, face
    return best, best_d


def _draw_sample_grid(frame, samples_so_far):
    fh, fw = frame.shape[:2]
    cols, rows = 3, 2
    cell_w = fw // (cols * 2)
    cell_h = fh // (rows * 3)
    pad = 8
    ox = fw - cols * (cell_w + pad) - pad
    oy = pad

    for idx, face in enumerate(SAMPLE_GRID_ORDER):
        r = idx // cols
        c = idx % cols
        x1 = ox + c * (cell_w + pad)
        y1 = oy + r * (cell_h + pad)
        x2 = x1 + cell_w
        y2 = y1 + cell_h

        if face in samples_so_far:
            b, g, rv = samples_so_far[face]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (int(b), int(g), int(rv)), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, face, (x1 + 4, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(frame, face + "?", (x1 + 4, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    cv2.putText(frame, "Mau da lay mau", (ox, oy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def _draw_sample_box(frame):
    fh, fw = frame.shape[:2]
    cols, rows = 3, 2
    cell_sz = min(fw // 4, fh // 3)
    total_w = cols * cell_sz
    total_h = rows * cell_sz
    ox = (fw - total_w) // 2
    oy = (fh - total_h) // 2

    FACE_DOT = {
        "R": (0,0,220), "F": (0,155,0), "D": (0,213,255),
        "B": (173,70,0), "L": (0,100,255), "U": (220,220,220),
    }

    # Bước 1: đọc màu từ frame gốc TRƯỚC khi vẽ bất cứ thứ gì
    sample_rois = {}
    pad = cell_sz // 5
    for idx, face in enumerate(SAMPLE_GRID_ORDER):
        r = idx // cols
        c = idx % cols
        x1 = ox + c * cell_sz
        y1 = oy + r * cell_sz
        x2 = x1 + cell_sz
        y2 = y1 + cell_sz
        ix1, iy1 = x1 + pad, y1 + pad
        ix2, iy2 = x2 - pad, y2 - pad
        roi = frame[iy1:iy2, ix1:ix2]
        if roi.size > 0:
            sample_rois[face] = _median_bgr(roi)

    # Bước 2: vẽ overlay tối ngoài khung
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Bước 3: vẽ lưới và nhãn lên frame
    for idx, face in enumerate(SAMPLE_GRID_ORDER):
        r = idx // cols
        c = idx % cols
        x1 = ox + c * cell_sz
        y1 = oy + r * cell_sz
        x2 = x1 + cell_sz
        y2 = y1 + cell_sz
        ix1, iy1 = x1 + pad, y1 + pad
        ix2, iy2 = x2 - pad, y2 - pad

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 255), 1)

        # Hiển thị màu đã đọc được bên trong vùng sampling
        if face in sample_rois:
            b_v, g_v, r_v = sample_rois[face]
            cv2.rectangle(frame, (ix1+1, iy1+1), (ix2-1, iy2-1),
                          (b_v, g_v, r_v), -1)
            cv2.putText(frame, f"B{b_v} G{g_v} R{r_v}",
                        (ix1+2, iy2-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (200,200,200), 1)

        dot = FACE_DOT.get(face, (180,180,180))
        cv2.circle(frame, (x1+14, y1+14), 8, dot, -1)
        cv2.circle(frame, (x1+14, y1+14), 8, (255,255,255), 1)
        cv2.putText(frame, face, (x1+27, y1+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    cv2.putText(frame, "BUOC 1: Dat bang mau vao khung  |  [Space] = Lay mau",
                (ox, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 220), 1)

    return sample_rois


def _draw_scan_grid(frame, samples, center_face):
    fh, fw = frame.shape[:2]
    cell_sz = fh // 3
    total   = cell_sz * 3
    ox      = (fw - total) // 2
    oy      = 0
    pad     = cell_sz // 5

    # Buoc 1: doc tat ca ROI tu frame goc TRUOC khi ve bat cu thu gi
    raw_bgr    = {}
    raw_labels = {}
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            if idx == 4:
                continue
            x1  = ox + col * cell_sz
            y1  = oy + row * cell_sz
            x2  = x1 + cell_sz
            y2  = y1 + cell_sz
            roi = frame[y1+pad: y2-pad, x1+pad: x2-pad]
            if roi.size > 0:
                bgr = _median_bgr(roi)
                raw_bgr[idx]    = bgr
                face, dist      = _classify_bgr(bgr, samples)
                raw_labels[idx] = (face, dist)

    # Buoc 2: ve overlay toi 2 ben
    if ox > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (ox, fh), (0, 0, 0), -1)
        cv2.rectangle(overlay, (ox + total, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Buoc 3: ve luoi va ket qua
    labels = []
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            x1  = ox + col * cell_sz
            y1  = oy + row * cell_sz
            x2  = x1 + cell_sz
            y2  = y1 + cell_sz
            cx  = (x1 + x2) // 2
            cy  = (y1 + y2) // 2

            cv2.rectangle(frame, (x1+1, y1+1), (x2-1, y2-1), (0, 255, 0), 2)

            if idx == 4:
                labels.append(center_face)
                dot = _face_dot_bgr(center_face)
                cv2.circle(frame, (cx, cy), 24, dot, -1)
                cv2.circle(frame, (cx, cy), 24, (255, 255, 255), 2)
                cv2.putText(frame, center_face, (cx-9, cy+7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                continue

            if idx not in raw_labels:
                labels.append(center_face)
                continue

            face, dist = raw_labels[idx]
            b_v, g_v, r_v = raw_bgr[idx]
            labels.append(face)

            cv2.rectangle(frame, (x1+pad, y1+pad), (x2-pad, y2-pad),
                          (b_v, g_v, r_v), -1)
            cv2.rectangle(frame, (x1+pad, y1+pad), (x2-pad, y2-pad),
                          (200, 200, 200), 1)

            dot = _face_dot_bgr(face)
            cv2.circle(frame, (cx, cy), 20, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), 20, dot, -1)
            cv2.circle(frame, (cx, cy), 20, (255, 255, 255), 1)
            cv2.putText(frame, face, (cx-8, cy+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"{int(dist)}", (x1+3, y2-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    face_name = FACE_LABEL.get(center_face, center_face)
    cv2.rectangle(frame, (0, fh-26), (fw, fh), (0, 0, 0), -1)
    cv2.putText(frame, f"BUOC 2: Mat [{face_name}]  |  [Space] = Chup mat",
                (8, fh-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return labels


def _face_dot_bgr(face):
    table = {
        "U": (220, 220, 220),
        "R": (0,   0,   220),
        "F": (0,   155, 0  ),
        "D": (0,   213, 255),
        "L": (0,   100, 255),
        "B": (173, 70,  0  ),
    }
    return table.get(face, (180, 180, 180))


class CameraWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray)
    sample_ready   = pyqtSignal(dict)
    scan_done      = pyqtSignal(list)
    stopped        = pyqtSignal()
    cam_error      = pyqtSignal(str)

    def __init__(self, cam_index=0):
        super().__init__()
        self._running     = False
        self._cam_index   = cam_index
        self._mode        = MODE_IDLE
        self._samples     = {}
        self._center_face = "U"
        self._lock        = threading.Lock()
        self._do_capture  = False

    def set_mode_idle(self):
        with self._lock:
            self._mode = MODE_IDLE

    def set_mode_sample(self):
        with self._lock:
            self._mode = MODE_SAMPLE

    def set_mode_scan(self, samples, center_face):
        with self._lock:
            self._mode        = MODE_SCAN
            self._samples     = dict(samples)
            self._center_face = center_face

    def trigger_capture(self):
        with self._lock:
            self._do_capture = True

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._cam_index, backend)
        if not cap.isOpened():
            self.cam_error.emit(f"Khong mo duoc camera #{self._cam_index}")
            self.stopped.emit()
            return

        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            with self._lock:
                mode        = self._mode
                samples     = dict(self._samples)
                center_face = self._center_face
                do_cap      = self._do_capture
                self._do_capture = False

            if mode == MODE_SAMPLE:
                current_rois = _draw_sample_box(frame)
                if do_cap and current_rois:
                    with self._lock:
                        self._samples = dict(current_rois)
                    self.sample_ready.emit(dict(current_rois))

            elif mode == MODE_SCAN:
                labels = _draw_scan_grid(frame, samples, center_face)
                if do_cap:
                    self.scan_done.emit(labels)

            self.frame_ready.emit(frame.copy())
            time.sleep(0.033)

        cap.release()
        self.stopped.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Color Picker Button  — ô màu có thể click để cycle qua 6 màu
# ─────────────────────────────────────────────────────────────────────────────
CELL_SIZE = 40   # px — kích thước ô vuông
CELL_GAP  = 3    # px — khoảng cách đều giữa các ô


class ColorCell(QPushButton):
    color_changed = pyqtSignal(str, int, str)  # face, index, new_color

    def __init__(self, face: str, index: int, color: str, is_center=False):
        super().__init__()
        self.face      = face
        self.index     = index
        self.is_center = is_center
        self._color    = color
        # Kích thước cố định tuyệt đối — không resize
        self.setFixedSize(CELL_SIZE, CELL_SIZE)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._apply_style()
        if not is_center:
            self.clicked.connect(self._cycle)

    def _apply_style(self):
        c  = FACE_COLOR[self._color]
        tc = FACE_TEXT_COLOR[self._color]
        if self.is_center:
            # Ô trung tâm: viền dày + chữ tên mặt
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {c};
                    color: {tc};
                    border: 3px solid rgba(255,255,255,0.6);
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 11px;
                }}
            """)
            self.setText(self.face)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {c};
                    color: {tc};
                    border: 1px solid rgba(0,0,0,0.35);
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 9px;
                }}
                QPushButton:hover {{
                    border: 2px solid #FFFFFF;
                }}
                QPushButton:pressed {{
                    border: 2px solid {ACCENT2};
                }}
            """)
            self.setText("")
        face_name = FACE_LABEL.get(self._color, self._color)
        tip = f"[Trung tâm — cố định]" if self.is_center else f"Màu: {face_name}\nClick để đổi màu"
        self.setToolTip(tip)

    def set_color(self, color: str):
        self._color = color
        self._apply_style()

    def get_color(self) -> str:
        return self._color

    def _cycle(self):
        idx    = FACES.index(self._color)
        next_c = FACES[(idx + 1) % len(FACES)]
        self.set_color(next_c)
        self.color_changed.emit(self.face, self.index, next_c)

    def set_color_direct(self, color: str):
        self._color = color
        self._apply_style()


# ─────────────────────────────────────────────────────────────────────────────
# Face Widget  — lưới 3×3
# ─────────────────────────────────────────────────────────────────────────────
class FaceWidget(QWidget):
    """
    Luoi 3x3 cho mot mat rubik.
    Kich thuoc co dinh = 3*CELL_SIZE + 2*CELL_GAP + 2*PAD (padding vien)
    """
    FACE_PAD = 5

    def __init__(self, face: str, parent=None):
        super().__init__(parent)
        self.face  = face
        self.cells = []
        self._captured = False

        inner = 3 * CELL_SIZE + 2 * CELL_GAP
        total = inner + 2 * self.FACE_PAD + 2
        self.setFixedSize(total, total)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.FACE_PAD, self.FACE_PAD,
                                  self.FACE_PAD, self.FACE_PAD)
        layout.setSpacing(0)

        grid = QGridLayout()
        grid.setSpacing(CELL_GAP)
        grid.setContentsMargins(0, 0, 0, 0)

        for row in range(3):
            grid.setRowMinimumHeight(row, CELL_SIZE)
            grid.setRowStretch(row, 0)
        for col in range(3):
            grid.setColumnMinimumWidth(col, CELL_SIZE)
            grid.setColumnStretch(col, 0)

        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                is_center = (idx == 4)
                cell = ColorCell(face, idx, face if is_center else "U", is_center)
                grid.addWidget(cell, row, col)
                self.cells.append(cell)

        layout.addLayout(grid)
        self.setLayout(layout)
        self._update_border()

    def set_face(self, labels: list):
        for i, cell in enumerate(self.cells):
            if i == 4:
                continue
            cell.set_color_direct(labels[i])
        self._captured = True
        self._update_border()

    def get_labels(self) -> list:
        return [c.get_color() for c in self.cells]

    def reset(self):
        for i, cell in enumerate(self.cells):
            if i == 4:
                continue
            cell.set_color_direct("U")
        self._captured = False
        self._update_border()

    def mark_captured(self, ok: bool):
        self._captured = ok
        self._update_border()

    def _update_border(self):
        bcolor = GREEN_LITE if self._captured else BORDER
        self.setStyleSheet(f"background: {BG_CARD}; border: 2px solid {bcolor}; border-radius: 7px;")


KEY_TO_FACE = {
    Qt.Key_W: "U",
    Qt.Key_R: "R",
    Qt.Key_G: "F",
    Qt.Key_Y: "D",
    Qt.Key_O: "L",
    Qt.Key_B: "B",
}


class QuickInputDialog(QWidget):
    confirmed = pyqtSignal(str, list)

    SHORTCUT_LABELS = [
        ("W", "U", "#FFFFFF", "#000000"),
        ("R", "R", "#C41E3A", "#FFFFFF"),
        ("G", "F", "#009B48", "#FFFFFF"),
        ("Y", "D", "#FFD500", "#000000"),
        ("O", "L", "#FF5800", "#FFFFFF"),
        ("B", "B", "#0046AD", "#FFFFFF"),
    ]

    def __init__(self, face, current_labels, parent=None):
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.face = face
        self.labels = list(current_labels)
        self.cursor = 0
        self._advance_past_center()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(f"background: {BG_CARD}; border: 2px solid {ACCENT2}; border-radius: 12px;")
        self._build()
        self.setFixedSize(self.sizeHint())
        self._refresh_grid()
        self._update_hint()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        title_row = QHBoxLayout()
        face_dot = QLabel(f"  {self.face}  ")
        face_dot.setStyleSheet(f"background:{FACE_COLOR[self.face]};color:{FACE_TEXT_COLOR[self.face]};border-radius:4px;font-weight:bold;font-size:13px;padding:3px 10px;")
        title_lbl = QLabel(f"Nhap mau nhanh — {FACE_LABEL[self.face]}")
        title_lbl.setStyleSheet(f"color:{ACCENT2};font-weight:bold;font-size:13px;")
        close_btn = QPushButton("x")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(f"background:#333;color:#aaa;border-radius:4px;font-size:11px;border:none;")
        close_btn.clicked.connect(self.close)
        title_row.addWidget(face_dot)
        title_row.addSpacing(8)
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        title_row.addWidget(close_btn)
        root.addLayout(title_row)

        self.cell_btns = []
        grid = QGridLayout()
        grid.setSpacing(4)
        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                btn = QPushButton()
                btn.setFixedSize(56, 56)
                btn.setFocusPolicy(Qt.NoFocus)
                if idx != 4:
                    btn.clicked.connect(lambda _, i=idx: self._jump_to(i))
                grid.addWidget(btn, row, col)
                self.cell_btns.append(btn)
        root.addLayout(grid)

        legend_row = QHBoxLayout()
        legend_row.setSpacing(4)
        for key, face, bg, fg in self.SHORTCUT_LABELS:
            lbl = QLabel(f"<b>{key}</b>={face}")
            lbl.setStyleSheet(f"background:{bg};color:{fg};border-radius:4px;font-size:11px;padding:3px 8px;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedHeight(26)
            legend_row.addWidget(lbl)
        root.addLayout(legend_row)

        self.hint_lbl = QLabel("")
        self.hint_lbl.setAlignment(Qt.AlignCenter)
        self.hint_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        root.addWidget(self.hint_lbl)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        for text, tip, slot in [
            ("Back [BkSp]", "", self._go_back),
            ("Skip [Space]", "", self._skip),
            ("Done [Enter]", "", self._confirm),
        ]:
            btn = QPushButton(text)
            btn.setFixedHeight(28)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(slot)
            btn.setStyleSheet(f"background:{BG_INPUT};color:{TEXT_MAIN};border:1px solid {BORDER};border-radius:5px;font-size:10px;")
            nav_row.addWidget(btn)
        root.addLayout(nav_row)

    def _advance_past_center(self):
        if self.cursor == 4:
            self.cursor = 5

    def _jump_to(self, idx):
        if idx == 4:
            return
        self.cursor = idx
        self._refresh_grid()
        self._update_hint()

    def _refresh_grid(self):
        for idx, btn in enumerate(self.cell_btns):
            if idx == 4:
                c = FACE_COLOR[self.face]
                tc = FACE_TEXT_COLOR[self.face]
                btn.setStyleSheet(f"QPushButton{{background:{c};color:{tc};border:3px solid rgba(255,255,255,0.6);border-radius:28px;font-weight:bold;font-size:13px;}}")
                btn.setText(self.face)
                continue
            color = self.labels[idx]
            bg  = FACE_COLOR[color]
            fg  = FACE_TEXT_COLOR[color]
            if idx == self.cursor:
                border = f"3px solid {ACCENT2}"
                txt = ">"
            elif idx < self.cursor:
                border = "2px solid #4CAF50"
                txt = ""
            else:
                border = "1px solid rgba(255,255,255,0.15)"
                txt = ""
            btn.setStyleSheet(f"QPushButton{{background:{bg};color:{fg};border:{border};border-radius:6px;font-size:16px;font-weight:bold;}}QPushButton:hover{{border:2px solid #fff;}}")
            btn.setText(txt)

    def _update_hint(self):
        if self.cursor >= 9:
            self.hint_lbl.setText("Xong! Enter de xac nhan.")
        else:
            pos = self.cursor + 1 if self.cursor < 4 else self.cursor
            self.hint_lbl.setText(f"O {pos}/8 — bam phim mau (W R G Y O B)  |  Space=bo qua  |  Esc=huy")

    def keyPressEvent(self, event):
        key = event.key()
        if key in KEY_TO_FACE and self.cursor < 9:
            self.labels[self.cursor] = KEY_TO_FACE[key]
            self.cursor += 1
            self._advance_past_center()
            self._refresh_grid()
            self._update_hint()
            if self.cursor >= 9:
                self._confirm()
        elif key in (Qt.Key_Space, Qt.Key_Tab):
            self._skip()
        elif key == Qt.Key_Backspace:
            self._go_back()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm()
        elif key == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def _skip(self):
        if self.cursor < 9:
            self.cursor += 1
            self._advance_past_center()
            self._refresh_grid()
            self._update_hint()

    def _go_back(self):
        self.cursor = max(0, self.cursor - 1)
        if self.cursor == 4:
            self.cursor = 3
        self._refresh_grid()
        self._update_hint()

    def _confirm(self):
        self.labels[4] = self.face
        self.confirmed.emit(self.face, self.labels)
        self.close()

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()


# ─────────────────────────────────────────────────────────────────────────────
# Cube Net Widget  — bố cục chữ thập
# ─────────────────────────────────────────────────────────────────────────────
class FaceLabel(QLabel):
    """Label tên mặt hiển thị bên dưới FaceWidget"""
    def __init__(self, face: str):
        super().__init__(FACE_LABEL[face])
        self.setAlignment(Qt.AlignCenter)
        c = FACE_COLOR[face]
        self.setStyleSheet(f"""
            color: {TEXT_MAIN};
            background: {c}22;
            border: 1px solid {c}55;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
            padding: 2px 4px;
        """)
        fw = 3 * CELL_SIZE + 2 * CELL_GAP + 2 * FaceWidget.FACE_PAD + 2
        self.setFixedWidth(fw)


class CubeNetWidget(QWidget):
    quick_input_clicked = pyqtSignal(str)

    """
    Bố cục trải phẳng khối rubik chuẩn dạng chữ thập:

              [ U ]
        [ L ] [ F ] [ R ] [ B ]
              [ D ]

    Dùng QGridLayout 3 hàng × 4 cột.
    Cột 0=L, 1=F, 2=R, 3=B
    Hàng 0=U(col1), 1=LFRB, 2=D(col1)
    """

    GAP_FACE = 8  # khoảng cách giữa các mặt (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.faces: dict[str, FaceWidget] = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(0)

        # Title
        title = QLabel("✏  Cube Net — Click vào ô để đổi màu thủ công")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {ACCENT2};
            font-size: 12px;
            font-weight: bold;
            padding: 0 0 12px 0;
        """)
        root.addWidget(title)

        # Hint legend — 6 màu tham khảo
        legend = QHBoxLayout()
        legend.setSpacing(6)
        legend.addStretch()
        for f in FACES:
            dot = QLabel(f"  {f}  ")
            dot.setStyleSheet(f"""
                background: {FACE_COLOR[f]};
                color: {FACE_TEXT_COLOR[f]};
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 6px;
            """)
            dot.setToolTip(FACE_LABEL[f])
            legend.addWidget(dot)
        legend.addStretch()
        root.addLayout(legend)

        spacer_h = QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)
        root.addItem(spacer_h)

        # ── Grid layout chính ──────────────────────────────────────────────
        # Layout: 3 hàng × 4 cột
        #   hàng 0: _ U _ _
        #   hàng 1: L F R B
        #   hàng 2: _ D _ _
        grid = QGridLayout()
        grid.setSpacing(self.GAP_FACE)
        grid.setContentsMargins(0, 0, 0, 0)

        face_size = 3 * CELL_SIZE + 2 * CELL_GAP + 2 * FaceWidget.FACE_PAD + 2

        for r in range(3):
            grid.setRowStretch(r, 0)
        for c in range(4):
            grid.setColumnStretch(c, 0)
            grid.setColumnMinimumWidth(c, face_size)

        def _add_face(face, row, col):
            fw = FaceWidget(face)
            self.faces[face] = fw
            cell_w = QWidget()
            cell_w.setFixedSize(face_size, face_size + 44)
            vl = QVBoxLayout(cell_w)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(2)
            vl.addWidget(fw)
            lbl = FaceLabel(face)
            vl.addWidget(lbl)
            c_bg = FACE_COLOR[face]
            btn_quick = QPushButton("\u2328  Nhap nhanh")
            btn_quick.setFixedHeight(20)
            btn_quick.setFocusPolicy(Qt.NoFocus)
            btn_quick.setStyleSheet(f"QPushButton{{background:{c_bg}33;color:{TEXT_MAIN};border:1px solid {c_bg}88;border-radius:4px;font-size:9px;font-weight:bold;}}QPushButton:hover{{background:{c_bg}66;}}")
            btn_quick.clicked.connect(lambda _, f=face: self.quick_input_clicked.emit(f))
            vl.addWidget(btn_quick)
            grid.addWidget(cell_w, row, col, Qt.AlignCenter)

        # Hàng 0 — U ở cột 1, placeholder ở các cột còn lại
        # (placeholder invisible để giữ spacing đúng)
        for c in [0, 2, 3]:
            ph = QWidget()
            ph.setFixedSize(face_size, face_size + 44)
            grid.addWidget(ph, 0, c)
        _add_face("U", 0, 1)

        # Hàng 1 — L F R B
        for i, f in enumerate(["L", "F", "R", "B"]):
            _add_face(f, 1, i)

        # Hàng 2 — D ở cột 1
        for c in [0, 2, 3]:
            ph = QWidget()
            ph.setFixedSize(face_size, face_size + 44)
            grid.addWidget(ph, 2, c)
        _add_face("D", 2, 1)

        root.addLayout(grid)
        root.addStretch()

    def set_face(self, face: str, labels: list):
        self.faces[face].set_face(labels)
        self.faces[face].mark_captured(True)

    def get_all_labels(self) -> dict:
        return {f: fw.get_labels() for f, fw in self.faces.items()}

    def reset(self):
        for fw in self.faces.values():
            fw.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Serial Panel
# ─────────────────────────────────────────────────────────────────────────────
class SerialPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("🔌 Kết nối Serial", parent)
        self._port   = None
        self._serial = None
        self._build()
        self.setStyleSheet(f"""
            QGroupBox {{
                color: {TEXT_MAIN};
                border: 1px solid {BORDER};
                border-radius: 8px;
                margin-top: 8px;
                font-weight: bold;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        """)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        # Port + Baud
        row1 = QHBoxLayout()
        self.cb_port = QComboBox()
        self.cb_port.setMinimumWidth(140)
        self.cb_port.setStyleSheet(self._combo_style())

        self.cb_baud = QComboBox()
        for b in BAUD_RATES:
            self.cb_baud.addItem(b)
        self.cb_baud.setCurrentText("115200")
        self.cb_baud.setStyleSheet(self._combo_style())

        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedSize(30, 30)
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_refresh.setStyleSheet(self._btn_style("#37474F"))

        row1.addWidget(QLabel("Cổng:", styleSheet=f"color:{TEXT_DIM}"))
        row1.addWidget(self.cb_port)
        row1.addWidget(self.btn_refresh)
        row1.addWidget(QLabel("Baud:", styleSheet=f"color:{TEXT_DIM}"))
        row1.addWidget(self.cb_baud)
        lay.addLayout(row1)

        # Connect button + status
        row2 = QHBoxLayout()
        self.btn_connect = QPushButton("Kết nối")
        self.btn_connect.setFixedHeight(32)
        self.btn_connect.clicked.connect(self.toggle_connect)
        self.btn_connect.setStyleSheet(self._btn_style("#0046AD"))

        self.lbl_status = QLabel("⚫ Chưa kết nối")
        self.lbl_status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        row2.addWidget(self.btn_connect)
        row2.addWidget(self.lbl_status)
        row2.addStretch()
        lay.addLayout(row2)

        self.refresh_ports()

    def refresh_ports(self):
        self.cb_port.clear()
        if not SERIAL_AVAILABLE:
            self.cb_port.addItem("pyserial không có")
            return
        ports = serial.tools.list_ports.comports()
        found = []
        for p in ports:
            desc = (p.description or "") + (p.manufacturer or "")
            if any(chip.lower() in desc.lower() for chip in SUPPORTED_CHIPS):
                found.append(p.device)
        if not found:
            # fallback: thêm tất cả
            for p in ports:
                found.append(p.device)
        for dev in found:
            self.cb_port.addItem(dev)
        if not found:
            self.cb_port.addItem("Không tìm thấy")

    def toggle_connect(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None
            self.lbl_status.setText("⚫ Đã ngắt kết nối")
            self.lbl_status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
            self.btn_connect.setText("Kết nối")
            self.btn_connect.setStyleSheet(self._btn_style("#0046AD"))
        else:
            port = self.cb_port.currentText()
            baud = int(self.cb_baud.currentText())
            if not SERIAL_AVAILABLE:
                QMessageBox.warning(self, "Lỗi", "Cài pyserial: pip install pyserial")
                return
            try:
                self._serial = serial.Serial(port, baud, timeout=1)
                self.lbl_status.setText(f"🟢 {port} @ {baud}")
                self.lbl_status.setStyleSheet(f"color: {GREEN_LITE}; font-size: 11px;")
                self.btn_connect.setText("Ngắt")
                self.btn_connect.setStyleSheet(self._btn_style("#B71C1C"))
            except Exception as e:
                QMessageBox.critical(self, "Lỗi kết nối", str(e))

    def send(self, data: str) -> bool:
        if self._serial and self._serial.is_open:
            try:
                self._serial.write((data + "\n").encode())
                return True
            except Exception as e:
                QMessageBox.critical(self, "Lỗi gửi", str(e))
        return False

    def is_connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    @staticmethod
    def _combo_style():
        return f"""
            QComboBox {{
                background: {BG_INPUT};
                color: {TEXT_MAIN};
                border: 1px solid {BORDER};
                border-radius: 5px;
                padding: 3px 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {BG_INPUT};
                color: {TEXT_MAIN};
            }}
        """

    @staticmethod
    def _btn_style(bg):
        return f"""
            QPushButton {{
                background: {bg};
                color: white;
                border-radius: 5px;
                font-weight: bold;
                padding: 4px 12px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
        """


# ─────────────────────────────────────────────────────────────────────────────
# Left Panel — Camera + Calibration + Capture
# ─────────────────────────────────────────────────────────────────────────────
class LeftPanel(QWidget):
    face_captured = pyqtSignal(str, list)
    log_msg       = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples     = {}
        self._cam_worker  = None
        self._cam_running = False
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        grp_cam = QGroupBox("📷 Camera")
        grp_cam.setStyleSheet(self._grp_style())
        cam_lay = QVBoxLayout(grp_cam)

        self.lbl_cam_status = QLabel("Camera chua mo")
        self.lbl_cam_status.setAlignment(Qt.AlignCenter)
        self.lbl_cam_status.setStyleSheet(f"background:#000;color:{TEXT_DIM};border-radius:6px;padding:30px 20px;font-size:13px;")
        self.lbl_cam_status.setMinimumHeight(90)
        cam_lay.addWidget(self.lbl_cam_status)

        sel_row = QHBoxLayout()
        sel_row.setSpacing(6)

        cam_lbl = QLabel("Camera:")
        cam_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        sel_row.addWidget(cam_lbl)

        self.cb_cam_index = QComboBox()
        self.cb_cam_index.setStyleSheet(f"""
            QComboBox {{
                background:{BG_INPUT};color:{TEXT_MAIN};
                border:1px solid {BORDER};border-radius:5px;
                padding:3px 8px;font-size:11px;
            }}
            QComboBox QAbstractItemView {{
                background:{BG_INPUT};color:{TEXT_MAIN};
            }}
        """)
        for i in range(4):
            self.cb_cam_index.addItem(f"#{i}  (Camera {i})")
        self.cb_cam_index.setCurrentIndex(0)
        sel_row.addWidget(self.cb_cam_index, 1)

        self.btn_scan_cams = QPushButton("🔍")
        self.btn_scan_cams.setFixedSize(28, 28)
        self.btn_scan_cams.setToolTip("Quet tim tat ca camera kha dung")
        self.btn_scan_cams.clicked.connect(self._scan_cameras)
        self.btn_scan_cams.setStyleSheet(f"background:{BG_INPUT};color:{TEXT_MAIN};border:1px solid {BORDER};border-radius:5px;font-size:12px;")
        sel_row.addWidget(self.btn_scan_cams)

        cam_lay.addLayout(sel_row)

        btn_row = QHBoxLayout()
        self.btn_open_cam = QPushButton("▶ Mo camera")
        self.btn_open_cam.clicked.connect(self.toggle_camera)
        self.btn_open_cam.setStyleSheet(self._btn("#00897B"))
        btn_row.addWidget(self.btn_open_cam)
        cam_lay.addLayout(btn_row)
        lay.addWidget(grp_cam)

        grp_sample = QGroupBox("🎨 Buoc 1 — Lay mau bang mau 2x3")
        grp_sample.setStyleSheet(self._grp_style())
        sl = QVBoxLayout(grp_sample)
        sl.setSpacing(6)

        hint1 = QLabel(
            "① Mo camera\n"
            "② Dat bang mau 6 o (hardware) vao khung tren camera\n"
            "   Thu tu: Do | Xanh la | Vang\n"
            "           Xanh duong | Cam | Trang\n"
            "③ Bam  Chup mau  khi anh on dinh"
        )
        hint1.setWordWrap(True)
        hint1.setStyleSheet(f"background:{BG_INPUT};color:{TEXT_MAIN};border:1px solid {BORDER};border-radius:6px;font-size:11px;padding:8px 10px;")
        sl.addWidget(hint1)

        self.sample_preview = QWidget()
        self.sample_preview.setFixedHeight(38)
        sp_lay = QHBoxLayout(self.sample_preview)
        sp_lay.setContentsMargins(0, 0, 0, 0)
        sp_lay.setSpacing(3)
        self.sample_dots = {}
        for face in SAMPLE_GRID_ORDER:
            dot = QLabel(face)
            dot.setFixedSize(34, 34)
            dot.setAlignment(Qt.AlignCenter)
            dot.setStyleSheet(f"background:#222;color:#555;border-radius:4px;font-weight:bold;font-size:11px;border:1px solid #333;")
            self.sample_dots[face] = dot
            sp_lay.addWidget(dot)
        sp_lay.addStretch()
        sl.addWidget(self.sample_preview)

        self.btn_capture_sample = QPushButton("📷 Chup mau (Space)")
        self.btn_capture_sample.setFixedHeight(32)
        self.btn_capture_sample.setEnabled(False)
        self.btn_capture_sample.clicked.connect(self._do_sample_capture)
        self.btn_capture_sample.setStyleSheet(self._btn("#7B1FA2"))
        sl.addWidget(self.btn_capture_sample)

        self.lbl_sample_progress = QLabel("Chua co mau mau nao")
        self.lbl_sample_progress.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        sl.addWidget(self.lbl_sample_progress)

        lay.addWidget(grp_sample)

        grp_scan = QGroupBox("📸 Buoc 2 — Scan tung mat")
        grp_scan.setStyleSheet(self._grp_style())
        scl = QVBoxLayout(grp_scan)
        scl.setSpacing(6)

        hint2 = QLabel(
            "① Dat mat rubik truoc camera (grid 3x3 hien thi tren camera)\n"
            "② Chon mat trung tam dung mau o duoi\n"
            "③ Bam  Chup mat  hoac phim Space"
        )
        hint2.setWordWrap(True)
        hint2.setStyleSheet(f"background:{BG_INPUT};color:{TEXT_MAIN};border:1px solid {BORDER};border-radius:6px;font-size:11px;padding:8px 10px;")
        scl.addWidget(hint2)

        center_lbl = QLabel("Mat trung tam:")
        center_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        scl.addWidget(center_lbl)

        center_row = QHBoxLayout()
        center_row.setSpacing(4)
        self.center_btns = {}
        self.center_btn_group = QButtonGroup(self)
        self.center_btn_group.setExclusive(True)
        for face in FACES:
            btn = QPushButton(face)
            btn.setFixedSize(36, 30)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {BG_INPUT};
                    color: {TEXT_MAIN};
                    border: 1px solid {BORDER};
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 11px;
                }}
                QPushButton:checked {{
                    background: {BG_INPUT};
                    color: #FFFFFF;
                    border: 3px solid #FFFFFF;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    border: 1px solid {TEXT_MAIN};
                }}
            """)
            btn.clicked.connect(lambda _, f=face: self._set_center_face(f))
            self.center_btns[face] = btn
            self.center_btn_group.addButton(btn)
            center_row.addWidget(btn)
        center_row.addStretch()
        scl.addLayout(center_row)

        self.btn_capture_face = QPushButton("📸 Chup mat (Space)")
        self.btn_capture_face.setFixedHeight(32)
        self.btn_capture_face.setEnabled(False)
        self.btn_capture_face.clicked.connect(self._do_face_capture)
        self.btn_capture_face.setStyleSheet(self._btn("#1565C0"))
        scl.addWidget(self.btn_capture_face)

        self.lbl_scan_status = QLabel("Can lay mau truoc (Buoc 1)")
        self.lbl_scan_status.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        scl.addWidget(self.lbl_scan_status)

        lay.addWidget(grp_scan)
        lay.addStretch()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self._cam_running:
                if self._samples and self.btn_capture_face.isEnabled():
                    self._do_face_capture()
                elif self.btn_capture_sample.isEnabled():
                    self._do_sample_capture()
        super().keyPressEvent(event)

    def toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _scan_cameras(self):
        self.btn_scan_cams.setEnabled(False)
        self.btn_scan_cams.setText("...")
        self.lbl_cam_status.setText("Dang quet camera...")

        class ScanThread(QThread):
            done = pyqtSignal(list)
            def run(self):
                found = []
                for i in range(6):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
                    if cap.isOpened():
                        ok, _ = cap.read()
                        if ok:
                            found.append(i)
                        cap.release()
                self.done.emit(found)

        self._scan_thread = ScanThread()
        self._scan_thread.done.connect(self._on_scan_cameras_done)
        self._scan_thread.start()

    def _on_scan_cameras_done(self, found):
        self.cb_cam_index.clear()
        if found:
            for i in found:
                self.cb_cam_index.addItem(f"#{i}  (Camera {i})")
            self.lbl_cam_status.setText(f"Tim thay {len(found)} camera: {found}")
        else:
            for i in range(4):
                self.cb_cam_index.addItem(f"#{i}  (Camera {i})")
            self.lbl_cam_status.setText("Khong tim thay camera. Thu thu cong.")
        self.btn_scan_cams.setText("🔍")
        self.btn_scan_cams.setEnabled(True)

    def _get_cam_index(self):
        text = self.cb_cam_index.currentText()
        try:
            return int(text.split()[0].replace("#", ""))
        except Exception:
            return 0

    def _start_camera(self):
        cam_idx = self._get_cam_index()
        self._cam_worker = CameraWorker(cam_idx)
        self._cam_worker.frame_ready.connect(self._on_frame)
        self._cam_worker.sample_ready.connect(self._on_sample_ready)
        self._cam_worker.scan_done.connect(self._on_scan_done)
        self._cam_worker.stopped.connect(self._on_cam_stopped)
        self._cam_worker.cam_error.connect(self._on_cam_error)
        self._cam_worker.start()
        self._cam_running = True
        self.btn_open_cam.setText("⏹ Dong camera")
        self.btn_open_cam.setStyleSheet(self._btn("#B71C1C"))
        self.cb_cam_index.setEnabled(False)
        self.btn_scan_cams.setEnabled(False)
        self.lbl_cam_status.setText(f"Camera #{cam_idx} dang chay")
        self.btn_capture_sample.setEnabled(True)
        if not self._samples:
            self._cam_worker.set_mode_sample()
            self.lbl_scan_status.setText("Dang o Buoc 1: Lay mau...")
        else:
            self._enter_scan_mode()

    def _stop_camera(self):
        if self._cam_worker:
            self._cam_worker.stop()
        cv2.destroyWindow("Rubik Camera")
        self._cam_running = False
        self.btn_open_cam.setText("▶ Mo camera")
        self.btn_open_cam.setStyleSheet(self._btn("#00897B"))
        self.cb_cam_index.setEnabled(True)
        self.btn_scan_cams.setEnabled(True)
        self.btn_capture_sample.setEnabled(False)
        self.btn_capture_face.setEnabled(False)
        self.lbl_cam_status.setText("Camera da dong")

    def _on_cam_stopped(self):
        self._cam_running = False
        self.cb_cam_index.setEnabled(True)
        self.btn_scan_cams.setEnabled(True)
        self.btn_open_cam.setText("▶ Mo camera")
        self.btn_open_cam.setStyleSheet(self._btn("#00897B"))
        self.btn_capture_sample.setEnabled(False)
        self.btn_capture_face.setEnabled(False)

    def _on_cam_error(self, msg):
        self._cam_running = False
        self.cb_cam_index.setEnabled(True)
        self.btn_scan_cams.setEnabled(True)
        self.btn_open_cam.setText("▶ Mo camera")
        self.btn_open_cam.setStyleSheet(self._btn("#00897B"))
        self.lbl_cam_status.setText(f"Loi: {msg}")
        self.log_msg.emit(f"[Camera] {msg}")

    def _on_frame(self, frame):
        cv2.imshow("Rubik Camera", frame)
        cv2.waitKey(1)

    def _do_sample_capture(self):
        if self._cam_worker:
            self._cam_worker.trigger_capture()

    def _on_sample_ready(self, samples):
        self._samples = samples
        self._update_sample_dots()
        n = len(samples)
        self.lbl_sample_progress.setText(f"Da lay mau {n}/6 mau  ✓ San sang scan!")
        self.lbl_sample_progress.setStyleSheet(f"color:{GREEN_LITE};font-size:10px;font-weight:bold;")
        self.log_msg.emit(f"[Sample] Lay mau xong {n} mau")
        self._enter_scan_mode()

    def _update_sample_dots(self):
        for face, dot in self.sample_dots.items():
            if face in self._samples:
                b, g, r = self._samples[face]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                tc = "#000" if (r * 0.299 + g * 0.587 + b * 0.114) > 128 else "#fff"
                dot.setStyleSheet(f"background:{hex_color};color:{tc};border-radius:4px;font-weight:bold;font-size:11px;border:2px solid #4CAF50;")
            else:
                dot.setStyleSheet(f"background:#222;color:#555;border-radius:4px;font-weight:bold;font-size:11px;border:1px solid #333;")

    def _enter_scan_mode(self):
        if not self._cam_worker:
            return
        center = self._current_center_face()
        self._cam_worker.set_mode_scan(self._samples, center)
        self.btn_capture_face.setEnabled(True)
        self.lbl_scan_status.setText(f"Scan mode — chon mat trung tam va bam Chup mat")

    def _current_center_face(self):
        for face, btn in self.center_btns.items():
            if btn.isChecked():
                return face
        return "U"

    def _set_center_face(self, face):
        if self._cam_worker and self._samples:
            self._cam_worker.set_mode_scan(self._samples, face)
        self.lbl_scan_status.setText(f"Mat trung tam: {FACE_LABEL[face]}")

    def _do_face_capture(self):
        if self._cam_worker:
            self._cam_worker.trigger_capture()

    def _on_scan_done(self, labels):
        center = self._current_center_face()
        labels[4] = center
        self.face_captured.emit(center, labels)
        self.log_msg.emit(f"[Scan] {center} = {''.join(labels)}")

    def reset(self):
        self._samples = {}
        self._update_sample_dots()
        self.lbl_sample_progress.setText("Chua co mau mau nao")
        self.lbl_sample_progress.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        self.lbl_scan_status.setText("Can lay mau truoc (Buoc 1)")
        self.btn_capture_face.setEnabled(False)
        for btn in self.center_btns.values():
            btn.setChecked(False)
        if self._cam_worker:
            self._cam_worker.set_mode_sample()

    @staticmethod
    def _grp_style():
        return f"""
            QGroupBox {{
                color: {TEXT_MAIN};
                border: 1px solid {BORDER};
                border-radius: 8px;
                margin-top: 8px;
                font-weight: bold;
                font-size: 12px;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        """

    @staticmethod
    def _btn(bg):
        return f"""
            QPushButton {{
                background: {bg};
                color: white;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{ background: {bg}cc; }}
            QPushButton:disabled {{ background: #333; color: #666; }}
        """



class RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.serial_panel = SerialPanel()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        lay.addWidget(self.serial_panel)

        grp_sol = QGroupBox("🔍 Kết quả Solver")
        grp_sol.setStyleSheet(self._grp_style())
        sol_lay = QVBoxLayout(grp_sol)

        self.lbl_moves_title = QLabel("Kociemba solution:")
        self.lbl_moves_title.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        sol_lay.addWidget(self.lbl_moves_title)

        self.txt_original = QTextEdit()
        self.txt_original.setReadOnly(True)
        self.txt_original.setMaximumHeight(60)
        self.txt_original.setStyleSheet(self._txt_style())
        sol_lay.addWidget(self.txt_original)

        self.lbl_frame_title = QLabel("Data frame gửi robot:")
        self.lbl_frame_title.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        sol_lay.addWidget(self.lbl_frame_title)

        self.txt_frame = QTextEdit()
        self.txt_frame.setReadOnly(True)
        self.txt_frame.setMaximumHeight(80)
        self.txt_frame.setStyleSheet(self._txt_style(accent=True))
        sol_lay.addWidget(self.txt_frame)

        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(f"color: {ACCENT2}; font-size: 11px;")
        sol_lay.addWidget(self.lbl_count)

        self.btn_send_data = QPushButton("📡 Gửi Data")
        self.btn_send_data.setFixedHeight(36)
        self.btn_send_data.setEnabled(False)
        self.btn_send_data.clicked.connect(self._send_data)
        self.btn_send_data.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT};
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: #c0392b; }}
            QPushButton:disabled {{ background: #333; color: #666; }}
        """)
        sol_lay.addWidget(self.btn_send_data)
        lay.addWidget(grp_sol)

        grp_ctrl = QGroupBox("🎮 Điều khiển Robot")
        grp_ctrl.setStyleSheet(self._grp_style())
        ctrl_lay = QVBoxLayout(grp_ctrl)
        ctrl_lay.setSpacing(8)

        hint = QLabel("Gửi lệnh điều khiển tới robot:")
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        ctrl_lay.addWidget(hint)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_start = QPushButton("▶  START")
        self.btn_start.setFixedHeight(38)
        self.btn_start.clicked.connect(self._send_start)
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background: #2E7D32;
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: #1B5E20; }}
        """)

        self.btn_stop = QPushButton("⏹  STOP")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.clicked.connect(self._send_stop)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: #B71C1C;
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: #7F0000; }}
        """)

        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_lay.addLayout(ctrl_row)

        self.lbl_ctrl_hint = QLabel(
            f'START  ->  "control: start\\n"\n'
            f'STOP   ->  "control: stop\\n"'
        )
        self.lbl_ctrl_hint.setStyleSheet(f"""
            color: {TEXT_DIM};
            font-size: 10px;
            font-family: 'Courier New', monospace;
            background: #0d0d1a;
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 5px 8px;
        """)
        ctrl_lay.addWidget(self.lbl_ctrl_hint)
        lay.addWidget(grp_ctrl)

        grp_log = QGroupBox("📋 Log")
        grp_log.setStyleSheet(self._grp_style())
        log_lay = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(160)
        self.txt_log.setStyleSheet(self._txt_style())
        log_lay.addWidget(self.txt_log)

        self.btn_clear_log = QPushButton("Xóa log")
        self.btn_clear_log.setFixedHeight(24)
        self.btn_clear_log.clicked.connect(self.txt_log.clear)
        self.btn_clear_log.setStyleSheet(f"background:{BG_INPUT};color:{TEXT_DIM};border-radius:4px;font-size:10px;")
        log_lay.addWidget(self.btn_clear_log)
        lay.addWidget(grp_log)

        lay.addStretch()
        self._frame_str = ""

    def set_solution(self, solution: str):
        self.txt_original.setPlainText(solution)
        frame = build_data_frame(solution)
        self._frame_str = frame
        self.txt_frame.setPlainText(frame)
        n = len(solution.strip().split())
        self.lbl_count.setText(f"Tổng {n} nước  |  Sẵn sàng gửi")
        self.btn_send_data.setEnabled(True)

    def set_error(self, msg: str):
        self.txt_original.setPlainText(f"LỖI: {msg}")
        self.txt_frame.clear()
        self.lbl_count.setText("")
        self.btn_send_data.setEnabled(False)
        self._frame_str = ""

    def log(self, msg: str):
        self.txt_log.append(msg)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _send_data(self):
        if not self._frame_str:
            return
        if not self.serial_panel.is_connected():
            QMessageBox.warning(self, "Serial", "Chưa kết nối cổng Serial!")
            return
        ok = self.serial_panel.send(self._frame_str)
        if ok:
            self.log(f"[TX data]  {self._frame_str}")

    def _send_start(self):
        if not self.serial_panel.is_connected():
            QMessageBox.warning(self, "Serial", "Chưa kết nối cổng Serial!")
            return
        frame = build_control_frame("start")
        ok = self.serial_panel.send(frame)
        if ok:
            self.log(f"[TX ctrl]  {frame}")

    def _send_stop(self):
        if not self.serial_panel.is_connected():
            QMessageBox.warning(self, "Serial", "Chưa kết nối cổng Serial!")
            return
        frame = build_control_frame("stop")
        ok = self.serial_panel.send(frame)
        if ok:
            self.log(f"[TX ctrl]  {frame}")

    @staticmethod
    def _grp_style():
        return f"""
            QGroupBox {{
                color: {TEXT_MAIN};
                border: 1px solid {BORDER};
                border-radius: 8px;
                margin-top: 8px;
                font-weight: bold;
                font-size: 12px;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        """

    @staticmethod
    def _txt_style(accent=False):
        border_color = ACCENT2 if accent else BORDER
        return f"""
            QTextEdit {{
                background: #0d0d1a;
                color: {'#F0C040' if accent else TEXT_MAIN};
                border: 1px solid {border_color};
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 4px;
            }}
        """


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🟧 Rubik's Cube Solver — Robot 5 Motor")
        self.setMinimumSize(1200, 750)
        self._faces: dict[str, list] = {f: None for f in FACES}
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(54)
        title_bar.setStyleSheet(f"background: {BG_CARD}; border-bottom: 1px solid {BORDER};")
        tbl = QHBoxLayout(title_bar)
        tbl.setContentsMargins(20, 0, 20, 0)

        ico = QLabel("🟧")
        ico.setFont(QFont("Segoe UI Emoji", 18))
        tbl.addWidget(ico)

        t = QLabel("Rubik's Cube Solver")
        t.setStyleSheet(f"color: {ACCENT2}; font-size: 20px; font-weight: bold; letter-spacing: 2px;")
        tbl.addWidget(t)
        tbl.addStretch()

        sub = QLabel("Robot 5 Motor · U->D Auto Convert · PyQt5")
        sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        tbl.addWidget(sub)
        root.addWidget(title_bar)

        # ── Main content: 3 columns
        content = QWidget()
        content.setStyleSheet(f"background: {BG_DARK};")
        cl = QHBoxLayout(content)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(12)

        # Left panel
        self.left_panel = LeftPanel()
        self.left_panel.setFixedWidth(310)
        self.left_panel.face_captured.connect(self._on_face_captured)
        self.left_panel.log_msg.connect(self._log)
        scroll_left = QScrollArea()
        scroll_left.setWidget(self.left_panel)
        scroll_left.setWidgetResizable(True)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_left.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_left.setFixedWidth(320)
        cl.addWidget(scroll_left)

        # Center: Cube net + Solve button
        center_w = QWidget()
        center_l = QVBoxLayout(center_w)
        center_l.setContentsMargins(0, 0, 0, 0)
        center_l.setSpacing(8)

        self.cube_net = CubeNetWidget()
        for face, fw in self.cube_net.faces.items():
            for cell in fw.cells:
                if not cell.is_center:
                    cell.color_changed.connect(self._on_cell_manual_change)
        self.cube_net.quick_input_clicked.connect(self._open_quick_input)

        scroll_center = QScrollArea()
        scroll_center.setWidget(self.cube_net)
        scroll_center.setWidgetResizable(True)
        scroll_center.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        center_l.addWidget(scroll_center)

        # Solve + Reset row
        btn_row = QHBoxLayout()
        self.btn_solve = QPushButton("🔍 Giải Rubik")
        self.btn_solve.setFixedHeight(42)
        self.btn_solve.clicked.connect(self.solve)
        self.btn_solve.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT2};
                color: #000;
                border-radius: 8px;
                font-weight: bold;
                font-size: 15px;
            }}
            QPushButton:hover {{ background: #d4a800; }}
        """)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setFixedHeight(42)
        self.btn_reset.setFixedWidth(100)
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_reset.setStyleSheet(f"""
            QPushButton {{
                background: #37474F;
                color: {TEXT_MAIN};
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
            }}
        """)

        btn_row.addWidget(self.btn_solve)
        btn_row.addWidget(self.btn_reset)
        center_l.addLayout(btn_row)
        cl.addWidget(center_w, 1)

        # Right panel
        self.right_panel = RightPanel()
        self.right_panel.setFixedWidth(340)
        scroll_right = QScrollArea()
        scroll_right.setWidget(self.right_panel)
        scroll_right.setWidgetResizable(True)
        scroll_right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_right.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_right.setFixedWidth(355)
        cl.addWidget(scroll_right)

        root.addWidget(content)

        # ── Status bar
        self.statusBar().setStyleSheet(f"background: {BG_CARD}; color: {TEXT_DIM}; font-size: 11px;")
        self.statusBar().showMessage("Sẵn sàng · Dùng camera hoặc nhập màu thủ công bằng cách click vào ô trên Cube Net")

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {BG_DARK}; }}
            QScrollBar:vertical {{
                background: {BG_CARD}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER}; border-radius: 4px; min-height: 20px;
            }}
        """)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_face_captured(self, face: str, labels: list):
        self._faces[face] = labels
        self.cube_net.set_face(face, labels)
        self._log(f"[Face] {face} đã scan: {''.join(labels)}")

    def _on_cell_manual_change(self, face: str, index: int, color: str):
        if self._faces[face] is None:
            self._faces[face] = self.cube_net.get_all_labels()[face]
            self.cube_net.faces[face].mark_captured(True)
        self._faces[face][index] = color
        self._log(f"[Manual] {face}[{index}] = {color}")

    def _open_quick_input(self, face: str):
        current = self.cube_net.get_all_labels()[face]
        dlg = QuickInputDialog(face, current, self)
        dlg.confirmed.connect(self._on_quick_input_confirmed)
        # Hien thi dialog giua man hinh
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        geom = self.geometry()
        dlg.move(
            geom.x() + (geom.width()  - dlg.width())  // 2,
            geom.y() + (geom.height() - dlg.height()) // 2,
        )

    def _on_quick_input_confirmed(self, face: str, labels: list):
        self._faces[face] = labels
        self.cube_net.set_face(face, labels)
        self._log(f"[Quick] {face} = {''.join(labels)}")

    def solve(self):
        all_labels = self.cube_net.get_all_labels()
        for f, labels in all_labels.items():
            self._faces[f] = labels

        cube_str = "".join("".join(self._faces[f]) for f in "URFDLB")
        self._log(f"[Solve] Cube string: {cube_str}")

        if len(cube_str) != 54:
            self.right_panel.set_error("Cube string không đủ 54 ký tự!")
            return

        try:
            solution = kociemba.solve(cube_str)
            self._log(f"[Solve] Kociemba: {solution}")
            self.right_panel.set_solution(solution)
            n = len(solution.strip().split())
            self.statusBar().showMessage(f"Giải thành công · {n} nước")
        except Exception as e:
            self._log(f"[Solve] LỖI: {e}")
            self.right_panel.set_error(str(e))
            self.statusBar().showMessage(f"Lỗi solver: {e}")

    def reset_all(self):
        self._faces = {f: None for f in FACES}
        self.cube_net.reset()
        self.left_panel.reset()
        self.right_panel.txt_original.clear()
        self.right_panel.txt_frame.clear()
        self.right_panel.lbl_count.setText("")
        self.right_panel.btn_send_data.setEnabled(False)
        self.statusBar().showMessage("Đã reset · Sẵn sàng")
        self._log("[Reset] Đã reset toàn bộ trạng thái")

    def _log(self, msg: str):
        self.right_panel.log(msg)

    def closeEvent(self, event):
        if self.left_panel._cam_running:
            self.left_panel._stop_camera()
        cv2.destroyAllWindows()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG_DARK))
    palette.setColor(QPalette.WindowText,      QColor(TEXT_MAIN))
    palette.setColor(QPalette.Base,            QColor(BG_CARD))
    palette.setColor(QPalette.AlternateBase,   QColor(BG_INPUT))
    palette.setColor(QPalette.Text,            QColor(TEXT_MAIN))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT_MAIN))
    palette.setColor(QPalette.Button,          QColor(BG_INPUT))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
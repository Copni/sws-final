from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMutex, QObject, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sws.application import AppController
from sws.capture import Mp4Reader, WebcamCapture, is_probably_black_frame, list_webcam_devices
from sws.inference import TranslationEngine
from sws.output import VirtualCameraSink, compose_frame, draw_debug_overlay, export_mp4
from sws.preprocessing import HandVectorResult, MediaPipeHandsExtractor
from sws.subtitles import PredictionStabilizer, SubtitleBuffer


MODEL_STATUS_TEXT = {
    "model_not_loaded": "Aucun modele charge",
    "model_loading": "Chargement du modele",
    "model_ready": "Modele charge",
    "labels_missing": "Modele charge, labels manquants",
    "model_invalid": "Modele invalide",
}


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    status_changed = Signal(str)
    output_changed = Signal(str)
    metrics_changed = Signal(float, float)

    def __init__(
        self,
        source_kind: str,
        source_value: str,
        engine: TranslationEngine,
        use_virtual_camera: bool,
        show_debug_data: bool,
    ) -> None:
        super().__init__()
        self.source_kind = source_kind
        self.source_value = source_value
        self.engine = engine
        self.use_virtual_camera = use_virtual_camera
        self.show_debug_data = show_debug_data
        self._running = True
        self._paused = False
        self._mutex = QMutex()
        self._black_frame_count = 0
        self._black_frame_reported = False
        self._last_hand_result: HandVectorResult | None = None
        self._last_confidence: float | None = None

    def pause(self, paused: bool) -> None:
        self._mutex.lock()
        self._paused = paused
        self._mutex.unlock()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        extractor = None
        virtual_camera = VirtualCameraSink()
        stabilizer = PredictionStabilizer()
        subtitles = SubtitleBuffer()
        last_inference = 0.0
        frame_counter = 0
        fps_window_start = time.monotonic()

        try:
            extractor = MediaPipeHandsExtractor()
        except Exception as exc:
            self.status_changed.emit(f"MediaPipe indisponible : {exc}")

        try:
            if self.source_kind == "webcam":
                self._run_webcam(extractor, virtual_camera, stabilizer, subtitles, last_inference)
            else:
                self._run_mp4(extractor, virtual_camera, stabilizer, subtitles, last_inference)
        finally:
            if extractor:
                extractor.close()
            virtual_camera.close()

    def _run_webcam(
        self,
        extractor: MediaPipeHandsExtractor | None,
        virtual_camera: VirtualCameraSink,
        stabilizer: PredictionStabilizer,
        subtitles: SubtitleBuffer,
        last_inference: float,
    ) -> None:
        index_text, backend_text = self.source_value.split(":", 1)
        capture = WebcamCapture(int(index_text), int(backend_text))
        ok, message = capture.start()
        self.status_changed.emit(message)
        if not ok:
            return

        try:
            last_frame_id = 0.0
            while self._running:
                frame = capture.read_latest()
                if not frame or frame.timestamp == last_frame_id:
                    self.msleep(5)
                    continue
                last_frame_id = frame.timestamp
                last_inference = self._handle_frame(
                    frame.bgr,
                    frame.timestamp,
                    frame.fps,
                    extractor,
                    stabilizer,
                    subtitles,
                    virtual_camera,
                    last_inference,
                )
        finally:
            capture.stop()

    def _run_mp4(
        self,
        extractor: MediaPipeHandsExtractor | None,
        virtual_camera: VirtualCameraSink,
        stabilizer: PredictionStabilizer,
        subtitles: SubtitleBuffer,
        last_inference: float,
    ) -> None:
        reader = Mp4Reader(self.source_value)
        ok, message = reader.open()
        self.status_changed.emit(message)
        if not ok:
            return

        frame_delay_ms = int(1000 / max(1.0, reader.fps))
        try:
            while self._running:
                self._mutex.lock()
                paused = self._paused
                self._mutex.unlock()
                if paused:
                    self.msleep(20)
                    continue

                frame = reader.read()
                if not frame:
                    self.status_changed.emit("Lecture terminee")
                    break

                last_inference = self._handle_frame(
                    frame.bgr,
                    frame.timestamp,
                    reader.fps,
                    extractor,
                    stabilizer,
                    subtitles,
                    virtual_camera,
                    last_inference,
                )
                self.msleep(frame_delay_ms)
        finally:
            reader.close()

    def _handle_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        fps: float,
        extractor: MediaPipeHandsExtractor | None,
        stabilizer: PredictionStabilizer,
        subtitles: SubtitleBuffer,
        virtual_camera: VirtualCameraSink,
        last_inference: float,
    ) -> float:
        latency = 0.0
        self._update_black_frame_diagnostic(frame_bgr)
        if extractor and timestamp - last_inference >= 0.05:
            try:
                hand_result = extractor.process(frame_bgr)
                translation = self.engine.translate(hand_result.vector)
                self._last_hand_result = hand_result
                self._last_confidence = translation.confidence
                latency = translation.latency_ms or 0.0
                subtitles.apply(stabilizer.update(translation, timestamp))
                last_inference = timestamp
            except Exception as exc:
                self.status_changed.emit(f"Pretraitement indisponible : {exc}")

        display_frame = frame_bgr
        if self.show_debug_data:
            display_frame = draw_debug_overlay(frame_bgr, self._last_hand_result, self._last_confidence)

        composed = compose_frame(display_frame, subtitles.current_text)
        if self.use_virtual_camera and not virtual_camera.active:
            ok, message = virtual_camera.start(composed.shape[1], composed.shape[0], fps)
            self.output_changed.emit(message)
            if not ok:
                self.use_virtual_camera = False
        if virtual_camera.active:
            virtual_camera.send_bgr(composed)

        self.frame_ready.emit(_bgr_to_qimage(composed))
        self.metrics_changed.emit(fps, latency)
        return last_inference

    def _update_black_frame_diagnostic(self, frame_bgr: np.ndarray) -> None:
        if is_probably_black_frame(frame_bgr):
            self._black_frame_count += 1
            if self._black_frame_count >= 20 and not self._black_frame_reported:
                self.status_changed.emit(
                    "Webcam active, mais image noire recue. Verifiez cache physique, "
                    "confidentialite Windows, exposition ou application camera ouverte."
                )
                self._black_frame_reported = True
            return

        if self._black_frame_reported:
            self.status_changed.emit("Webcam active")
        self._black_frame_count = 0
        self._black_frame_reported = False


class ExportWorker(QThread):
    progress_changed = Signal(int)
    finished_with_message = Signal(bool, str)

    def __init__(self, source: str, destination: str, engine: TranslationEngine) -> None:
        super().__init__()
        self.source = source
        self.destination = destination
        self.engine = engine
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        stabilizer = PredictionStabilizer()
        subtitles = SubtitleBuffer()
        extractor = None
        last_inference = -1.0

        try:
            extractor = MediaPipeHandsExtractor()
        except Exception:
            extractor = None

        def compose(frame: np.ndarray, media_time: float) -> np.ndarray:
            nonlocal last_inference
            if extractor and media_time - last_inference >= 0.05:
                result = extractor.process(frame)
                translation = self.engine.translate(result.vector)
                subtitles.apply(stabilizer.update(translation, media_time))
                last_inference = media_time
            return compose_frame(frame, subtitles.current_text)

        ok, message = export_mp4(
            self.source,
            self.destination,
            compose,
            progress=lambda value: self.progress_changed.emit(int(value * 100)),
            should_cancel=lambda: self._cancel,
        )
        if extractor:
            extractor.close()
        self.finished_with_message.emit(ok, message)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speak With Sign")
        self.resize(980, 760)
        self.controller = AppController()
        self.video_worker: VideoWorker | None = None
        self.export_worker: ExportWorker | None = None
        self.mp4_path = ""

        self._build_ui()
        self._refresh_webcams()
        self._update_model_status()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_video()
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel()
            self.export_worker.wait(2000)
        event.accept()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        self.video_label = QLabel("Aucune source")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background:#222;color:white;")
        layout.addWidget(self.video_label, 1)

        source_box = QGroupBox("Source")
        source_layout = QGridLayout(source_box)
        self.webcam_radio = QRadioButton("Webcam")
        self.webcam_radio.setChecked(True)
        self.mp4_radio = QRadioButton("Fichier MP4")
        self.webcam_combo = QComboBox()
        self.refresh_webcam_button = QPushButton("Actualiser")
        self.mp4_path_edit = QLineEdit()
        self.mp4_path_edit.setReadOnly(True)
        self.choose_mp4_button = QPushButton("Choisir MP4")
        source_layout.addWidget(self.webcam_radio, 0, 0)
        source_layout.addWidget(self.webcam_combo, 0, 1)
        source_layout.addWidget(self.refresh_webcam_button, 0, 2)
        source_layout.addWidget(self.mp4_radio, 1, 0)
        source_layout.addWidget(self.mp4_path_edit, 1, 1)
        source_layout.addWidget(self.choose_mp4_button, 1, 2)
        layout.addWidget(source_box)

        model_box = QGroupBox("Modele")
        model_layout = QHBoxLayout(model_box)
        self.model_status = QLabel()
        self.load_model_button = QPushButton("Choisir dossier modele")
        self.clear_model_button = QPushButton("Retirer")
        self.cpu_label = QLabel("Backend scikit-learn CPU")
        model_layout.addWidget(self.model_status, 1)
        model_layout.addWidget(self.cpu_label)
        model_layout.addWidget(self.load_model_button)
        model_layout.addWidget(self.clear_model_button)
        layout.addWidget(model_box)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Demarrer")
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("Arreter")
        self.stop_button.setEnabled(False)
        self.virtual_camera_check = QCheckBox("Sortie camera virtuelle")
        self.debug_data_check = QCheckBox("Afficher les donnees")
        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.virtual_camera_check)
        controls.addWidget(self.debug_data_check)
        controls.addStretch(1)
        layout.addLayout(controls)

        export_box = QGroupBox("Export MP4")
        export_layout = QGridLayout(export_box)
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setReadOnly(True)
        self.choose_export_button = QPushButton("Destination")
        self.export_button = QPushButton("Exporter")
        self.export_progress = QProgressBar()
        export_layout.addWidget(self.export_path_edit, 0, 0)
        export_layout.addWidget(self.choose_export_button, 0, 1)
        export_layout.addWidget(self.export_button, 0, 2)
        export_layout.addWidget(self.export_progress, 1, 0, 1, 3)
        layout.addWidget(export_box)

        status_layout = QHBoxLayout()
        self.source_status = QLabel("Aucune source")
        self.output_status = QLabel("Sortie externe inactive")
        self.metrics_label = QLabel("FPS: - | Latence: -")
        status_layout.addWidget(self.source_status)
        status_layout.addWidget(self.output_status)
        status_layout.addWidget(self.metrics_label)
        layout.addLayout(status_layout)

        self.setCentralWidget(root)

        self.refresh_webcam_button.clicked.connect(self._refresh_webcams)
        self.choose_mp4_button.clicked.connect(self._choose_mp4)
        self.load_model_button.clicked.connect(self._choose_model_dir)
        self.clear_model_button.clicked.connect(self._clear_model)
        self.start_button.clicked.connect(self._start_video)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_video)
        self.choose_export_button.clicked.connect(self._choose_export)
        self.export_button.clicked.connect(self._start_export)

    def _refresh_webcams(self) -> None:
        self.webcam_combo.clear()
        devices = list_webcam_devices()
        if not devices:
            self.webcam_combo.addItem("Webcam 0 - Auto", "0:0")
            self.source_status.setText("Aucune webcam detectee automatiquement")
            return
        for device in devices:
            self.webcam_combo.addItem(device.display_name, f"{device.index}:{device.backend}")
        self.source_status.setText(f"{len(devices)} webcam(s) detectee(s). Cliquez sur Demarrer.")

    def _choose_mp4(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un MP4", "", "Videos (*.mp4)")
        if path:
            self.mp4_path = path
            self.mp4_path_edit.setText(path)
            self.mp4_radio.setChecked(True)
            self.source_status.setText(f"MP4 selectionne : {Path(path).name}")

    def _choose_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choisir le dossier modele")
        if not path:
            return
        self.model_status.setText("Chargement du modele...")
        state = self.controller.load_model_dir(path)
        self._update_model_status()
        if state.model_status == "model_invalid":
            QMessageBox.warning(self, "Modele invalide", state.model_message)

    def _clear_model(self) -> None:
        self.controller.clear_model()
        self._update_model_status()

    def _update_model_status(self) -> None:
        status = self.controller.state.model_status
        label = MODEL_STATUS_TEXT.get(status, status)
        self.model_status.setText(f"{label} - {self.controller.state.model_message}")

    def _start_video(self) -> None:
        if self.video_worker and self.video_worker.isRunning():
            return
        if self.webcam_radio.isChecked():
            source_kind = "webcam"
            source_value = str(self.webcam_combo.currentData() or "0:0")
        else:
            if not self.mp4_path:
                QMessageBox.information(self, "Source manquante", "Choisissez un fichier MP4.")
                return
            source_kind = "mp4"
            source_value = self.mp4_path

        self.video_worker = VideoWorker(
            source_kind,
            source_value,
            self.controller.engine,
            self.virtual_camera_check.isChecked(),
            self.debug_data_check.isChecked(),
        )
        self.video_worker.frame_ready.connect(self._set_frame)
        self.video_worker.status_changed.connect(self.source_status.setText)
        self.video_worker.output_changed.connect(self.output_status.setText)
        self.video_worker.metrics_changed.connect(self._set_metrics)
        self.video_worker.finished.connect(self._video_finished)
        self.video_worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(source_kind == "mp4")
        if self.virtual_camera_check.isChecked():
            self.output_status.setText("Initialisation de la camera virtuelle")
        else:
            self.output_status.setText("Apercu local actif. Sortie externe inactive")

    def _toggle_pause(self) -> None:
        if not self.video_worker:
            return
        paused = self.pause_button.text() == "Pause"
        self.video_worker.pause(paused)
        self.pause_button.setText("Reprendre" if paused else "Pause")

    def _stop_video(self) -> None:
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait(2000)
            self.video_worker = None
        self._video_finished()

    def _video_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.output_status.setText("Sortie externe inactive")

    def _set_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def _set_metrics(self, fps: float, latency: float) -> None:
        self.metrics_label.setText(f"FPS: {fps:.1f} | Latence: {latency:.1f} ms")

    def _choose_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Destination MP4", "", "Videos (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.export_path_edit.setText(path)

    def _start_export(self) -> None:
        if not self.mp4_path:
            QMessageBox.information(self, "Source manquante", "L'export utilise un fichier MP4 source.")
            return
        destination = self.export_path_edit.text().strip()
        if not destination:
            QMessageBox.information(self, "Destination manquante", "Choisissez un fichier de sortie.")
            return
        if Path(destination).exists():
            answer = QMessageBox.question(
                self,
                "Fichier existant",
                "Le fichier existe deja. Choisir une autre destination ?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._choose_export()
            destination = self.export_path_edit.text().strip()
            if Path(destination).exists():
                return

        self.export_progress.setValue(0)
        self.export_worker = ExportWorker(self.mp4_path, destination, self.controller.engine)
        self.export_worker.progress_changed.connect(self.export_progress.setValue)
        self.export_worker.finished_with_message.connect(self._export_finished)
        self.export_worker.start()
        self.output_status.setText("Export en cours")

    def _export_finished(self, ok: bool, message: str) -> None:
        self.output_status.setText(message)
        if ok:
            self.export_progress.setValue(100)
        else:
            QMessageBox.warning(self, "Export MP4", message)


def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    frame_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    height, width, channels = frame_rgb.shape
    bytes_per_line = channels * width
    image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return image.copy()


def run_app() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

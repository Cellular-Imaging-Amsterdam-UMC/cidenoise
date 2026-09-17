from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import yaml
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('ci.w_cidenoise.bilayers_launcher')
from PyQt6.QtCore import QProcess, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QTextCursor
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSpinBox, QTextEdit, QToolButton, QVBoxLayout, QWidget
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'config.yaml'
ICON = ROOT / 'gui' / 'icon.svg'
SETTINGS = ROOT / '.last_launcher_settings.json'

class MultiSelectList(QListWidget):

    def __init__(self, options: list[dict], selected: list[str]):
        super().__init__()
        selected_values = set(selected)
        self.setMinimumHeight(120)
        self.setMaximumHeight(160)
        for option in options:
            item = QListWidgetItem(str(option['label']))
            item.setData(Qt.ItemDataRole.UserRole, option['value'])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if option['value'] in selected_values else Qt.CheckState.Unchecked)
            self.addItem(item)

    def values(self) -> list[str]:
        return [str(self.item(index).data(Qt.ItemDataRole.UserRole)) for index in range(self.count()) if self.item(index).checkState() == Qt.CheckState.Checked]

class NoWheelComboBox(QComboBox):
    """A selector that never changes its value while the form is scrolled."""

    def wheelEvent(self, event) -> None:
        event.ignore()

class NoWheelSpinBox(QSpinBox):
    """An integer editor that ignores mouse-wheel value changes."""

    def wheelEvent(self, event) -> None:
        event.ignore()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """A floating-point editor that ignores mouse-wheel value changes."""

    def wheelEvent(self, event) -> None:
        event.ignore()

class CollapsiblePanel(QWidget):

    def __init__(self, title: str):
        super().__init__()
        self._collapsed_window_height: int | None = None
        self.toggle = QToolButton(text=title, checkable=True, checked=False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.clicked.connect(self._set_open)
        self.content = QWidget()
        self.content.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def _set_open(self, opened: bool) -> None:
        window = self.window()
        if opened and (not window.isMaximized()) and (not window.isFullScreen()):
            self._collapsed_window_height = window.height()
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        self.content.setVisible(opened)
        self.content.updateGeometry()
        self.updateGeometry()
        target_height = None if opened else self._collapsed_window_height
        QTimer.singleShot(0, lambda: self._fit_window_to_visible_content(target_height, retry=True))

    def _fit_window_to_visible_content(self, target_height: int | None=None, *, retry: bool=False) -> None:
        window = self.window()
        if window.isMaximized() or window.isFullScreen():
            return
        own_layout = self.layout()
        if own_layout is not None:
            own_layout.activate()
        central = getattr(window, 'centralWidget', lambda: None)()
        if central is not None and central.layout() is not None:
            central.layout().activate()
            central.updateGeometry()
        window.updateGeometry()
        hint = window.sizeHint()
        if target_height is not None:
            window.resize(window.width(), target_height)
        elif hint.isValid():
            window.resize(max(window.width(), hint.width()), hint.height())
        if retry:
            QTimer.singleShot(0, lambda: self._fit_window_to_visible_content(target_height))

def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))

def _append_parameters(command: list[str], config: dict, values: dict) -> list[str]:
    for item in sorted(config.get('parameters', []), key=lambda entry: int(entry.get('cli_order', 0))):
        value = values.get(item['name'], item.get('default'))
        passes_boolean_value = item.get('type') == 'checkbox' and item.get('append_value', False)
        if value in (None, '', []) or (value is False and (not passes_boolean_value)):
            continue
        command.append(str(item['cli_tag']))
        if item.get('type') != 'checkbox' or item.get('append_value', False):
            if isinstance(value, list):
                value = ','.join(value)
            command.append(str(value))
    command.append('--local')
    return command

def build_docker_command(config: dict, values: dict, input_dir: str, output_dir: str, gpu: bool=True) -> list[str]:
    image = config['docker_image']
    image_name = f"{image['name']}:{image.get('tag', 'latest')}"
    command = ['docker', 'run', '--rm']
    if gpu:
        command += ['--gpus', 'all']
    command += ['-v', f'{Path(input_dir).resolve()}:/data/in:ro', '-v', f'{Path(output_dir).resolve()}:/data/out', image_name]
    return _append_parameters(command, config, values)

def build_local_command(config: dict, values: dict, input_dir: str, output_dir: str, python_executable: str | None=None) -> list[str]:
    command = [python_executable or sys.executable, str(ROOT / 'wrapper.py'), '--infolder', input_dir, '--outfolder', output_dir]
    return _append_parameters(command, config, values)

class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.widgets: dict[str, QWidget] = {}
        self.run_buttons: list[QPushButton] = []
        self.setWindowTitle('CI Denoise - Bilayers Launcher')
        self.setWindowIcon(QIcon(str(ICON)))
        self.setMinimumWidth(960)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        title = QLabel('CI Denoise - Bilayers')
        title.setFont(QFont('Segoe UI', 16, QFont.Weight.Bold))
        layout.addWidget(title)
        folders = QGroupBox('Data folders')
        folder_form = QFormLayout(folders)
        self.input_path = QLineEdit(str(ROOT / 'inputfolder'))
        self.output_path = QLineEdit(str(ROOT / 'outputfolder'))
        folder_form.addRow('Input folder:', self._folder_row(self.input_path))
        folder_form.addRow('Output folder:', self._folder_row(self.output_path))
        layout.addWidget(folders)
        runtime = QGroupBox('Docker runtime')
        runtime_form = QFormLayout(runtime)
        self.gpu = QCheckBox('Expose NVIDIA GPU to container')
        self.gpu.setChecked(True)
        self.gpu.setToolTip("Adds '--gpus all' to the Docker command.")
        runtime_form.addRow('GPU:', self.gpu)
        layout.addWidget(runtime)
        parameters = QGroupBox('Parameters')
        parameter_layout = QVBoxLayout(parameters)
        main = QWidget()
        main_grid = self._parameter_grid(main)
        advanced = CollapsiblePanel('Advanced parameters')
        self.advanced_panel = advanced
        advanced_grid = self._parameter_grid(advanced.content, left_margin=18)
        main_count = advanced_count = 0
        for spec in self.config.get('parameters', []):
            widget = self._widget(spec)
            widget.setToolTip(spec.get('description', ''))
            label = QLabel(spec.get('label', spec['name']))
            label.setToolTip(spec.get('description', ''))
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if spec.get('mode') == 'advanced':
                self._add_two_column_row(advanced_grid, advanced_count, label, widget)
                advanced_count += 1
            else:
                self._add_two_column_row(main_grid, main_count, label, widget)
                main_count += 1
            self.widgets[spec['name']] = widget
        parameter_layout.addWidget(main)
        if advanced_count:
            parameter_layout.addWidget(advanced)
        layout.addWidget(parameters)
        layout.addWidget(QLabel('Command preview:'))
        self.preview = QTextEdit(readOnly=True)
        self.preview.setMaximumHeight(125)
        self.preview.setFont(QFont('Consolas', 9))
        layout.addWidget(self.preview)
        buttons = QHBoxLayout()
        restore = QPushButton('Restore settings')
        restore.clicked.connect(self.restore)
        buttons.addWidget(restore)
        load = QPushButton('Load settings')
        load.clicked.connect(self.load)
        buttons.addWidget(load)
        save = QPushButton('Save settings')
        save.clicked.connect(self.save_as)
        buttons.addWidget(save)
        buttons.addStretch()
        run_local = QPushButton('Run Locally')
        run_local.setToolTip('Run wrapper.py with the Python environment used to launch this window.')
        run_local.clicked.connect(self.run_local)
        buttons.addWidget(run_local)
        self.run_buttons.append(run_local)
        run_docker = QPushButton('Run Docker')
        run_docker.setToolTip('Run the configured container image with Docker.')
        run_docker.clicked.connect(self.run_docker)
        buttons.addWidget(run_docker)
        self.run_buttons.append(run_docker)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.log = QTextEdit(readOnly=True)
        self.log.setMinimumHeight(180)
        self.log.setFont(QFont('Consolas', 9))
        layout.addWidget(self.log)
        self._connect_signals()
        self.refresh()

    @staticmethod
    def _parameter_grid(parent: QWidget, left_margin: int=0) -> QGridLayout:
        grid = QGridLayout(parent)
        grid.setContentsMargins(left_margin, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    @staticmethod
    def _add_two_column_row(grid: QGridLayout, index: int, label: QLabel, widget: QWidget) -> None:
        column = 0 if index % 2 == 0 else 2
        row = index // 2
        grid.addWidget(label, row, column)
        grid.addWidget(widget, row, column + 1)

    def _folder_row(self, edit: QLineEdit) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        button = QPushButton('Browse...')
        button.clicked.connect(lambda: self._browse(edit))
        row.addWidget(button)
        return box

    def _browse(self, edit: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, 'Select folder', edit.text())
        if selected:
            edit.setText(selected)

    def _widget(self, spec: dict) -> QWidget:
        if spec.get('multiselect'):
            return MultiSelectList(spec.get('options', []), spec.get('default', []))
        if spec.get('type') == 'checkbox':
            widget = QCheckBox()
            widget.setChecked(bool(spec.get('default')))
            return widget
        if spec.get('options'):
            widget = NoWheelComboBox()
            for option in spec['options']:
                widget.addItem(str(option['label']), option['value'])
            widget.setCurrentIndex(max(widget.findData(spec.get('default')), 0))
            return widget
        if spec.get('type') == 'integer':
            widget = NoWheelSpinBox()
            widget.setRange(int(spec.get('minimum', -999999)), int(spec.get('maximum', 999999)))
            widget.setValue(int(spec.get('default', 0)))
            return widget
        if spec.get('type') == 'float':
            widget = NoWheelDoubleSpinBox()
            widget.setDecimals(6)
            widget.setRange(float(spec.get('minimum', -999999)), float(spec.get('maximum', 999999)))
            widget.setValue(float(spec.get('default', 0)))
            return widget
        return QLineEdit(str(spec.get('default', '')))

    def _connect_signals(self) -> None:
        self.input_path.textChanged.connect(self.refresh)
        self.output_path.textChanged.connect(self.refresh)
        self.gpu.stateChanged.connect(self.refresh)
        for widget in self.widgets.values():
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.refresh)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.refresh)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.refresh)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.refresh)
            elif isinstance(widget, MultiSelectList):
                widget.itemChanged.connect(self.refresh)

    def values(self) -> dict:
        values = {}
        for name, widget in self.widgets.items():
            if isinstance(widget, MultiSelectList):
                values[name] = widget.values()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[name] = widget.value()
            else:
                values[name] = widget.text()
        return values

    def docker_command(self) -> list[str]:
        return build_docker_command(self.config, self.values(), self.input_path.text(), self.output_path.text(), self.gpu.isChecked())

    def local_command(self) -> list[str]:
        return build_local_command(self.config, self.values(), self.input_path.text(), self.output_path.text())

    def command(self) -> list[str]:
        """Return the Docker command for backward compatibility."""
        return self.docker_command()

    def refresh(self) -> None:
        self.preview.setPlainText('Docker:\n' + subprocess.list2cmdline(self.docker_command()) + '\n\nLocal Python:\n' + subprocess.list2cmdline(self.local_command()))

    def run_docker(self) -> None:
        self.save()
        self.start_process(self.docker_command())

    def run_local(self) -> None:
        self.save()
        self.start_process(self.local_command())

    def start_process(self, command):
        Path(self.output_path.text()).mkdir(parents=True, exist_ok=True)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(lambda error: self.log.append(f'Process error: {error.name}'))
        self.log.clear()
        for button in self.run_buttons:
            button.setEnabled(False)
        self.process.start(command[0], command[1:])

    def read_log(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def finished(self, exit_code, status):
        self.read_log()
        self.log.append(f'Exit code: {exit_code}')
        for button in self.run_buttons:
            button.setEnabled(True)

    def save(self, path: str | Path | None=None) -> None:
        destination = Path(path) if path is not None else SETTINGS
        destination.write_text(json.dumps(self._settings_data(), indent=2), encoding='utf-8')

    def save_as(self) -> None:
        selected, _filter = QFileDialog.getSaveFileName(self, 'Save settings', str(ROOT / 'cidenoise-settings.json'), 'JSON settings (*.json);;All files (*)')
        if not selected:
            return
        path = Path(selected)
        if not path.suffix:
            path = path.with_suffix('.json')
        try:
            self.save(path)
        except OSError as exc:
            QMessageBox.critical(self, 'Could not save settings', str(exc))

    def load(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(self, 'Load settings', str(ROOT), 'JSON settings (*.json);;All files (*)')
        if selected:
            self._load_settings(Path(selected), show_errors=True)

    def restore(self) -> None:
        if not SETTINGS.exists():
            return
        self._load_settings(SETTINGS, show_errors=True)

    def _settings_data(self) -> dict:
        return {'values': self.values(), 'input': self.input_path.text(), 'output': self.output_path.text(), 'gpu': self.gpu.isChecked()}

    def _load_settings(self, path: Path, *, show_errors: bool=False) -> None:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('The settings file must contain a JSON object.')
        except (OSError, ValueError) as exc:
            if show_errors:
                QMessageBox.critical(self, 'Could not load settings', str(exc))
            return
        self._apply_settings(data)

    def _apply_settings(self, data: dict) -> None:
        self.input_path.setText(data.get('input', ''))
        self.output_path.setText(data.get('output', ''))
        self.gpu.setChecked(data.get('gpu', True))
        restored_values = dict(data.get('values', {}))
        if isinstance(restored_values.get('benchmark'), bool):
            restored_values['benchmark'] = '2d-crop' if restored_values['benchmark'] else 'off'
        # Preserve descriptions from launcher settings saved before the eight menus.
        legacy = json.loads(restored_values.get('structures', '{}'))
        for channel, description in legacy.items():
            restored_values.setdefault(f'structure_{channel}', description or 'task-only')
        for name, value in restored_values.items():
            widget = self.widgets.get(name)
            if isinstance(widget, QComboBox):
                if name.startswith('structure_') and widget.findData(value) < 0:
                    widget.addItem(str(value), value)
                widget.setCurrentIndex(max(0, widget.findData(value)))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, MultiSelectList):
                selected = set(value)
                for index in range(widget.count()):
                    item = widget.item(index)
                    item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in selected else Qt.CheckState.Unchecked)
        self.refresh()

def main() -> int:
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(ICON)))
    window = Window()
    window.show()
    return app.exec()
if __name__ == '__main__':
    raise SystemExit(main())

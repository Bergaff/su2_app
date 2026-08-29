"""
diagnose_license.py — диагностика лицензии AeroOpt.

Положить файл в корень проекта AeroOpt (рядом с main.py / папкой
license_client) и запустить:

    python diagnose_license.py

Показывает: версию API лицензий, соединение с сервером и подпись ответов,
наличие/активацию ключа, лимит машин, срок, грейс, оффлайн-кэш,
токен расчёта. Затем открывает главное окно (с уже выполненным bootstrap).
"""

import sys
import os
import time
import traceback
from datetime import datetime, timezone

from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout,
                             QLabel, QTextEdit, QPushButton, QHBoxLayout)

from license_client import LicenseChecker


def fmt_unix(ts):
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


def run_diagnostics(checker: LicenseChecker) -> str:
    out = []
    out.append("Версия API лицензий: 4.1 (Cloudflare Worker + D1)")
    out.append(f"Сервер: {checker._server_url}")
    out.append(f"Кэш:   {checker.STATE_FILE}")
    out.append(f"HWID этой машины: {checker.hwid}")
    out.append("")

    # 1) сеть + подпись
    out.append("1) Проверка соединения с сервером...")
    resp = checker._api("/v1/activate", {
        "license_key": "__probe__",
        "hwid": checker.hwid,
        "app_version": checker._app_version,
    })
    if resp is None:
        out.append("   НЕТ СЕТИ / сервер недоступен. Работа из оффлайн-кэша.")
    elif not resp.get("ok"):
        # invalid_key у несуществующего ключа означает: сервер жив.
        sig_ok = checker._verify_response_signature(resp)
        out.append(f"   Сервер отвечает (probe code = {resp.get('code')}).")
        out.append(f"   Подпись ответа сервера: {'ВЕРНА (HMAC совпадает)' if sig_ok else '!!! НЕ СОВПАЛА — разные LICENSE_HMAC_KEY у клиента и сервера !!!'}")
    else:
        out.append("   Сервер ответил ok на probe (неожиданно, но связь есть).")
    out.append("")

    # 2) bootstrap
    out.append("2) Bootstrap (авто-проверка по кэшу/сети)...")
    try:
        st = checker.bootstrap()
        out.append(f"   status: {st}")
        out.append(f"   Текст:  {checker.get_status_text()}")
    except Exception:
        out.append("   ОШИБКА bootstrap:")
        out.append(traceback.format_exc())
    out.append("")

    # 3) локальный кэш
    out.append("3) Локальный кэш:")
    if checker.license_key:
        out.append(f"   license_key: {checker.license_key}")
        out.append(f"   product:     {checker._product}")
        out.append(f"   features:    {checker._features}")
        out.append(f"   machines:    {checker._hwid_count} из {checker._hwid_max}")
        out.append(f"   expires_at:  {fmt_unix(checker._expires_at)}")
        out.append(f"   grace_until: {fmt_unix(checker._grace_until)}")
        out.append(f"   offline до:  {fmt_unix(checker._offline_until)}")
        days = (time.time() - checker._last_check_ts) / 86400 if checker._last_check_ts else 9999
        out.append(f"   Последняя успешная связь: {days:.1f} дн. назад "
                   f"(онлайн-проверка раз в {checker.CHECK_INTERVAL // 86400} дн.; "
                   f"грейс после конца: {checker.GRACE_PERIOD // 86400} дн.; "
                   f"без сети до {checker.OFFLINE_GRACE // 86400} дн.)")
        out.append(f"   run-токен в кэше: {'есть' if checker._token else 'нет'}")
    else:
        out.append("   пусто — ключ ещё ни разу не активирован на этой машине.")
    out.append("")

    # 4) гейты
    out.append("4) Гейты:")
    allowed, why = checker.is_calculation_allowed()
    out.append(f"   is_calculation_allowed(): {allowed} ({why})")
    rt_ok, rt_msg = checker.acquire_run_token()
    out.append(f"   acquire_run_token():      {rt_ok} ({rt_msg})")
    out.append("")

    # 5) где лежит кэш
    out.append("5) Полный статус для меню/диалога:")
    for line in checker.get_status_text().splitlines():
        out.append(f"   {line}")
    return "\n".join(out)


class DiagDialog(QDialog):
    def __init__(self, checker: LicenseChecker, parent=None):
        super().__init__(parent)
        self.checker = checker
        self.setWindowTitle("Диагностика лицензии AeroOpt")
        self.setMinimumWidth(720)
        self.setMinimumHeight(580)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Диагностика лицензии AeroOpt 4.1</h3>"))

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFontFamily("Consolas, monospace")
        try:
            self.text.setText(run_diagnostics(checker))
        except Exception:
            self.text.setText("Ошибка диагностики:\n" + traceback.format_exc())
        layout.addWidget(self.text, 1)

        buttons = QHBoxLayout()
        btn_copy = QPushButton("Скопировать отчёт")
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        btn_ok = QPushButton("Продолжить в AeroOpt")
        btn_ok.clicked.connect(self.accept)
        buttons.addWidget(btn_copy)
        buttons.addStretch(1)
        buttons.addWidget(btn_ok)
        layout.addLayout(buttons)


def main():
    app = QApplication(sys.argv)
    checker = LicenseChecker()
    try:
        checker.bootstrap()
    except Exception:
        pass
    DiagDialog(checker).exec()

    # Запуск главного окна — как в вашем main.py (способ передачи checker
    # оставьте ваш; MainWindow создаёт LicenseChecker сам, если не передать):
    from ui.main_window import MainWindow
    try:
        win = MainWindow(license_checker=checker)
    except TypeError:
        win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
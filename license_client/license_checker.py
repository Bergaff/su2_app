"""
license_client/license_checker.py

Клиент лицензирования AeroOpt (НОВАЯ ВЕРСИЯ — Cloudflare Worker + D1).

Как это работает теперь:
  1. При активации приложение шлёт POST https://<license-server>/v1/activate
     с {license_key, hwid, hostname, os, app_version}.
  2. Сервер (Cloudflare Worker) проверяет ключ в базе D1:
     существует, не отозван, не истёк, есть свободный слот под HWID.
     При первой активации HWID записывается в БД (привязка машины).
  3. Ответ сервера подписан HMAC-SHA256 по КАНОНИЧЕСКОМУ JSON
     (ключи отсортированы, без пробелов). Подпись проверяется встроенным
     ключом — подделать ответ нельзя без секрета.
  4. Раз в ~30 дней — POST /v1/heartbeat (обновляет токен и статус).
  5. Без сети работает локальный кэш (XOR-шифрование на HWID+секрете):
     офлайн-режим до 60 дней, после истечения лицензии — grace 7 дней.

Никакого licenses.json в открытом Git больше нет: ключи генерируются
в админке на сайте и лежат только в базе Cloudflare D1.

Публичный API класса не менялся (bootstrap/activate/heartbeat/
deactivate/is_calculation_allowed/acquire_run_token/get_status_text/
get_activation_info) — остальной код приложения (ui/main_window.py)
работает без правок.
"""

from __future__ import annotations

import sys as _sys
# На Windows cp1251 не может вывести Unicode — принудительно UTF-8.
# В PyInstaller --windowed stdout может быть None.
if _sys.platform == "win32" and hasattr(_sys.stdout, 'reconfigure'):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if _sys.platform == "win32" and hasattr(_sys.stderr, 'reconfigure'):
    try:
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import hashlib
import hmac as hmac_mod
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
from urllib import request as _urlrequest
from urllib.error import URLError, HTTPError

# Логгер пишем в общий лог AeroOpt (app_logging.setup()), а если его не
# вызывали (standalone-запуск) — логи просто пойдут в стандартный logging.
try:
    from app_logging import get_logger as _get_logger
    log = _get_logger("license")
except Exception:  # pragma: no cover
    log = logging.getLogger("aeroopt.license")


# =========================================================================
# Статус лицензии
# =========================================================================

class LicenseStatus(Enum):
    ACTIVE = "active"
    GRACE = "grace"               # лицензия истекла < 7 дней назад
    EXPIRED = "expired"
    INVALID = "invalid"           # ключ не найден / подпись не сошлась
    NO_KEY = "no_key"
    NETWORK_ERROR = "network_error"
    # Алиасы для совместимости со старым UI (маппятся на базовые статусы):
    # trial — это активная лицензия с expires_at; revoked/hwid_limit —
    # частные случаи INVALID, но с отдельными текстами в get_status_text.
    TRIAL = "active"
    REVOKED = "invalid"
    HWID_LIMIT = "invalid"


# Поля, которые подписывает сервер. БЕЛЫЙ СПИСОК + сортировка ключей —
# ровно как в license_server/src/worker.js (SIGNED_FIELDS).
_SIGNED_FIELDS = (
    "status", "license_key", "hwid", "product", "plan",
    "expires_at", "grace_until", "offline_until",
    "hwid_count", "hwid_max", "features", "token", "server_ts",
)


# =========================================================================
# Канонический JSON (идентичен JS-функции canonical() в воркере)
# =========================================================================

def _canonical(v) -> str:
    if isinstance(v, dict):
        return "{" + ",".join(
            json.dumps(k, ensure_ascii=True) + ":" + _canonical(v[k])
            for k in sorted(v.keys())
        ) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_canonical(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=True)


class LicenseChecker:
    """Клиент проверки лицензий AeroOpt через Cloudflare Worker + D1.

    Параметры конструктора (все необязательные):
        server_url  — базовый URL воркера (по умолчанию зашитый).
                      Можно переопределить переменной окружения
                      AEROOPT_LICENSE_SERVER.
        hmac_key    — HMAC-секрет (bytes/str); по умолчанию встроенный.
                      Можно переопределить AEROOPT_LICENSE_HMAC_KEY.
        app_version — версия приложения (для User-Agent и статистики).
    """

    # --- URL по умолчанию ------------------------------------------------
    # Воркер Cloudflare. Можно заменить на https://api.aeroopt.app после
    # настройки кастомного домена (см. license_server/DEPLOY_STEPS.md).
    DEFAULT_URL = "https://aeroopt-license-server.tgmg.workers.dev"

    # --- Периоды (секунды) -----------------------------------------------
    CHECK_INTERVAL = 30 * 24 * 3600     # онлайн-проверка раз в 30 дней
    GRACE_PERIOD   =  7 * 24 * 3600     # 7 дней после окончания лицензии
    OFFLINE_GRACE  = 60 * 24 * 3600     # до 60 дней без сети
    HTTP_TIMEOUT   = 20

    # --- Локальное хранилище ---------------------------------------------
    CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".aeroopt")
    STATE_FILE = os.path.join(CACHE_DIR, "license.dat")

    # ---------------------------------------------------------------------
    def __init__(self, server_url: Optional[str] = None,
                 hmac_key=None, app_version: str = "4.1.0"):
        env_url = os.environ.get("AEROOPT_LICENSE_SERVER", "").strip()
        self._server_url = (server_url or env_url or self.DEFAULT_URL).rstrip("/")
        self._app_version = app_version

        if hmac_key is not None:
            self._hmac_key = (
                hmac_key.encode("utf-8") if isinstance(hmac_key, str)
                else bytes(hmac_key)
            )
        else:
            env_key = os.environ.get("AEROOPT_LICENSE_HMAC_KEY", "").strip()
            self._hmac_key = env_key.encode("utf-8") if env_key \
                else self._decode_hmac_key()

        # Состояние
        self.license_key: str = ""
        try:
            self.hwid: str = self._compute_hwid()
        except Exception:
            log.exception("Не удалось вычислить HWID этой машины")
            # запасной стабильный идентификатор, чтобы не падать
            self.hwid = hashlib.sha256(
                f"fallback-{platform.node()}".encode("utf-8")).hexdigest()
        log.debug("LicenseChecker init: server=%s app=%s hwid=%s…",
                  self._server_url, self._app_version, self.hwid[:8])
        self._status = LicenseStatus.NO_KEY
        self._features: list = []
        self._product: str = ""
        self._expires_at: Optional[int] = None     # unix ts
        self._grace_until: Optional[int] = None
        self._offline_until: Optional[int] = None
        self._hwid_count: int = 0
        self._hwid_max: int = 2
        self._token: str = ""
        self._last_check_ts: float = 0.0

        try:
            self._load_state()
            log.info("Локальный кэш лицензии: ключ=%s статус=%s",
                     self.license_key or "нет", self._status.value)
        except Exception:
            log.exception("Ошибка чтения локального кэша лицензии")

    # =====================================================================
    # ОБФУСЦИРОВАННЫЙ HMAC-КЛЮЧ (XOR 0x5A, hex)
    # =====================================================================
    # ВНИМАНИЕ: это СИММЕТРИЧНЫЙ секрет. Он защищает ответ сервера от
    # подделки в MITM-сценарии и от «локального сервера пиратов», но
    # извлекается из бинаря при большом желании (как и любой вшитый ключ).
    # Должен совпадать с wrangler secret LICENSE_HMAC_KEY на сервере.
    # Ротация: сгенерировать новый (openssl rand -hex 32), положить в
    # секрет воркера, пересобрать приложение с этой строкой.
    @staticmethod
    def _decode_hmac_key() -> bytes:
        _enc = (
            "6d3e6d3b6b623b6c6c69683f693f3c6c3c6a3b6369693f626938386e386f396e"
            "626a63683869636d6b623f3b6b6a6c6f6f6e3f6e38626d6a3e6a6e3c683f3b3c"
        )
        _xor = 0x5A
        try:
            raw = bytes.fromhex(_enc)
            return bytes(b ^ _xor for b in raw)
        except Exception:
            return b""

    # =====================================================================
    # HWID — УНИКАЛЬНЫЙ ИДЕНТИФИКАТОР МАШИНЫ
    # =====================================================================
    def _compute_hwid(self) -> str:
        """Стабильный хеш оборудования: hostname + MAC + UUID/BIOS/диск."""
        parts: list = []
        parts.append(platform.node())
        try:
            parts.append(str(uuid.getnode()))
        except Exception:
            parts.append("0")
        parts.append(platform.processor() or "unknown")

        if sys.platform == "win32":
            _cmds = [
                (["wmic", "csproduct", "get", "UUID"], "UUID"),
                (["wmic", "bios", "get", "SerialNumber"], "SerialNumber"),
                (["wmic", "diskdrive", "get", "SerialNumber"], "SerialNumber"),
            ]
            for cmd, header in _cmds:
                try:
                    r = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=5,
                        creationflags=0x08000000,  # CREATE_NO_WINDOW
                    )
                    if r.returncode == 0:
                        for line in r.stdout.strip().split("\n"):
                            val = line.strip()
                            if val and val != header:
                                parts.append(val)
                                break
                except Exception:
                    pass
        elif sys.platform == "linux":
            for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                try:
                    with open(p, "r") as f:
                        parts.append(f.read().strip())
                        break
                except Exception:
                    pass
        elif sys.platform == "darwin":
            try:
                r = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    for line in r.stdout.split("\n"):
                        if "IOPlatformUUID" in line:
                            parts.append(line.split('"')[-2])
                            break
            except Exception:
                pass

        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    # =====================================================================
    # HTTP
    # =====================================================================
    def _api(self, path: str, payload: dict) -> Optional[dict]:
        """POST JSON на воркер. Возвращает разобранный ответ или None
        при сетевой ошибке/таймауте."""
        url = self._server_url + path
        body = json.dumps(payload).encode("utf-8")
        req = _urlrequest.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"AeroOpt/{self._app_version}",
                "Accept": "application/json",
                "X-AeroOpt-Client": f"python/{self._app_version}",
            },
        )
        try:
            with _urlrequest.urlopen(req, timeout=self.HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict):
                    return data
        except HTTPError as e:
            # Тело ошибки тоже JSON ({ok:false, code, message})
            try:
                raw = e.read().decode("utf-8", errors="replace")
                log.warning("HTTP %s от %s: %s", e.code, url, raw[:200])
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        except (URLError, socket.timeout, TimeoutError, OSError) as e:
            log.warning("Сеть недоступна (%s): %r", url, e)
        except Exception as e:
            log.exception("Ошибка запроса %s", url)
        return None

    # =====================================================================
    # ПРОВЕРКА ПОДПИСИ ОТВЕТА
    # =====================================================================
    def _verify_response_signature(self, data: dict) -> bool:
        """Проверяет HMAC подпись ответа сервера по каноническому JSON."""
        sig = data.get("signature", "")
        if not sig:
            return False
        payload = {"server_ts": data.get("server_ts")}
        for f in _SIGNED_FIELDS:
            if f == "server_ts":
                continue
            if f in data:
                payload[f] = data[f]
        expected = hmac_mod.new(
            self._hmac_key,
            _canonical(payload).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac_mod.compare_digest(str(sig).lower(), expected.lower())

    def _apply_signed_state(self, data: dict) -> None:
        """Раскладывает подписанный ответ сервера по полям и кэширует."""
        self.license_key = str(data.get("license_key", "")).strip().upper()
        self._product = data.get("product") or data.get("plan") or ""
        self._features = list(data.get("features") or [])
        self._expires_at = data.get("expires_at")
        self._grace_until = data.get("grace_until")
        self._offline_until = data.get("offline_until")
        self._hwid_count = int(data.get("hwid_count") or 0)
        self._hwid_max = int(data.get("hwid_max") or 2)
        self._token = data.get("token") or ""
        self._last_check_ts = time.time()

        status = data.get("status", "active")
        if status == "grace":
            self._status = LicenseStatus.GRACE
        elif status == "expired":
            self._status = LicenseStatus.EXPIRED
        else:
            self._status = LicenseStatus.ACTIVE
        self._save_state()

    def _online_call(self, path: str, extra: Optional[dict] = None) -> dict:
        """Вызов сервера с проверкой подписи и HWID.
        Возвращает {ok, data?, code?, message?, network:bool}."""
        payload = {
            "license_key": self.license_key,
            "hwid": self.hwid,
            "hostname": platform.node(),
            "os": f"{sys.platform}/{platform.version()}",
            "app_version": self._app_version,
        }
        if extra:
            payload.update(extra)
        log.debug("→ %s key=%s hwid=%s… extra=%s", path, self.license_key,
                  (self.hwid or "")[:8], bool(extra))
        data = self._api(path, payload)
        if data is None:
            log.warning("← %s: нет ответа/сеть недоступна", path)
            return {"ok": False, "network": True, "code": "network_error"}
        if not data.get("ok"):
            log.warning("← %s ошибка: code=%s message=%s", path,
                        data.get("code"), data.get("message"))
            return {"ok": False, "network": False,
                    "code": data.get("code", "error"),
                    "message": data.get("message", "")}
        if str(data.get("hwid", "")) != self.hwid:
            log.error("← %s hwid_mismatch (ответ для другой машины)", path)
            return {"ok": False, "network": False,
                    "code": "hwid_mismatch",
                    "message": "Ответ сервера относится к другой машине"}
        if not self._verify_response_signature(data):
            log.error("← %s bad_signature: подпись не совпала с встроенным HMAC", path)
            return {"ok": False, "network": False,
                    "code": "bad_signature",
                    "message": "Подпись ответа сервера неверна"}
        log.debug("← %s ok: status=%s plan=%s machines=%s/%s",
                  path, data.get("status"), data.get("plan"),
                  data.get("hwid_count"), data.get("hwid_max"))
        return {"ok": True, "network": False, "data": data}

    # =====================================================================
    # ОФЛАЙН-РЕЖИМ (кэш)
    # =====================================================================
    def _offline_status(self) -> LicenseStatus:
        """Статус по локальному кэшу, когда сервер недоступен."""
        now = time.time()
        # Кэш протух (60 дней без контакта)
        if now - self._last_check_ts > self.OFFLINE_GRACE:
            self._status = LicenseStatus.NETWORK_ERROR
            return self._status

        if self._status == LicenseStatus.ACTIVE:
            return LicenseStatus.ACTIVE
        if self._status == LicenseStatus.GRACE:
            # Сервер уже сказал «grace» — доверяем в пределах офлайн-грейса
            return LicenseStatus.GRACE
        return LicenseStatus.NETWORK_ERROR

    def _map_server_error(self, code: str, message: str = "") -> Tuple[LicenseStatus, str]:
        """Код ошибки сервера → (статус, человекочитаемое сообщение)."""
        if code in ("invalid_key", "not_found"):
            return LicenseStatus.INVALID, "Ключ не найден. Проверьте правильность ключа."
        if code == "revoked":
            return LicenseStatus.REVOKED, "Лицензия отозвана. Обратитесь на support@aeroopt.app."
        if code == "expired":
            return LicenseStatus.EXPIRED, "Срок лицензии истёк. Продлите на https://aeroopt.app."
        if code == "hwid_limit":
            return (LicenseStatus.HWID_LIMIT,
                    "Достигнут лимит машин. Отвяжите старую машину в личном "
                    "кабинете https://aeroopt.app/account или в меню "
                    "«Лицензия → Отвязать эту машину» на старом ПК.")
        if code in ("not_bound", "deactivated"):
            return LicenseStatus.INVALID, "Эта машина не привязана к ключу (или привязка снята)."
        if code in ("bad_signature", "hwid_mismatch"):
            return LicenseStatus.INVALID, "Ошибка проверки ответа сервера лицензий."
        if code == "bad_token":
            return LicenseStatus.INVALID, "Сессия устарела. Выполните активацию ключа заново."
        return LicenseStatus.INVALID, message or "Ошибка сервера лицензий."

    # =====================================================================
    # ЛОКАЛЬНОЕ ХРАНИЛИЩЕ (XOR-шифрование на HWID+секрете)
    # =====================================================================
    def _make_xor_key(self) -> bytes:
        hmac_part = self._hmac_key.decode("ascii", errors="ignore")
        combined = (self.hwid + hmac_part)[:64]
        return combined.encode("utf-8")

    @staticmethod
    def _xor_bytes(data: bytes, key: bytes) -> bytes:
        if not key:
            return data
        kl = len(key)
        return bytes(d ^ key[i % kl] for i, d in enumerate(data))

    def _cache_checksum(self) -> str:
        parts = [self.license_key, self._product, str(self._expires_at),
                 self.hwid, str(self._last_check_ts), self._token]
        raw = "|".join(parts)
        return hashlib.sha256(
            (raw + self._hmac_key.decode("ascii", errors="ignore")).encode()
        ).hexdigest()[:16]

    def _save_state(self):
        state = {
            "k":  self.license_key,
            "ts": self._last_check_ts,
            "st": self._status.value,
            "ft": self._features,
            "pr": self._product,
            "ex": self._expires_at,
            "gu": self._grace_until,
            "ou": self._offline_until,
            "hc": self._hwid_count,
            "hm": self._hwid_max,
            "tk": self._token,
            "hw": self.hwid,
            "ck": self._cache_checksum(),
        }
        raw = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        encrypted = self._xor_bytes(raw.encode("utf-8"), self._make_xor_key())
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            with open(self.STATE_FILE, "wb") as f:
                f.write(encrypted)
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetFileAttributesW(self.STATE_FILE, 0x02)
                except Exception:
                    pass
        except Exception as e:
            log.exception("Не удалось сохранить кэш лицензии")

    def _load_state(self):
        if not os.path.exists(self.STATE_FILE):
            return
        try:
            with open(self.STATE_FILE, "rb") as f:
                encrypted = f.read()
            raw = self._xor_bytes(encrypted, self._make_xor_key())
            state = json.loads(raw.decode("utf-8"))

            if state.get("hw") != self.hwid:
                self._reset_state()
                return

            self.license_key = state.get("k", "")
            self._last_check_ts = state.get("ts", 0)
            self._features = state.get("ft", [])
            self._product = state.get("pr", "")
            self._expires_at = state.get("ex")
            self._grace_until = state.get("gu")
            self._offline_until = state.get("ou")
            self._hwid_count = state.get("hc", 0)
            self._hwid_max = state.get("hm", 2)
            self._token = state.get("tk", "")
            try:
                self._status = LicenseStatus(state.get("st", "no_key"))
            except ValueError:
                self._status = LicenseStatus.NO_KEY
        except Exception:
            self._reset_state()

    def _reset_state(self):
        self.license_key = ""
        self._last_check_ts = 0.0
        self._status = LicenseStatus.NO_KEY
        self._features = []
        self._product = ""
        self._expires_at = None
        self._grace_until = None
        self._offline_until = None
        self._hwid_count = 0
        self._hwid_max = 2
        self._token = ""
        try:
            if os.path.exists(self.STATE_FILE):
                os.remove(self.STATE_FILE)
        except Exception:
            pass

    # =====================================================================
    # ПУБЛИЧНЫЙ API
    # =====================================================================

    # --- Совместимость со старым UI (вызывает методы прежнего клиента) ---
    def _fetch_licenses(self, *args, **kwargs):
        """Старый клиент тянул публичный список ключей из licenses.json.
        Теперь ключей в открытом виде нет — это no-op, возвращает True,
        чтобы диалог активации в старом UI открывался без ошибки."""
        log.debug("_fetch_licenses() — заглушка (ключи теперь в D1), no-op")
        return True

    def get_features(self):
        return list(self._features)

    def has_feature(self, name: str) -> bool:
        return name in self._features

    def is_active(self) -> bool:
        return self._status == LicenseStatus.ACTIVE

    def get_plan(self) -> str:
        return self._product

    def bootstrap(self) -> LicenseStatus:
        """Проверка при запуске приложения.

        Свежий кэш (< 30 дней, статус рабочий) → кэш без сети.
        Иначе — онлайн heartbeat; нет сети → офлайн-грейс по кэшу.
        """
        if not self.license_key:
            self._status = LicenseStatus.NO_KEY
            return self._status

        elapsed = time.time() - self._last_check_ts
        if elapsed < self.CHECK_INTERVAL and self._status in (
                LicenseStatus.ACTIVE, LicenseStatus.GRACE):
            return self._status

        result = self._online_call("/v1/heartbeat")
        if result["ok"]:
            self._apply_signed_state(result["data"])
            return self._status

        if result.get("network"):
            return self._offline_status()

        # Сервер ответил ошибкой
        status, _msg = self._map_server_error(
            result.get("code", ""), result.get("message", ""))
        # not_bound/deactivated — могла быть рассинхронизация: даём
        # офлайн-грейс, если кэш был активен.
        if result.get("code") in ("not_bound", "deactivated", "bad_token"):
            if self._status in (LicenseStatus.ACTIVE, LicenseStatus.GRACE):
                return self._offline_status()
        self._status = status
        if status == LicenseStatus.EXPIRED and self._last_check_ts:
            # сервер говорит «истекла» — проверим grace по дате
            if self._grace_until and time.time() < self._grace_until:
                self._status = LicenseStatus.GRACE
        self._save_state()
        return self._status

    def activate(self, key: str) -> Tuple[bool, str]:
        """Активирует лицензионный ключ: POST /v1/activate (привязка HWID
        на сервере). Возвращает (ok, message)."""
        try:
            self.license_key = (key or "").strip().upper()
            log.info("activate() ключ=%s server=%s hwid=%s…",
                     self.license_key, self._server_url, (self.hwid or "")[:8])

            result = self._online_call("/v1/activate")
            if result["ok"]:
                self._apply_signed_state(result["data"])
                log.info("Активация успешна: %s", self.get_status_text())
                return True, f"Лицензия активирована: {self.get_status_text()}"

            if result.get("network"):
                # Активация без сети невозможна (HWID нужно записать в БД)
                log.warning("Активация без сети невозможна")
                self.license_key = ""
                return False, (
                    "Нет связи с сервером лицензий. Проверьте подключение к "
                    "интернету и повторите активацию.")

            status, message = self._map_server_error(
                result.get("code", ""), result.get("message", ""))
            log.warning("Активация отклонена: code=%s → %s",
                        result.get("code"), message)
            self._reset_state()
            self._status = status
            return False, f"Ошибка активации: {message}"
        except Exception:
            log.exception("Исключение в activate()")
            raise

    def heartbeat(self) -> Tuple[bool, str]:
        """Принудительная онлайн-проверка (меню «Статус лицензии»)."""
        if not self.license_key:
            return False, "Нет активной лицензии"
        result = self._online_call("/v1/heartbeat",
                                   {"token": self._token} if self._token else None)
        if result["ok"]:
            self._apply_signed_state(result["data"])
            return True, self.get_status_text()
        if result.get("network"):
            offline = self._offline_status()
            if offline in (LicenseStatus.ACTIVE, LicenseStatus.GRACE):
                return True, f"{self.get_status_text()} (офлайн-режим)"
            return False, "Нет связи с сервером и локальный кэш истёк."
        status, message = self._map_server_error(
            result.get("code", ""), result.get("message", ""))
        if status == LicenseStatus.EXPIRED:
            self._status = LicenseStatus.EXPIRED
            self._save_state()
        elif status == LicenseStatus.INVALID:
            # не торопимся стирать кэш при разовой ошибке — скажем как есть
            pass
        return False, message

    def deactivate(self) -> Tuple[bool, str]:
        """Отвязывает эту машину: снимает привязку на сервере (если есть
        сеть) и очищает локальный кэш."""
        if self.license_key:
            result = self._online_call("/v1/deactivate",
                                       {"token": self._token} if self._token else None)
            # Даже если сети нет — локально отвязываемся обязательно:
            # пользователь должен иметь возможность убрать лицензию с ПК.
            if result["ok"]:
                log.info("Сервер подтвердил отвязку HWID")
            elif result.get("network"):
                log.warning("Сеть недоступна — снимаем только локальную привязку")
        self._reset_state()
        return True, "Лицензия отвязана от этой машины"

    def acquire_run_token(self) -> Tuple[bool, str]:
        """Разрешение на запуск расчёта.

        Онлайн запрашивает короткоживущий run_token у сервера;
        при отсутствии сети разрешает работу по валидному кэшу
        (офлайн-грейс обрабатывается в is_calculation_allowed)."""
        allowed, reason = self.is_calculation_allowed()
        if not allowed:
            return False, reason
        if not self.license_key or not self._token:
            return True, "OK (offline cache)"
        result = self._online_call("/v1/run_token", {"token": self._token})
        if result["ok"]:
            return True, "OK"
        # Сеть/сервер недоступны, но статус рабочий — не блокируем расчёт
        status_now = self.bootstrap()
        if status_now in (LicenseStatus.ACTIVE, LicenseStatus.GRACE):
            return True, "OK (offline)"
        return False, "Не удалось подтвердить лицензию для запуска расчёта."

    def is_calculation_allowed(self) -> Tuple[bool, str]:
        """(allowed, reason)."""
        if self._status == LicenseStatus.ACTIVE:
            return True, ""
        if self._status == LicenseStatus.GRACE:
            return True, "Grace-период — продлите лицензию"
        if self._status == LicenseStatus.NO_KEY:
            return False, (
                "Лицензия не активирована.\n"
                "Откройте меню «Лицензия → Активировать ключ».")
        if self._status == LicenseStatus.EXPIRED:
            return False, "Срок лицензии истёк. Продлите на https://aeroopt.app."
        if self._status == LicenseStatus.INVALID:
            return False, (
                "Лицензия недействительна, отозвана или привязана к другой "
                "машине.\nПроверьте ключ или обратитесь на support@aeroopt.app.")
        if self._status == LicenseStatus.NETWORK_ERROR:
            return False, (
                "Нет связи с сервером лицензий и локальный кэш истёк.\n"
                "Подключитесь к интернету и повторите попытку.")
        return False, "Неизвестная ошибка лицензии."

    def get_status_text(self) -> str:
        texts = {
            LicenseStatus.ACTIVE:        "✅ Лицензия активна",
            LicenseStatus.GRACE:         "⚠️ Grace-период (продлите лицензию)",
            LicenseStatus.EXPIRED:       "❌ Лицензия истекла",
            LicenseStatus.INVALID:       "❌ Недействительная лицензия",
            LicenseStatus.NO_KEY:        "🔑 Лицензия не активирована",
            LicenseStatus.NETWORK_ERROR: "🌐 Нет связи с сервером лицензий",
        }
        result = texts.get(self._status, "Неизвестный статус")
        if self._status == LicenseStatus.REVOKED:
            result = "❌ Лицензия отозвана"
        elif self._status == LicenseStatus.HWID_LIMIT:
            result = "⚠️ Достигнут лимит машин"
        if self._expires_at and self._status in (
                LicenseStatus.ACTIVE, LicenseStatus.GRACE):
            try:
                d = datetime.utcfromtimestamp(int(self._expires_at)).strftime("%Y-%m-%d")
                result += f" (до {d})"
            except Exception:
                pass
        if self._product:
            result += f" · {self._product}"
        return result

    def get_activation_info(self) -> dict:
        return {
            "status_text":    self.get_status_text(),
            "license_key":    self.license_key,
            "product":        self._product,
            "features":       list(self._features),
            "hwid":           self.hwid,
            "hwid_count":     self._hwid_count,
            "hwid_max":       self._hwid_max,
            "expires_at":     self._expires_at,
            "last_heartbeat": self._last_check_ts,
            "server_url":     self._server_url,
        }
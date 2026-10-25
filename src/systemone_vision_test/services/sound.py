"""正誤判定の効果音。

**1 回の再生ごとに、新しい再生パイプライン（``QMediaPlayer`` + ``QAudioOutput``）を
作って使い終わったら破棄する。** 前回の再生で使ったストリームやデコード済みバッファを
一切再利用しないため、前の音が次の音に混じることが原理的に起こらない。

* 素材はアプリ全体で共通（:meth:`SoundEffects.shared`）
* 同時に鳴るのは常に 1 つ（新しい音を鳴らす前に、前の再生を止めて破棄する）
* 再生が終わったら速やかに破棄する（次の判定のときには前の再生が残っていない）
* 音声ファイルが無い／再生できない場合でも例外にせず ``False`` を返す
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, MutableMapping
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import QLibraryInfo, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from systemone_vision_test import config

logger = logging.getLogger(__name__)

#: 音声ファイルとして受け付ける拡張子（先に見つかったものを使う）
SUPPORTED_SUFFIXES: tuple[str, ...] = (".mp3", ".wav", ".ogg")
#: 再生時の音量
DEFAULT_VOLUME = 1.0

#: Qt のマルチメディアバックエンドを選ぶ環境変数
BACKEND_ENV_VAR = "QT_MEDIA_BACKEND"
#: Qt のログ出力を制御する環境変数
LOGGING_RULES_ENV_VAR = "QT_LOGGING_RULES"
#: Windows 用（Media Foundation）バックエンドの名前
WINDOWS_BACKEND = "windows"
#: Windows 用バックエンドのプラグイン名
WINDOWS_BACKEND_PLUGIN = "windowsmediaplugin.dll"
#: FFmpeg バックエンドがファイルを開くたびに出すログを抑えるルール
FFMPEG_LOGGING_RULE = "qt.multimedia.ffmpeg.*=false"


@lru_cache(maxsize=1)
def windows_backend_available() -> bool:
    """Windows 用（Media Foundation）バックエンドのプラグインがあるかどうか。

    Returns:
        プラグイン（``windowsmediaplugin.dll``）が見つかれば ``True``。
    """
    plugins_dir = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    return (plugins_dir / "multimedia" / WINDOWS_BACKEND_PLUGIN).is_file()


def configure_audio_backend(
    environ: MutableMapping[str, str] | None = None, platform: str | None = None
) -> str:
    """音声再生に使うバックエンドを選ぶ（**QtMultimedia を使う前に呼ぶ**）。

    Qt の既定は FFmpeg バックエンドで、音声ファイルを開くたびに FFmpeg のログ
    （``Input #0 ...`` や MFT 探索のメッセージ）を標準エラーへ出し、読み込みにも
    時間がかかる。Windows 用の Media Foundation バックエンド（``windows``）は
    ログを出さずに読み込めるため、Windows ではそちらを既定にする。

    * ``QT_MEDIA_BACKEND`` が既に設定されている場合は、その指定を尊重する
    * Windows 以外、または Windows 用プラグインが無い場合は FFmpeg のログだけを抑える
    * 環境変数は QtMultimedia の初期化時に読まれるため、Qt のオブジェクトを
      作る前に呼ぶこと（:func:`systemone_vision_test.gui.app.create_application` で呼んでいる）

    Args:
        environ: 設定先（既定は ``os.environ``。テスト用に差し替え可能）。
        platform: 判定に使うプラットフォーム名（既定は ``sys.platform``）。

    Returns:
        何を設定したかの説明（ログ表示用）。
    """
    target = os.environ if environ is None else environ
    current_platform = sys.platform if platform is None else platform

    configured = target.get(BACKEND_ENV_VAR)
    if configured:
        return f"{BACKEND_ENV_VAR}={configured} の指定を使用します"  # ユーザーの指定を尊重

    if current_platform.startswith("win") and windows_backend_available():
        target[BACKEND_ENV_VAR] = WINDOWS_BACKEND
        return f"{BACKEND_ENV_VAR}={WINDOWS_BACKEND} を使用します"

    rules = target.get(LOGGING_RULES_ENV_VAR)
    target[LOGGING_RULES_ENV_VAR] = (
        f"{rules};{FFMPEG_LOGGING_RULE}" if rules else FFMPEG_LOGGING_RULE
    )
    return f"{WINDOWS_BACKEND} 用プラグインが無いため {FFMPEG_LOGGING_RULE} でログを抑えます"


class SoundEffect(str, Enum):
    """効果音の種類。値はファイル名（拡張子を除く）。"""

    CORRECT = "right"
    WRONG = "wrong"


class SoundPlayback:
    """1 回分の効果音再生（``QMediaPlayer`` + ``QAudioOutput`` を 1 組持つ）。

    再生が終わったら :meth:`stop` で資源を破棄し、次に鳴らすときは新しい組を作る。
    前回の再生のストリームやバッファを再利用しない。

    Args:
        on_finished: 再生が終わったときに呼ぶコールバック。
    """

    def __init__(self, on_finished: Callable[[], None] | None = None) -> None:
        configure_audio_backend()  # 最初の QtMultimedia 利用前にバックエンドを決めておく
        self._on_finished = on_finished
        self._finished = False
        self._output = QAudioOutput()
        self._output.setVolume(DEFAULT_VOLUME)
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._output)
        self._player.mediaStatusChanged.connect(self._handle_media_status)

    def start(self, path: Path) -> None:
        """音声ファイルの先頭から再生を開始する。"""
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._player.setPosition(0)
        self._player.play()

    def stop(self) -> None:
        """再生を止めて、この再生で使った資源を破棄する。"""
        self._player.stop()
        self._player.setSource(QUrl())  # ストリームを閉じる（バッファを残さない）
        self._player.deleteLater()
        self._output.deleteLater()

    @property
    def is_playing(self) -> bool:
        """再生中かどうか。"""
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def _handle_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        """再生が終わった（または失敗した）ときに通知する。"""
        finished_statuses = (
            QMediaPlayer.MediaStatus.EndOfMedia,
            QMediaPlayer.MediaStatus.InvalidMedia,
        )
        if status in finished_statuses and not self._finished:
            self._finished = True
            if self._on_finished is not None:
                self._on_finished()


class SoundEffects:
    """効果音の再生を担うクラス。

    再生のたびに :class:`SoundPlayback`（新しい再生パイプライン）を作り、
    終わったら破棄する。前回の再生のストリームやバッファを再利用しないため、
    前の音が次の音に混じらない。

    Args:
        sound_dir: 音声ファイルを置いたディレクトリ。既定は ``config.SOUND_DIR``。
        playback_factory: :class:`SoundPlayback` の代わりに使う生成関数（テスト用）。
    """

    #: アプリ全体で共有するインスタンス（遅延生成）
    _shared: ClassVar[SoundEffects | None] = None

    def __init__(
        self,
        sound_dir: Path | None = None,
        playback_factory: Callable[[Callable[[], None]], SoundPlayback] | None = None,
    ) -> None:
        self.sound_dir = Path(sound_dir) if sound_dir is not None else config.SOUND_DIR
        self._create_playback = playback_factory or SoundPlayback
        self._missing: set[SoundEffect] = set()
        #: いま鳴っている再生（常に 1 つだけ）
        self._active: SoundPlayback | None = None

    # -- 共有インスタンス ---------------------------------------------------

    @classmethod
    def shared(cls) -> SoundEffects:
        """アプリ全体で共有するインスタンスを返す（初回のみ生成）。"""
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    @classmethod
    def reset_shared(cls) -> None:
        """共有インスタンスを破棄する（主にテスト用）。"""
        if cls._shared is not None:
            cls._shared.stop_all()
        cls._shared = None

    # -- 公開 API -----------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        """効果音が鳴っているかどうか。"""
        return self._active is not None and self._active.is_playing

    @property
    def active_playback(self) -> SoundPlayback | None:
        """いま鳴っている再生（無ければ ``None``）。"""
        return self._active

    def available_effects(self) -> tuple[SoundEffect, ...]:
        """音声ファイルが見つかった効果音の種類を返す。

        見つからない種類があれば警告する（その効果音は無音になるため）。

        Returns:
            ファイルが見つかった効果音の種類。
        """
        found = tuple(name for name in SoundEffect if self._find_file(name) is not None)
        if len(found) != len(SoundEffect):
            missing = [name.value for name in SoundEffect if name not in found]
            logger.warning("効果音が見つかりません（無音になります）: %s", ", ".join(missing))
        return found

    def play(self, name: SoundEffect) -> bool:
        """効果音を鳴らす。

        鳴らす前に、いま鳴っている再生を止めて破棄する（同時に鳴るのは常に 1 つ）。
        ファイルが無い場合は何もせず ``False`` を返す。

        Args:
            name: 再生する効果音の種類。

        Returns:
            再生を開始できたら ``True``。
        """
        path = self._find_file(name)
        if path is None:
            return False

        self.stop_all()  # 前の再生は止めて破棄する（残った状態を持ち越さない）
        playback = self._create_playback(self._release)
        playback.start(path)
        self._active = playback
        return True

    def stop_all(self) -> None:
        """鳴っている効果音を止めて破棄する。

        検査の開始・終了など、音をまたがせたくない区切りでも呼ぶ。
        """
        playback, self._active = self._active, None
        if playback is not None:
            playback.stop()

    def play_correct(self) -> bool:
        """正答の効果音を鳴らす。"""
        return self.play(SoundEffect.CORRECT)

    def play_wrong(self) -> bool:
        """誤答の効果音を鳴らす。"""
        return self.play(SoundEffect.WRONG)

    # -- 内部処理 -----------------------------------------------------------

    def _find_file(self, name: SoundEffect) -> Path | None:
        """種類に対応する音声ファイルを探す。

        Args:
            name: 効果音の種類。

        Returns:
            見つかったファイル。見つからなければ ``None``。
        """
        for suffix in SUPPORTED_SUFFIXES:
            candidate = self.sound_dir / f"{name.value}{suffix}"
            if candidate.is_file():
                return candidate
        self._missing.add(name)  # 見つからないものは覚えておく（探し直さない）
        return None

    def _release(self) -> None:
        """再生が終わった再生を破棄する。

        次の判定のときには前の再生が残っていない状態にしておく。
        """
        self.stop_all()

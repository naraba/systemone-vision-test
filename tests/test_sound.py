"""効果音（services.sound）のテスト。

「再生のたびに新しい再生パイプラインを作って破棄する」構造を、
音声デバイスを使うテストと使わないテストの両方で検証する。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from systemone_vision_test import config
from systemone_vision_test.services import sound
from systemone_vision_test.services.sound import (
    SUPPORTED_SUFFIXES,
    SoundEffect,
    SoundEffects,
)

#: 再生が始まるのを待つ上限（秒）
PLAY_TIMEOUT_SEC = 3.0

#: 効果音の素材（``sound/right.*`` / ``sound/wrong.*``）がプロジェクトにあるかどうか。
#: 素材は第三者の生成物なのでリポジトリには含めない（``.gitignore`` の ``sound/*``）。
SOUND_FILES_PRESENT = all(
    any((config.SOUND_DIR / f"{name.value}{suffix}").is_file() for suffix in SUPPORTED_SUFFIXES)
    for name in SoundEffect
)
#: 実際に音を鳴らす（音声デバイスを使う）テストに付けるマーカー
requires_sounds = pytest.mark.skipif(
    not SOUND_FILES_PRESENT, reason="sound/ に効果音の素材がありません"
)


class FakePlayback:
    """``SoundPlayback`` の代替（音声不要）。

    Args:
        on_finished: 再生終了時に呼ぶコールバック。
    """

    def __init__(self, on_finished) -> None:
        self.on_finished = on_finished
        self.started_path: Path | None = None
        self.stop_calls = 0
        self.playing = False

    def start(self, path: Path) -> None:
        """再生開始を記録する。"""
        self.started_path = path
        self.playing = True

    def stop(self) -> None:
        """停止を記録する。"""
        self.stop_calls += 1
        self.playing = False

    @property
    def is_playing(self) -> bool:
        """再生中かどうか。"""
        return self.playing


def make_fake_player(sound_dir: Path) -> tuple[SoundEffects, list[FakePlayback]]:
    """``FakePlayback`` を使うプレイヤと、生成された再生の一覧を返す。

    Args:
        sound_dir: 効果音を探すディレクトリ。

    Returns:
        ``(プレイヤ, 生成された再生の一覧)``。
    """
    created: list[FakePlayback] = []

    def factory(on_finished) -> FakePlayback:
        playback = FakePlayback(on_finished)
        created.append(playback)
        return playback

    return SoundEffects(sound_dir, playback_factory=factory), created


@pytest.fixture
def sounds_dir(tmp_path: Path) -> Path:
    """効果音のファイル（中身は空でよい）を置いたディレクトリを返す。

    再生そのものは ``FakePlayback`` に差し替えるので、中身は問わない。
    素材をリポジトリに含めないため、テストはこの一時ディレクトリを使う。
    """
    for name in SoundEffect:
        (tmp_path / f"{name.value}.mp3").write_bytes(b"")
    return tmp_path


@pytest.fixture
def fake_sounds(sounds_dir: Path):
    """``FakePlayback`` を使うプレイヤ。"""
    return make_fake_player(sounds_dir)


def wait_until(predicate, process_events, timeout: float = PLAY_TIMEOUT_SEC) -> bool:
    """条件が満たされるまでイベントループを回す。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        process_events(0.05)
    return predicate()


# ---------------------------------------------------------------------------
# 環境に依存しないテスト（フェイクの再生で検証）
# ---------------------------------------------------------------------------


def test_shared_instance_is_singleton():
    """共有インスタンスは常に同一。"""
    assert SoundEffects.shared() is SoundEffects.shared()


def test_reset_shared_discards_and_recreates():
    """``reset_shared`` で作り直される（テスト用）。"""
    first = SoundEffects.shared()
    SoundEffects.reset_shared()

    assert SoundEffects.shared() is not first


def test_sound_effect_values_are_file_stems():
    """効果音の種類の値がファイル名（拡張子なし）になっている。"""
    assert SoundEffect.CORRECT.value == "right"
    assert SoundEffect.WRONG.value == "wrong"


def test_available_effects_reports_existing_files(fake_sounds):
    """``available_effects`` は存在する効果音を返す。"""
    player, _ = fake_sounds

    assert set(player.available_effects()) == set(SoundEffect)


def test_sound_directory_exists_in_project():
    """プロジェクトに効果音のディレクトリがある（設定の前提）。"""
    assert config.SOUND_DIR.is_dir()


def test_supported_suffixes_are_documented():
    """対応する拡張子が明示されている。"""
    assert ".mp3" in SUPPORTED_SUFFIXES


# ---------------------------------------------------------------------------
# 音声バックエンドの選択
# ---------------------------------------------------------------------------


def test_configure_audio_backend_uses_windows_plugin(monkeypatch):
    """Windows で Windows 用プラグインがあれば Media Foundation を使う。"""
    monkeypatch.setattr(sound, "windows_backend_available", lambda: True)
    environ: dict[str, str] = {}

    message = sound.configure_audio_backend(environ, platform="win32")

    assert environ[sound.BACKEND_ENV_VAR] == sound.WINDOWS_BACKEND
    assert sound.WINDOWS_BACKEND in message


def test_configure_audio_backend_silences_ffmpeg_logs(monkeypatch):
    """Windows 用プラグインが無い環境では FFmpeg のログだけを抑える。"""
    monkeypatch.setattr(sound, "windows_backend_available", lambda: False)
    environ: dict[str, str] = {}

    sound.configure_audio_backend(environ, platform="win32")

    assert sound.BACKEND_ENV_VAR not in environ  # バックエンドは変更しない
    assert sound.FFMPEG_LOGGING_RULE in environ[sound.LOGGING_RULES_ENV_VAR]


def test_configure_audio_backend_keeps_existing_logging_rules(monkeypatch):
    """既存のログ設定を壊さない（追記する）。"""
    monkeypatch.setattr(sound, "windows_backend_available", lambda: False)
    environ = {sound.LOGGING_RULES_ENV_VAR: "my.category.debug=true"}

    sound.configure_audio_backend(environ, platform="linux")

    rules = environ[sound.LOGGING_RULES_ENV_VAR]
    assert "my.category.debug=true" in rules
    assert sound.FFMPEG_LOGGING_RULE in rules


def test_configure_audio_backend_respects_user_setting(monkeypatch):
    """ユーザーがバックエンドを指定している場合は変更しない。"""
    monkeypatch.setattr(sound, "windows_backend_available", lambda: True)
    environ = {sound.BACKEND_ENV_VAR: "ffmpeg"}

    message = sound.configure_audio_backend(environ, platform="win32")

    assert environ[sound.BACKEND_ENV_VAR] == "ffmpeg"
    assert "ffmpeg" in message


def test_windows_backend_plugin_is_available():
    """PySide6 に Windows 用バックエンドのプラグインが同梱されている。"""
    if not sys.platform.startswith("win"):
        pytest.skip("Windows 専用の確認")

    assert sound.windows_backend_available() is True


def test_sound_playback_selects_backend_before_creating_objects(monkeypatch):
    """``SoundPlayback`` も作る前にバックエンドを設定する（単体で使う場合の保険）。"""
    calls: list[str] = []
    monkeypatch.setattr(sound, "configure_audio_backend", lambda: calls.append("called"))

    sound.SoundPlayback()

    assert calls == ["called"]


def test_play_creates_a_new_playback(fake_sounds, sounds_dir):
    """再生のたびに新しい再生パイプラインを作る。"""
    player, created = fake_sounds

    assert player.play_correct() is True

    assert len(created) == 1
    assert created[0].started_path == sounds_dir / "right.mp3"
    assert created[0].playing is True


def test_next_play_discards_the_previous_playback(fake_sounds, sounds_dir):
    """次の音を鳴らす前に、前の再生を止めて破棄する。"""
    player, created = fake_sounds
    player.play_wrong()
    previous = created[-1]
    assert previous.playing is True

    player.play_correct()

    assert previous.stop_calls == 1  # 止めて破棄されている
    assert previous.playing is False
    assert len(created) == 2  # 新しい再生が作られている
    assert created[-1].started_path == sounds_dir / "right.mp3"


def test_only_one_playback_plays_at_a_time(fake_sounds):
    """次の判定のときには、前の再生が残っていない（同時に鳴るのは 1 つだけ）。"""
    player, created = fake_sounds

    for name in (SoundEffect.WRONG, SoundEffect.CORRECT, SoundEffect.WRONG):
        player.play(name)
        playing = [p for p in created if p.playing]
        assert playing == [created[-1]]  # 鳴っているのは最後の 1 つだけ
        assert all(p.stop_calls >= 1 for p in created[:-1])  # 前のものは停止済み


def test_finished_playback_is_released(fake_sounds):
    """再生が終わったら破棄され、次は新しい再生になる。"""
    player, created = fake_sounds
    player.play_correct()
    finished = created[-1]

    finished.on_finished()  # 再生終了を通知

    assert finished.stop_calls == 1
    assert player.active_playback is None
    assert player.is_playing is False

    player.play_correct()
    assert len(created) == 2  # 新しい再生が作られる


def test_stop_all_discards_the_active_playback(fake_sounds):
    """``stop_all`` で鳴っている再生を止めて破棄する。"""
    player, created = fake_sounds
    player.play_correct()

    player.stop_all()

    assert created[0].stop_calls == 1
    assert player.active_playback is None
    assert player.is_playing is False


def test_play_returns_false_when_files_missing(tmp_path: Path):
    """音声ファイルが無い場合は何も作らず ``False`` を返す。"""
    player, created = make_fake_player(tmp_path / "not-found")

    assert player.available_effects() == ()
    assert player.play_correct() is False
    assert created == []


# ---------------------------------------------------------------------------
# 音声デバイスを使うテスト
# ---------------------------------------------------------------------------


@requires_sounds
def test_play_starts_playback():
    """実際に音を鳴らせる。"""
    sounds = SoundEffects(config.SOUND_DIR)

    assert sounds.play_correct() is True


@requires_sounds
def test_next_sound_replaces_the_previous_playback(process_events):
    """次の音を鳴らすと、前の再生は破棄されて新しい再生になる。"""
    sounds = SoundEffects(config.SOUND_DIR)
    sounds.play_wrong()
    assert wait_until(lambda: sounds.is_playing, process_events)
    previous = sounds.active_playback

    sounds.play_correct()

    assert previous is not None
    assert previous.is_playing is False  # 前の再生は止まっている
    assert sounds.active_playback is not None
    assert sounds.active_playback is not previous  # 別の再生パイプライン


@requires_sounds
def test_playback_is_released_after_finishing(process_events):
    """再生が終わったら破棄される（次に鳴らすときには残っていない）。"""
    sounds = SoundEffects(config.SOUND_DIR)
    sounds.play_wrong()  # 短い音（約 0.5 秒）
    assert wait_until(lambda: sounds.is_playing, process_events)

    assert wait_until(
        lambda: sounds.active_playback is None and not sounds.is_playing, process_events
    )


@requires_sounds
def test_stop_all_stops_playback(process_events):
    """``stop_all`` で鳴っている音が止まる。"""
    sounds = SoundEffects(config.SOUND_DIR)
    sounds.play_correct()
    assert wait_until(lambda: sounds.is_playing, process_events)

    sounds.stop_all()

    assert sounds.is_playing is False
    assert sounds.active_playback is None

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from io import BytesIO
from typing import Any, ByteString, Union

import audioread
import librosa
import numpy as np

from .config import MultimodalConfig, get_audio_sample_rate


AudioInput = Union[
    np.ndarray,
    ByteString,
    str,
]


def load_audio_from_bytes(audio_bytes: bytes, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Load audio waveform from raw bytes.

    Parameters
    - audio_bytes: Raw audio file bytes (for example WAV data).
    - config: MultimodalConfig object.

    Returns
    - 1-D numpy array with audio samples.
    """
    sample_rate = get_audio_sample_rate(config)
    with BytesIO(audio_bytes) as wav_io:
        audio, _ = librosa.load(wav_io, sr=sample_rate)
    return audio


def load_audio_from_path(audio_path: str, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Load audio from a filesystem path or HTTP(S) URL.

    Parameters
    - audio_path: Local path, `file://` path, or HTTP(S) URL to the audio file.
    - config: MultimodalConfig object.

    Returns
    - 1-D numpy array with audio samples.
    """
    sample_rate = get_audio_sample_rate(config)
    if audio_path.startswith(("http://", "https://")):
        return librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=sample_rate)[0]
    else:
        return librosa.load(audio_path, sr=sample_rate)[0]


def load_audio(audio: AudioInput, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Unified loader for different audio input types.

    Parameters
    - audio: One of:
        - `np.ndarray`: a waveform already in memory (returned unchanged).
        - `bytes`: raw audio bytes (WAV/etc.) — handled by `load_audio_from_bytes`.
        - `str`: a path or URL — handled by `load_audio_from_path`.
    - config: MultimodalConfig object

    Returns
    - 1-D numpy array with audio samples.
    """
    if isinstance(audio, np.ndarray):
        return audio
    if isinstance(audio, str):
        return load_audio_from_path(audio, config=config, **kwargs)
    elif isinstance(audio, bytes):
        return load_audio_from_bytes(audio, config=config, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported audio input type: {type(audio)}")


def fetch_audio(info: dict, config: MultimodalConfig = None, **kwargs: Any) -> np.ndarray:
    """Convenience helper to extract and load audio from an `info` mapping.

    Parameters
    - info: Mapping that must contain an `"audio"` key whose value is an
        `AudioInput` (see `load_audio`).
    - config: Optional MultimodalConfig object for audio processing parameters.

    Returns
    - 1-D numpy array with audio samples.
    """
    audio_info = info["audio"]
    audio = load_audio(audio_info, config=config, **kwargs)
    return audio

from __future__ import annotations

from types import TracebackType
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

_StreamT = TypeVar("_StreamT", bound="Stream")

DeviceSpecifier = int | str | None
DTypeLike = str | np.dtype[Any] | type[np.generic] | None
CallbackData = NDArray[np.float32]
StreamCallback = Callable[[CallbackData, int, Any, "CallbackFlags"], None]


class CallbackFlags(int):
    """Bitmask describing callback status flags."""
    ...


class Stream:
    def __init__(
        self,
        *,
        samplerate: float | int | None = ...,
        blocksize: int | None = ...,
        device: DeviceSpecifier = ...,
        dtype: DTypeLike = ...,
        **kwargs: Any,
    ) -> None: ...

    def __enter__(self: _StreamT) -> _StreamT: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...

    def start(self: _StreamT, ignore_errors: bool = ...) -> _StreamT: ...

    def stop(self: _StreamT, ignore_errors: bool = ...) -> _StreamT: ...

    def close(self) -> None: ...


class InputStream(Stream):
    def __init__(
        self,
        *,
        callback: StreamCallback | None = ...,
        channels: int | None = ...,
        samplerate: float | int | None = ...,
        blocksize: int | None = ...,
        device: DeviceSpecifier = ...,
        dtype: DTypeLike = ...,
        **kwargs: Any,
    ) -> None: ...

    def __enter__(self) -> InputStream: ...


class OutputStream(Stream):
    def __init__(
        self,
        *,
        callback: StreamCallback | None = ...,
        channels: int | None = ...,
        samplerate: float | int | None = ...,
        blocksize: int | None = ...,
        device: DeviceSpecifier = ...,
        dtype: DTypeLike = ...,
        **kwargs: Any,
    ) -> None: ...

    def write(self, data: CallbackData) -> None: ...

    def __enter__(self) -> OutputStream: ...


def play(
    data: ArrayLike,
    samplerate: float | int | None = ...,
    mapping: Sequence[int] | None = ...,
    blocking: bool = ...,
    loop: bool = ...,
    **kwargs: Any,
) -> OutputStream | None: ...


def wait(ignore_errors: bool = ...) -> None: ...


def stop(ignore_errors: bool = ...) -> None: ...

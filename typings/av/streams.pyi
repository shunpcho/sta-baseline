from collections.abc import Iterable

from .frame import Frame

class Stream:
    index: int
    type: str  # "video", "audio", etc.

    codec_context: object | None
    average_rate: float | None
    time_base: float | None

    def decode(self, packets: Iterable[object]) -> Iterable[Frame]: ...

class VideoStream(Stream):
    width: int | None
    height: int | None
    pix_fmt: str | None

    def decode(self, packets: Iterable[object]) -> Iterable[Frame]: ...

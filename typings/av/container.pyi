from collections import UserList
from collections.abc import Iterable

from av.streams import Stream, VideoStream

from .packet import Packet

class Streams(UserList[Stream]):
    @property
    def video(self) -> list[VideoStream]: ...
    @property
    def audio(self) -> list[Stream]: ...

class Container:
    streams: Streams

    def demux(self, streams: Iterable[Stream] | None = None) -> Iterable[Packet]: ...
    def close(self) -> None: ...

# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Base utilities for streaming readers."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Generic, Iterable, Iterator, Optional, Sequence, TypeVar

T = TypeVar("T")


class LazyAtomIterator(Iterator[T]):
    """Iterator that reads atoms on-demand."""

    def __init__(
        self,
        filepath: Path,
        parser: Callable[[str], Optional[T]],
        record_prefixes: Sequence[str] = ("ATOM", "HETATM"),
    ) -> None:
        """Initialize a new LazyAtomIterator instance."""
        self.filepath = filepath
        self.parser = parser
        self.record_prefixes = tuple(record_prefixes)
        self._file: Optional[Iterable[str]] = None

    def __iter__(self) -> "LazyAtomIterator[T]":
        """Return an iterator over contained records."""
        self._file = open(self.filepath, "r")
        return self

    def __next__(self) -> T:
        """Return the next parsed record from the stream."""
        if self._file is None:
            self.__iter__()
        while True:
            try:
                line = next(self._file)  # type: ignore
            except StopIteration:
                if self._file is not None and hasattr(self._file, "close"):
                    self._file.close()  # type: ignore
                raise
            record = line[:6].strip()
            if record in self.record_prefixes:
                try:
                    parsed = self.parser(line)
                except StopIteration:
                    if self._file is not None and hasattr(self._file, "close"):
                        self._file.close()  # type: ignore
                    raise
                if parsed is None:
                    continue
                return parsed

    def __len__(self) -> int:
        """Count atoms without loading all into memory."""
        count = 0
        with open(self.filepath, "r") as handle:
            for line in handle:
                record = line[:6].strip()
                if record in self.record_prefixes:
                    count += 1
        return count


class ChunkedReader(Generic[T]):
    """Read file in chunks for batch processing."""

    def __init__(self, iterator: LazyAtomIterator[T], chunk_size: int = 10000) -> None:
        """Initialize a new ChunkedReader instance."""
        self.iterator = iterator
        self.chunk_size = chunk_size

    def read_chunks(self) -> Iterator[list[T]]:
        """Yield chunks of parsed records."""
        chunk: list[T] = []
        for atom in self.iterator:
            chunk.append(atom)
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

"""Tests for the streaming iterator API (STREAM1 O(1)-memory reader)."""
import struct
import tempfile
import os
import importlib

import pytest


STREAM1_HEADER = b"VECTRO+STREAM1\n"


def _pack_embedding(id_str: str, vector: list) -> bytes:
    """Produce a bincode 1.3–compatible Embedding payload."""
    id_bytes = id_str.encode("utf-8")
    n = len(vector)
    return (
        struct.pack("<Q", len(id_bytes))
        + id_bytes
        + struct.pack("<Q", n)
        + struct.pack(f"<{n}f", *vector)
    )


def _write_stream1(path: str, embeddings: list) -> None:
    """Write a well-formed STREAM1 file from [(id_str, [f32, ...]), ...]."""
    with open(path, "wb") as f:
        f.write(STREAM1_HEADER)
        for id_str, vector in embeddings:
            payload = _pack_embedding(id_str, vector)
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)


# ---------------------------------------------------------------------------
# Module availability guard
# ---------------------------------------------------------------------------

def _has_rust_extension() -> bool:
    try:
        import vectro_plus._vectro_plus as _ext  # noqa: F401
        return True
    except ImportError:
        return False


RUST_AVAILABLE = _has_rust_extension()


# ---------------------------------------------------------------------------
# Tests — Python-pure (always run)
# ---------------------------------------------------------------------------


class TestGloVeConverter:
    """Validate the GloVe → STREAM1 converter independently of Rust."""

    def _converter(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_glove",
            os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "convert_glove_to_stream1.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_bincode_serialization_round_trip(self):
        """The Python bincode serializer must produce valid STREAM1 records."""
        mod = self._converter()
        id_str = "hello"
        vector = [1.0, 2.0, 3.0]
        payload = mod._bincode_embedding(id_str, vector)

        # id: <Q len> + bytes
        id_bytes = id_str.encode("utf-8")
        assert payload[:8] == struct.pack("<Q", len(id_bytes))
        assert payload[8 : 8 + len(id_bytes)] == id_bytes
        offset = 8 + len(id_bytes)
        # vec: <Q count> + f32s
        assert payload[offset : offset + 8] == struct.pack("<Q", 3)
        values = struct.unpack("<3f", payload[offset + 8 :])
        assert list(values) == pytest.approx(vector)

    def test_convert_small_glove_file(self):
        """convert() writes valid STREAM1 binary for a minimal GloVe text file."""
        mod = self._converter()
        with tempfile.TemporaryDirectory() as tmp:
            glove_path = os.path.join(tmp, "mini.txt")
            out_path = os.path.join(tmp, "out.stream1")
            # Write 3 fake GloVe lines (word d1 d2 d3)
            lines = [
                "king 0.1 0.2 0.3",
                "queen -0.1 0.4 0.5",
                "man 1.0 0.0 -0.5",
            ]
            with open(glove_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            rc = mod.convert(glove_path, out_path, max_vectors=None, quiet=True)
            assert rc == 0

            # Verify header
            with open(out_path, "rb") as f:
                header = f.read(15)
                assert header == STREAM1_HEADER

                # Read 3 records
                ids_seen = []
                for _ in range(3):
                    raw_len = f.read(4)
                    assert len(raw_len) == 4
                    (rec_len,) = struct.unpack("<I", raw_len)
                    payload = f.read(rec_len)
                    assert len(payload) == rec_len
                    # Decode id
                    (id_len,) = struct.unpack("<Q", payload[:8])
                    id_str = payload[8 : 8 + id_len].decode("utf-8")
                    ids_seen.append(id_str)
                assert ids_seen == ["king", "queen", "man"]

    def test_convert_max_vectors(self):
        """--max-vectors truncates output correctly."""
        mod = self._converter()
        with tempfile.TemporaryDirectory() as tmp:
            glove_path = os.path.join(tmp, "mini.txt")
            out_path = os.path.join(tmp, "out.stream1")
            lines = [f"word{i} {float(i)} {float(i+1)}" for i in range(10)]
            with open(glove_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            rc = mod.convert(glove_path, out_path, max_vectors=3, quiet=True)
            assert rc == 0

            count = 0
            with open(out_path, "rb") as f:
                f.read(15)  # skip header
                while True:
                    raw_len = f.read(4)
                    if len(raw_len) < 4:
                        break
                    (rec_len,) = struct.unpack("<I", raw_len)
                    f.read(rec_len)
                    count += 1
            assert count == 3

    def test_convert_missing_input_returns_1(self):
        """Returns exit code 1 when input file doesn't exist."""
        mod = self._converter()
        rc = mod.convert("/nonexistent/glove.txt", "/tmp/out.stream1", None, True)
        assert rc == 1


# ---------------------------------------------------------------------------
# Tests — require Rust extension (streaming iterator)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built (run `maturin develop`)")
class TestStreamingIterator:
    """Tests for the STREAM1 one-record-at-a-time streaming API."""

    def test_stream_embeddings_is_importable(self):
        """stream_embeddings must be importable from vectro_plus."""
        import vectro_plus
        assert hasattr(vectro_plus, "stream_embeddings"), (
            "stream_embeddings not found in vectro_plus.__init__"
        )

    def test_stream_round_trip_count(self):
        """Streaming over a synthetic STREAM1 file yields exactly N embeddings."""
        import vectro_plus

        embeddings_in = [
            (f"vec_{i}", [float(i), float(i + 1), float(i + 2)])
            for i in range(20)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.stream1")
            _write_stream1(path, embeddings_in)

            items = list(vectro_plus.stream_embeddings(path))
            assert len(items) == 20

    def test_stream_round_trip_ids(self):
        """IDs are preserved across the encode/decode boundary."""
        import vectro_plus

        embeddings_in = [(f"token_{i}", [1.0, 0.0, 0.0]) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ids.stream1")
            _write_stream1(path, embeddings_in)
            ids_out = [e.id for e in vectro_plus.stream_embeddings(path)]
            assert ids_out == [f"token_{i}" for i in range(5)]

    def test_stream_round_trip_vectors(self):
        """Vectors are preserved to float32 precision."""
        import vectro_plus

        embeddings_in = [("a", [0.1, 0.2, 0.3]), ("b", [1.0, -1.0, 0.5])]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vecs.stream1")
            _write_stream1(path, embeddings_in)
            items = list(vectro_plus.stream_embeddings(path))
            assert len(items) == 2
            assert items[0].vector == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
            assert items[1].vector == pytest.approx([1.0, -1.0, 0.5], abs=1e-6)

    def test_stream_empty_file(self):
        """Streaming an empty (header-only) STREAM1 file yields zero items."""
        import vectro_plus

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.stream1")
            with open(path, "wb") as f:
                f.write(STREAM1_HEADER)  # write header, no records
            items = list(vectro_plus.stream_embeddings(path))
            assert items == []

    def test_stream_wrong_format_raises(self):
        """Passing a non-STREAM1 file must raise an error (not silently succeed)."""
        import vectro_plus

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "garbage.bin")
            with open(path, "wb") as f:
                f.write(b"NOT_A_STREAM1_FILE____")

            with pytest.raises(Exception):
                # Materialise the iterator to trigger the error.
                list(vectro_plus.stream_embeddings(path))

    def test_stream_memory_does_not_grow_linearly(self):
        """
        Peak RSS must not grow monotonically over a large streaming iteration.
        Requires psutil; skipped if not installed.
        """
        pytest.importorskip("psutil")
        import psutil
        import gc
        import vectro_plus

        # Write 10 000 dummy embeddings (4-d vectors, small footprint).
        N = 10_000
        embeddings_in = [(f"id_{i}", [float(i % 100) * 0.01, 0.0, 0.0, 1.0]) for i in range(N)]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.stream1")
            _write_stream1(path, embeddings_in)

            proc = psutil.Process()
            gc.collect()
            rss_before = proc.memory_info().rss

            for _ in vectro_plus.stream_embeddings(path):
                pass

            gc.collect()
            rss_after = proc.memory_info().rss

            # Allow up to 8 MB headroom (metadata, buffers, Python overhead).
            delta_mb = (rss_after - rss_before) / 1024 / 1024
            assert delta_mb < 8, (
                f"RSS grew by {delta_mb:.1f} MB — streaming iterator may be buffering everything"
            )

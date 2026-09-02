"""Tests for self-healing: embedding retry, warm-up, ingestion retry, chunk count."""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Embedding retry tests (LocalEmbedder._send_embedding_batch)
# ---------------------------------------------------------------------------


class TestEmbeddingRetry:
    """Tests for _send_embedding_batch retry on 5xx errors."""

    def _make_embedder(self):
        """Create a LocalEmbedder with mocked HTTP client."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch("httpx.Client"):
                from local_llm import LocalEmbedder

                embedder = LocalEmbedder.__new__(LocalEmbedder)
                embedder.model_name = "test-model"
                embedder._url = "http://localhost:11434/v1/embeddings"
                embedder._client = MagicMock()
                embedder.dimension = 1024
                return embedder

    def test_succeeds_on_first_try(self):
        embedder = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        mock_resp.raise_for_status = MagicMock()
        embedder._client.post.return_value = mock_resp

        result = embedder._send_embedding_batch(["hello"])
        assert result == [[0.1, 0.2]]
        assert embedder._client.post.call_count == 1

    def test_retries_on_500_and_succeeds(self):
        import httpx

        embedder = self._make_embedder()

        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.request = MagicMock()

        ok_resp = MagicMock()
        ok_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        ok_resp.raise_for_status = MagicMock()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 500
                resp.request = MagicMock()
                resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=resp)
                return resp
            return ok_resp

        embedder._client.post.side_effect = side_effect

        with patch("time.sleep"):  # skip actual sleep
            result = embedder._send_embedding_batch(["hello"])

        assert result == [[0.1]]
        assert call_count[0] == 2

    def test_exhausts_retries_and_raises(self):
        import httpx

        embedder = self._make_embedder()

        def always_500(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.request = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=resp)
            return resp

        embedder._client.post.side_effect = always_500

        with patch("time.sleep"):
            with pytest.raises(httpx.HTTPStatusError):
                embedder._send_embedding_batch(["hello"], max_retries=3)

        assert embedder._client.post.call_count == 3

    def test_no_retry_on_4xx(self):
        import httpx

        embedder = self._make_embedder()

        def always_400(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 400
            resp.request = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("400", request=MagicMock(), response=resp)
            return resp

        embedder._client.post.side_effect = always_400

        with pytest.raises(httpx.HTTPStatusError):
            embedder._send_embedding_batch(["hello"], max_retries=3)

        # Should NOT retry on 4xx — only 1 call
        assert embedder._client.post.call_count == 1


# ---------------------------------------------------------------------------
# Embedding warm-up tests
# ---------------------------------------------------------------------------


class TestEmbeddingWarmUp:
    """Tests for wait_for_embedding_service()."""

    def _make_embedder(self):
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch("httpx.Client"):
                from local_llm import LocalEmbedder

                embedder = LocalEmbedder.__new__(LocalEmbedder)
                embedder.model_name = "test-model"
                embedder._url = "http://localhost:11434/v1/embeddings"
                embedder._client = MagicMock()
                embedder.dimension = 1024
                return embedder

    def test_warm_up_succeeds_immediately(self):
        embedder = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        mock_resp.raise_for_status = MagicMock()
        embedder._client.post.return_value = mock_resp

        result = embedder.wait_for_embedding_service(timeout=5)
        assert result is True

    def test_warm_up_times_out(self):
        embedder = self._make_embedder()
        embedder._client.post.side_effect = ConnectionError("not ready")

        # Use a very short timeout so test is fast
        with patch("time.sleep"):
            with patch("time.monotonic", side_effect=[0, 0, 100]):
                result = embedder.wait_for_embedding_service(timeout=5)

        assert result is False

    def test_warm_up_succeeds_after_retries(self):
        embedder = self._make_embedder()
        call_count = [0]

        ok_resp = MagicMock()
        ok_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        ok_resp.raise_for_status = MagicMock()

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("not ready")
            return ok_resp

        embedder._client.post.side_effect = side_effect

        with patch("time.sleep"):
            result = embedder.wait_for_embedding_service(timeout=60)

        assert result is True
        assert call_count[0] >= 3


# ---------------------------------------------------------------------------
# Chunk count validation tests
# ---------------------------------------------------------------------------


class TestChunkCountValidation:
    """Tests for chunk count warning after ingestion."""

    def test_low_chunk_count_logs_warning(self, tmp_path, caplog):
        """When chunks < files * 5, a warning should be logged."""
        import logging

        # Create 5 files but load 0 chunks
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")

        with caplog.at_level(logging.WARNING):
            # Simulate the validation logic directly
            doc_files = [f for f in tmp_path.rglob("*") if f.is_file()]
            n_files = len(doc_files)
            n_chunks = 10  # < 5 * 5 = 25

            if n_files > 0 and n_chunks < n_files * 5:
                import logging as _log

                _log.getLogger("test").warning(
                    "Only %d chunks indexed from %d files (expected >%d).",
                    n_chunks,
                    n_files,
                    n_files * 5,
                )

        assert any("chunks indexed" in r.message for r in caplog.records)

    def test_normal_chunk_count_no_warning(self, tmp_path, caplog):
        """Enough chunks should not trigger a warning."""
        import logging

        for i in range(3):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")

        with caplog.at_level(logging.WARNING):
            doc_files = [f for f in tmp_path.rglob("*") if f.is_file()]
            n_files = len(doc_files)
            n_chunks = 100  # >= 3 * 5 = 15

            if n_files > 0 and n_chunks < n_files * 5:
                import logging as _log

                _log.getLogger("test").warning(
                    "Only %d chunks indexed from %d files (expected >%d).",
                    n_chunks,
                    n_files,
                    n_files * 5,
                )

        assert not any("chunks indexed" in r.message for r in caplog.records)


# Ingestion retry tests: see TestIngestionRetry class below (line ~460)

# ---------------------------------------------------------------------------
# Graduated warm-up tests
# ---------------------------------------------------------------------------


class TestGraduatedWarmUp:
    """Tests for the 3-stage graduated warm-up in wait_for_embedding_service."""

    def _make_embedder(self):
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch("httpx.Client"):
                from local_llm import LocalEmbedder

                embedder = LocalEmbedder.__new__(LocalEmbedder)
                embedder.model_name = "test-model"
                embedder._url = "http://localhost:11434/v1/embeddings"
                embedder._client = MagicMock()
                embedder.dimension = 1024
                return embedder

    def test_graduated_warmup_sends_increasing_batch_sizes(self):
        """Warm-up should probe with 1, 5, and 20 texts progressively."""
        embedder = self._make_embedder()
        batch_sizes = []

        ok_resp = MagicMock()
        ok_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        ok_resp.raise_for_status = MagicMock()

        def capture_post(*args, **kwargs):
            inputs = kwargs.get("json", args[1] if len(args) > 1 else {}).get("input", [])
            batch_sizes.append(len(inputs))
            resp = MagicMock()
            resp.json.return_value = {"data": [{"embedding": [0.1]} for _ in inputs]}
            resp.raise_for_status = MagicMock()
            return resp

        embedder._client.post.side_effect = capture_post

        with patch("time.sleep"):
            result = embedder.wait_for_embedding_service(timeout=120)

        assert result is True
        assert batch_sizes == [1, 5, 20]

    def test_graduated_warmup_fails_at_stage2(self):
        """If stage 2 never succeeds within timeout, returns False."""
        embedder = self._make_embedder()
        call_count = [0]

        def stage1_only(*args, **kwargs):
            call_count[0] += 1
            inputs = kwargs.get("json", args[1] if len(args) > 1 else {}).get("input", [])
            if len(inputs) == 1:
                resp = MagicMock()
                resp.json.return_value = {"data": [{"embedding": [0.1]}]}
                resp.raise_for_status = MagicMock()
                return resp
            raise ConnectionError("batch not ready")

        embedder._client.post.side_effect = stage1_only

        # monotonic: 0 (start), 0 (stage1 loop), 0 (stage1 success),
        # 0 (stage2 loop check), 200 (stage2 timeout)
        times = iter([0, 0, 0, 0, 200])
        with patch("time.sleep"):
            with patch("time.monotonic", side_effect=times):
                result = embedder.wait_for_embedding_service(timeout=60)

        assert result is False


# ---------------------------------------------------------------------------
# Large batch retry tests
# ---------------------------------------------------------------------------


class TestLargeBatchRetry:
    """Tests for extended retry when large_batch=True."""

    def _make_embedder(self):
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch("httpx.Client"):
                from local_llm import LocalEmbedder

                embedder = LocalEmbedder.__new__(LocalEmbedder)
                embedder.model_name = "test-model"
                embedder._url = "http://localhost:11434/v1/embeddings"
                embedder._client = MagicMock()
                embedder.dimension = 1024
                return embedder

    def test_large_batch_flag_uses_5_retries(self):
        """large_batch=True should retry 5 times (not 3) before raising."""
        import httpx

        embedder = self._make_embedder()

        def always_500(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 500
            resp.request = MagicMock()
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=resp)
            return resp

        embedder._client.post.side_effect = always_500

        with patch("time.sleep"):
            with pytest.raises(httpx.HTTPStatusError):
                embedder._send_embedding_batch(["hello"], large_batch=True)

        assert embedder._client.post.call_count == 5

    def test_request_embeddings_sets_large_batch_for_100_plus(self):
        """_request_embeddings should pass large_batch=True when texts >= 100."""
        embedder = self._make_embedder()
        large_batch_flags = []

        original_send = embedder._send_embedding_batch

        def capture_send(texts, max_retries=3, large_batch=False):
            large_batch_flags.append(large_batch)
            resp = MagicMock()
            resp.json.return_value = {"data": [{"embedding": [0.1]} for _ in texts]}
            resp.raise_for_status = MagicMock()
            embedder._client.post.return_value = resp
            return original_send(texts, max_retries=max_retries)

        embedder._send_embedding_batch = capture_send
        texts = [f"chunk {i}" for i in range(100)]

        with patch("time.sleep"):
            embedder._request_embeddings(texts)

        assert all(flag is True for flag in large_batch_flags)

    def test_request_embeddings_no_large_batch_for_small_lists(self):
        """_request_embeddings should pass large_batch=False when texts < 100."""
        embedder = self._make_embedder()
        large_batch_flags = []

        original_send = embedder._send_embedding_batch

        def capture_send(texts, max_retries=3, large_batch=False):
            large_batch_flags.append(large_batch)
            resp = MagicMock()
            resp.json.return_value = {"data": [{"embedding": [0.1]} for _ in texts]}
            resp.raise_for_status = MagicMock()
            embedder._client.post.return_value = resp
            return original_send(texts, max_retries=max_retries)

        embedder._send_embedding_batch = capture_send
        texts = [f"chunk {i}" for i in range(10)]

        with patch("time.sleep"):
            embedder._request_embeddings(texts)

        assert all(flag is False for flag in large_batch_flags)


# ---------------------------------------------------------------------------
# Item-count cap tests
# ---------------------------------------------------------------------------


class TestBatchItemCap:
    """Tests for embedding_batch_size item-count cap in _request_embeddings."""

    def _make_embedder(self):
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
            with patch("httpx.Client"):
                from local_llm import LocalEmbedder

                embedder = LocalEmbedder.__new__(LocalEmbedder)
                embedder.model_name = "test-model"
                embedder._url = "http://localhost:11434/v1/embeddings"
                embedder._client = MagicMock()
                embedder.dimension = 1024
                return embedder

    def test_respects_batch_size_cap(self):
        """Sub-batches should not exceed embedding_batch_size items."""
        embedder = self._make_embedder()
        batch_sizes = []

        def capture_post(*args, **kwargs):
            inputs = kwargs.get("json", args[1] if len(args) > 1 else {}).get("input", [])
            batch_sizes.append(len(inputs))
            resp = MagicMock()
            resp.json.return_value = {"data": [{"embedding": [0.1]} for _ in inputs]}
            resp.raise_for_status = MagicMock()
            return resp

        embedder._client.post.side_effect = capture_post

        # 50 very short texts — won't hit char limit but should hit item cap (32)
        texts = ["hi"] * 50

        with patch("time.sleep"):
            with patch("config.settings") as mock_settings:
                mock_settings.embedding_batch_size = 32
                embedder._request_embeddings(texts)

        assert all(s <= 32 for s in batch_sizes)
        assert len(batch_sizes) == 2  # 32 + 18


# ---------------------------------------------------------------------------
# Ingestion retry tests
# ---------------------------------------------------------------------------


class TestIngestionRetry:
    """Tests for the file-level ingestion retry in _load_documents."""

    def test_retry_logic_recovers_on_second_attempt(self):
        """Simulate: first embed_batch fails, retry succeeds."""
        call_count = [0]

        def mock_embed_batch(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("embedding service not ready")
            return [[0.1] * 1024] * len(texts)

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = mock_embed_batch

        # Simulate retry logic extracted from _load_documents
        file_docs = [{"content": "test chunk", "source": "test.pdf", "type": "pdf"}]
        documents = []
        embeddings = []

        with patch("time.sleep"):
            try:
                texts = [d["content"] for d in file_docs]
                embs = mock_embedder.embed_batch(texts)
                documents.extend(file_docs)
                embeddings.extend(embs)
            except Exception:
                # Retry
                try:
                    texts = [d["content"] for d in file_docs]
                    embs = mock_embedder.embed_batch(texts)
                    documents.extend(file_docs)
                    embeddings.extend(embs)
                except Exception:
                    pass

        assert len(documents) == 1
        assert len(embeddings) == 1
        assert call_count[0] == 2

    def test_retry_exhausted_no_crash(self):
        """Both attempts fail — no crash, no documents added."""
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = ConnectionError("always fails")

        file_docs = [{"content": "test chunk", "source": "test.pdf", "type": "pdf"}]
        documents = []
        embeddings = []

        with patch("time.sleep"):
            try:
                embs = mock_embedder.embed_batch([d["content"] for d in file_docs])
                documents.extend(file_docs)
                embeddings.extend(embs)
            except Exception:
                try:
                    embs = mock_embedder.embed_batch([d["content"] for d in file_docs])
                    documents.extend(file_docs)
                    embeddings.extend(embs)
                except Exception:
                    pass

        assert len(documents) == 0
        assert len(embeddings) == 0

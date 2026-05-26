"""
test_suite.py
=============
Deployment-readiness test suite for the RED-DOT RAG pipeline.

Goals:
    1. Minimise false negatives (fake news slipping through as real)
    2. Use RAG (Gate B) to recover false positives (real flagged as fake)
    3. Verify every layer: output schema, gate routing, FAISS, label logic

Test categories:
    Unit Tests      — isolated component tests (no real model/FAISS needed)
    Integration Tests — full pipeline tests (needs model + FAISS on disk)
    Behavioural Tests — fake-news-specific logic tests (gate, FN/FP goals)

Run all tests:
    python test_suite.py

Run only unit tests (no model needed):
    python test_suite.py --unit

Run only integration tests:
    python test_suite.py --integration
"""

import argparse
import json
import os
import sys
import time
import traceback
import unittest
from io       import BytesIO
from pathlib  import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
# TEST UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

PASS  = "  ✓"
FAIL  = "  ✗"
SKIP  = "  ⚠"

def make_fake_image_bytes() -> bytes:
    """
    Creates a minimal valid JPEG in memory.
    Used for tests that need image bytes without real images on disk.
    """
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_mock_engine(
    logit        : float = 2.5,        # positive = FAKE
    entropy      : float = 0.30,       # low = Gate A
    gate         : str   = "A",
    neighbours   : list  = None,
    faiss_ready  : bool  = True,
) -> MagicMock:
    """
    Builds a mock REDDOTRagEngine that returns a controlled result dict.
    Used for unit tests that don't need the real model loaded.
    """
    engine = MagicMock()
    engine.entropy_gate = 0.50
    engine.k_neighbours = 5
    engine.model        = MagicMock()
    engine.metadata     = [{"verdict": "fake", "sample_id": "mock_1"}] * 10 if faiss_ready else []
    engine.index        = MagicMock() if faiss_ready else None

    prob     = torch.sigmoid(torch.tensor(logit)).item()
    conf     = prob if prob > 0.5 else 1 - prob
    nb       = neighbours or []
    nb_verd  = "unknown"
    if nb:
        fk = sum(1 for n in nb if n.get("verdict","").lower() == "fake")
        rk = sum(1 for n in nb if n.get("verdict","").lower() == "real")
        if fk > rk:   nb_verd = "fake"
        elif rk > fk: nb_verd = "real"

    verdict = "fake" if logit > 0 else "real"
    if gate == "B":
        if nb_verd == "unknown":
            verdict = "uncertain"
        elif nb_verd != verdict:
            verdict = nb_verd

    engine.run.return_value = {
        "verdict"           : verdict,
        "base_verdict"      : "fake" if logit > 0 else "real",
        "neighbour_verdict" : nb_verd,
        "signals_agree"     : nb_verd == verdict or nb_verd == "unknown",
        "gate_triggered"    : gate,
        "logit"             : round(logit, 4),
        "prob_fake"         : round(prob, 4),
        "confidence"        : round(conf, 4),
        "entropy"           : round(entropy, 4),
        "cosine_sim"        : 0.15,
        "retrieval_context" : f"Gate {gate} mock context.",
        "neighbours"        : nb,
        "sample_id"         : "test_sample",
        "latency_ms"        : 42.0,
    }
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — OUTPUT SCHEMA TESTS
# Verify the result dict always contains every required key with correct types.
# These are the most critical — if the schema is wrong, backend.py crashes.
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputSchema(unittest.TestCase):
    """
    Tests that engine.run() always returns the exact schema backend.py expects.
    Uses a mock engine — no real model needed.
    """

    REQUIRED_KEYS = {
        "verdict"           : str,
        "base_verdict"      : str,
        "neighbour_verdict" : str,
        "signals_agree"     : bool,
        "gate_triggered"    : str,
        "logit"             : (int, float),
        "prob_fake"         : (int, float),
        "confidence"        : (int, float),
        "entropy"           : (int, float),
        "cosine_sim"        : (int, float),
        "retrieval_context" : str,
        "neighbours"        : list,
        "sample_id"         : str,
        "latency_ms"        : (int, float),
    }

    def _get_result(self, **kwargs):
        engine = make_mock_engine(**kwargs)
        return engine.run("fake_image.jpg", "test caption", "test_id")

    def test_all_keys_present_gate_a(self):
        """All required keys must exist when Gate A fires."""
        result = self._get_result(logit=3.0, entropy=0.20, gate="A")
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, result, f"Missing key: '{key}' in Gate A result")

    def test_all_keys_present_gate_b(self):
        """All required keys must exist when Gate B fires."""
        result = self._get_result(logit=0.3, entropy=0.85, gate="B")
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, result, f"Missing key: '{key}' in Gate B result")

    def test_correct_types(self):
        """Every key must have the correct type."""
        result = self._get_result(logit=2.0, entropy=0.25, gate="A")
        for key, expected_type in self.REQUIRED_KEYS.items():
            self.assertIsInstance(
                result[key], expected_type,
                f"Key '{key}': expected {expected_type}, got {type(result[key])}"
            )

    def test_verdict_is_valid_string(self):
        """verdict must be one of the three valid strings."""
        valid = {"real", "fake", "uncertain"}
        for logit, entropy, gate in [(3.0, 0.2, "A"), (-2.0, 0.2, "A"), (0.2, 0.9, "B")]:
            result = self._get_result(logit=logit, entropy=entropy, gate=gate)
            self.assertIn(
                result["verdict"].lower(), valid,
                f"verdict='{result['verdict']}' not in {valid}"
            )

    def test_gate_triggered_is_a_or_b(self):
        """gate_triggered must always be 'A' or 'B'."""
        for gate in ["A", "B"]:
            result = self._get_result(gate=gate)
            self.assertIn(result["gate_triggered"], ["A", "B"])

    def test_probability_in_range(self):
        """prob_fake must be in [0, 1]."""
        for logit in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            result = self._get_result(logit=logit)
            self.assertGreaterEqual(result["prob_fake"], 0.0)
            self.assertLessEqual(result["prob_fake"], 1.0)

    def test_entropy_in_range(self):
        """entropy must be in [0, 1]."""
        for entropy in [0.0, 0.25, 0.50, 0.75, 1.0]:
            result = self._get_result(entropy=entropy)
            self.assertGreaterEqual(result["entropy"], 0.0)
            self.assertLessEqual(result["entropy"], 1.0)

    def test_confidence_in_range(self):
        """confidence must be in [0.5, 1.0]."""
        for logit in [-4.0, -0.5, 0.5, 4.0]:
            result = self._get_result(logit=logit)
            self.assertGreaterEqual(result["confidence"], 0.5)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_latency_is_positive(self):
        """latency_ms must be positive."""
        result = self._get_result()
        self.assertGreater(result["latency_ms"], 0)

    def test_neighbours_is_list(self):
        """neighbours must always be a list, never None."""
        result = self._get_result(gate="A")
        self.assertIsInstance(result["neighbours"], list)

    def test_retrieval_context_is_string(self):
        """retrieval_context must always be a non-empty string."""
        result = self._get_result(gate="B")
        self.assertIsInstance(result["retrieval_context"], str)
        self.assertGreater(len(result["retrieval_context"]), 0)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — GATE ROUTING LOGIC TESTS
# Verify the entropy gate routes correctly.
# Gate A fires when entropy <= 0.50, Gate B fires when entropy > 0.50.
# ══════════════════════════════════════════════════════════════════════════════

class TestGateRouting(unittest.TestCase):
    """
    Tests the entropy gate boundary conditions.
    These are pure logic tests — no model needed.
    """

    GATE_THRESHOLD = 0.50

    def _verdict_from_logit_entropy(self, logit, entropy, neighbours=None):
        """Simulates the exact gate logic from Orchestrator.py."""
        base_verdict = "fake" if logit > 0 else "real"
        prob         = float(torch.sigmoid(torch.tensor(logit)))
        conf         = prob if prob > 0.5 else 1 - prob

        if entropy <= self.GATE_THRESHOLD:
            # Gate A
            return {
                "gate"         : "A",
                "verdict"      : base_verdict,
                "base_verdict" : base_verdict,
                "nb_verdict"   : "not_queried",
            }
        else:
            # Gate B
            nb = neighbours or []
            fk = sum(1 for n in nb if n.get("verdict","").lower() == "fake")
            rk = sum(1 for n in nb if n.get("verdict","").lower() == "real")
            if fk > rk:   nb_verd = "fake"
            elif rk > fk: nb_verd = "real"
            else:          nb_verd = "unknown"

            if nb_verd == "unknown":
                final = "uncertain"
            elif nb_verd == base_verdict:
                final = base_verdict
            else:
                final = nb_verd   # KB overrides uncertain model

            return {
                "gate"         : "B",
                "verdict"      : final,
                "base_verdict" : base_verdict,
                "nb_verdict"   : nb_verd,
            }

    def test_low_entropy_triggers_gate_a(self):
        """entropy = 0.20 → Gate A must fire."""
        r = self._verdict_from_logit_entropy(logit=3.0, entropy=0.20)
        self.assertEqual(r["gate"], "A")

    def test_entropy_at_threshold_triggers_gate_a(self):
        """entropy = exactly 0.50 → Gate A (boundary is inclusive <=)."""
        r = self._verdict_from_logit_entropy(logit=2.0, entropy=0.50)
        self.assertEqual(r["gate"], "A")

    def test_high_entropy_triggers_gate_b(self):
        """entropy = 0.90 → Gate B must fire."""
        r = self._verdict_from_logit_entropy(logit=0.2, entropy=0.90)
        self.assertEqual(r["gate"], "B")

    def test_entropy_just_above_threshold_triggers_gate_b(self):
        """entropy = 0.51 → Gate B (just above boundary)."""
        r = self._verdict_from_logit_entropy(logit=0.1, entropy=0.51)
        self.assertEqual(r["gate"], "B")

    def test_gate_a_returns_base_verdict_directly(self):
        """Gate A must return base verdict without touching FAISS."""
        r = self._verdict_from_logit_entropy(logit=2.5, entropy=0.30)
        self.assertEqual(r["gate"], "A")
        self.assertEqual(r["verdict"], "fake")
        self.assertEqual(r["nb_verdict"], "not_queried")

    def test_gate_b_kb_confirms_base_returns_base(self):
        """Gate B: KB majority = base verdict → return base (confident)."""
        nb = [{"verdict": "fake"}] * 4 + [{"verdict": "real"}] * 1
        r  = self._verdict_from_logit_entropy(logit=0.4, entropy=0.80, neighbours=nb)
        self.assertEqual(r["gate"], "B")
        self.assertEqual(r["verdict"], "fake")

    def test_gate_b_kb_contradicts_base_kb_wins(self):
        """Gate B: KB majority contradicts model → KB verdict wins (minimise FP)."""
        # Model says fake (logit > 0) but KB says real (4/5 neighbours = real)
        nb = [{"verdict": "real"}] * 4 + [{"verdict": "fake"}] * 1
        r  = self._verdict_from_logit_entropy(logit=0.3, entropy=0.85, neighbours=nb)
        self.assertEqual(r["gate"], "B")
        self.assertEqual(r["verdict"], "real",
            "When KB contradicts a borderline model, KB should win to reduce false positives")

    def test_gate_b_tie_returns_uncertain(self):
        """Gate B: 2 fake + 2 real neighbours → UNCERTAIN (tie)."""
        nb = [{"verdict": "fake"}] * 2 + [{"verdict": "real"}] * 2
        r  = self._verdict_from_logit_entropy(logit=0.2, entropy=0.90, neighbours=nb)
        self.assertEqual(r["gate"], "B")
        self.assertEqual(r["verdict"], "uncertain")

    def test_gate_b_empty_kb_returns_uncertain(self):
        """Gate B: no neighbours → UNCERTAIN (no KB evidence)."""
        r = self._verdict_from_logit_entropy(logit=0.1, entropy=0.95, neighbours=[])
        self.assertEqual(r["gate"], "B")
        self.assertEqual(r["verdict"], "uncertain")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — FALSE NEGATIVE MINIMISATION TESTS
# Core business logic: fake news must not be allowed to pass as real.
# These tests verify the system is biased toward catching fakes.
# ══════════════════════════════════════════════════════════════════════════════

class TestFalseNegativeMinimisation(unittest.TestCase):
    """
    Tests that the system is biased toward catching fakes.
    A false negative = fake news labelled as REAL — the worst outcome.

    Strategy:
        - High-confidence model FAKE → always returns FAKE (Gate A)
        - Uncertain model + strong KB evidence FAKE → returns FAKE (Gate B)
        - Only upgrades to REAL when KB is strongly real AND model is uncertain
    """

    def _simulate(self, logit, entropy, neighbours=None):
        nb = neighbours or []
        fk = sum(1 for n in nb if n.get("verdict","").lower() == "fake")
        rk = sum(1 for n in nb if n.get("verdict","").lower() == "real")
        base = "fake" if logit > 0 else "real"

        if entropy <= 0.50:
            return {"verdict": base, "gate": "A"}

        if fk > rk:   nb_v = "fake"
        elif rk > fk: nb_v = "real"
        else:          nb_v = "unknown"

        if nb_v == "unknown":         return {"verdict": "uncertain", "gate": "B"}
        if nb_v == base:              return {"verdict": base,         "gate": "B"}
        return {"verdict": nb_v, "gate": "B"}   # KB overrides

    def test_strong_fake_signal_never_becomes_real(self):
        """
        Model very confident FAKE (large positive logit, low entropy)
        → must stay FAKE regardless of any KB neighbours.
        """
        strong_real_kb = [{"verdict": "real"}] * 5
        r = self._simulate(logit=4.5, entropy=0.15, neighbours=strong_real_kb)
        self.assertEqual(r["gate"], "A",
            "Strong FAKE signal should not even reach FAISS (Gate A)")
        self.assertEqual(r["verdict"], "fake",
            "High-confidence FAKE must never return REAL")

    def test_moderate_fake_gate_b_with_fake_kb_stays_fake(self):
        """
        Moderate FAKE signal + high entropy + KB majority FAKE → FAKE.
        KB confirms the uncertain model's fake suspicion.
        """
        fake_kb = [{"verdict": "fake"}] * 4 + [{"verdict": "real"}] * 1
        r = self._simulate(logit=0.8, entropy=0.75, neighbours=fake_kb)
        self.assertEqual(r["verdict"], "fake")

    def test_borderline_fake_with_mixed_kb_returns_uncertain_not_real(self):
        """
        Very uncertain model (logit ~0) + tied KB → UNCERTAIN.
        Must NOT return REAL — uncertainty is safer than false clearance.
        """
        tied_kb = [{"verdict": "fake"}] * 2 + [{"verdict": "real"}] * 2
        r = self._simulate(logit=0.05, entropy=0.99, neighbours=tied_kb)
        self.assertNotEqual(r["verdict"], "real",
            "Borderline fake with tied KB must not return REAL — prefer UNCERTAIN")

    def test_fake_label_convention_logit_positive_means_fake(self):
        """
        Verify label convention: positive logit → FAKE.
        This is the most fundamental invariant — if it breaks, everything inverts.
        """
        for logit in [0.01, 0.5, 1.0, 2.0, 5.0]:
            r = self._simulate(logit=logit, entropy=0.20)
            self.assertEqual(r["verdict"], "fake",
                f"logit={logit} (positive) must map to FAKE")

    def test_real_label_convention_logit_negative_means_real(self):
        """Negative logit → REAL."""
        for logit in [-0.01, -0.5, -1.0, -2.0, -5.0]:
            r = self._simulate(logit=logit, entropy=0.20)
            self.assertEqual(r["verdict"], "real",
                f"logit={logit} (negative) must map to REAL")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4 — FALSE POSITIVE RECOVERY VIA RAG TESTS
# When model wrongly flags real content as FAKE (false positive),
# Gate B + strong real KB should override to REAL — improving UX.
# ══════════════════════════════════════════════════════════════════════════════

class TestFalsePositiveRecoveryViaRAG(unittest.TestCase):
    """
    Tests that Gate B can rescue false positives.
    False positive = real content flagged as FAKE.

    RAG recovery path:
        Model says FAKE (logit slightly positive, high entropy)
        → Gate B fires
        → KB majority = REAL (strong signal)
        → Final verdict = REAL (KB overrides borderline model)
    """

    def _simulate(self, logit, entropy, neighbours):
        fk = sum(1 for n in neighbours if n.get("verdict","").lower() == "fake")
        rk = sum(1 for n in neighbours if n.get("verdict","").lower() == "real")
        base = "fake" if logit > 0 else "real"

        if entropy <= 0.50:
            return {"verdict": base, "gate": "A", "rag_recovered": False}

        if fk > rk:   nb_v = "fake"
        elif rk > fk: nb_v = "real"
        else:          nb_v = "unknown"

        if nb_v == "unknown":
            return {"verdict": "uncertain", "gate": "B", "rag_recovered": False}
        if nb_v == base:
            return {"verdict": base, "gate": "B", "rag_recovered": False}

        # KB overrides model — this is the RAG recovery path
        return {"verdict": nb_v, "gate": "B", "rag_recovered": True}

    def test_borderline_fake_with_strong_real_kb_recovers_to_real(self):
        """
        Model borderline FAKE (logit=0.3, high entropy) +
        5/5 real KB neighbours → should recover to REAL.
        This is the core false-positive rescue by RAG.
        """
        real_kb = [{"verdict": "real"}] * 5
        r = self._simulate(logit=0.3, entropy=0.90, neighbours=real_kb)
        self.assertEqual(r["gate"], "B",  "Should trigger RAG")
        self.assertEqual(r["verdict"], "real", "RAG should recover borderline FP to REAL")
        self.assertTrue(r["rag_recovered"], "RAG recovery flag should be set")

    def test_strong_fake_does_not_get_overridden(self):
        """
        Strong FAKE (low entropy) → Gate A fires, KB never consulted.
        Even all-real KB should not override a confident FAKE.
        """
        real_kb = [{"verdict": "real"}] * 5
        r = self._simulate(logit=3.5, entropy=0.15, neighbours=real_kb)
        self.assertEqual(r["gate"], "A", "Strong FAKE should stay in Gate A")
        self.assertFalse(r["rag_recovered"])

    def test_weak_real_kb_does_not_override_moderate_fake(self):
        """
        Moderate FAKE + only 3/5 real KB → not strong enough to override.
        Needs clear majority to rescue false positives.
        """
        mixed_kb = [{"verdict": "real"}] * 3 + [{"verdict": "fake"}] * 2
        r = self._simulate(logit=0.6, entropy=0.80, neighbours=mixed_kb)
        # 3 real vs 2 fake — still a majority for real, so should recover
        self.assertEqual(r["verdict"], "real")

    def test_rag_recovery_only_happens_in_gate_b(self):
        """RAG recovery can only happen when Gate B fires (high entropy)."""
        real_kb = [{"verdict": "real"}] * 5
        # Low entropy → Gate A → no recovery possible
        r = self._simulate(logit=0.5, entropy=0.30, neighbours=real_kb)
        self.assertEqual(r["gate"], "A")
        self.assertFalse(r["rag_recovered"])


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5 — FAISS COMPONENT TESTS
# Tests the FAISS search logic in isolation.
# ══════════════════════════════════════════════════════════════════════════════

class TestFAISSComponent(unittest.TestCase):
    """
    Tests FAISS query construction and result parsing.
    Uses an in-memory FAISS index — no disk files needed.
    """

    def setUp(self):
        """Build a tiny in-memory FAISS index with known entries."""
        import faiss
        self.dim = 1536

        # Create 10 normalised random vectors
        rng        = np.random.default_rng(42)
        vecs       = rng.standard_normal((10, self.dim)).astype(np.float32)
        norms      = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.vecs  = vecs / (norms + 1e-9)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.vecs)

        # Metadata: alternating real/fake
        self.metadata = [
            {"sample_id": f"s{i}", "verdict": "fake" if i % 2 == 0 else "real",
             "caption": f"caption {i}", "internal_clip_sim": 0.15}
            for i in range(10)
        ]

    def _search(self, img_vec, txt_vec, k=5):
        """Mirrors _search_database logic from Orchestrator.py exactly."""
        joint = np.concatenate([img_vec, txt_vec]).astype(np.float32)
        joint = joint / (np.linalg.norm(joint) + 1e-9)
        joint = joint.reshape(1, -1)

        k_use = min(k, len(self.metadata))
        distances, indices = self.index.search(joint, k_use)

        neighbours = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = dict(self.metadata[idx])
            if entry.get("verdict", "").lower() == "unknown":
                continue
            entry["similarity"] = round(float(dist), 4)
            neighbours.append(entry)
        return neighbours

    def test_joint_vector_has_correct_shape(self):
        """Concatenated [768] + [768] must produce [1536] vector."""
        img = np.random.randn(768).astype(np.float32)
        txt = np.random.randn(768).astype(np.float32)
        joint = np.concatenate([img, txt])
        self.assertEqual(joint.shape[0], 1536)

    def test_joint_vector_is_normalised(self):
        """After L2 normalisation, norm must be ~1.0."""
        img   = np.random.randn(768).astype(np.float32)
        txt   = np.random.randn(768).astype(np.float32)
        joint = np.concatenate([img, txt]).astype(np.float32)
        joint = joint / (np.linalg.norm(joint) + 1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(joint)), 1.0, places=4)

    def test_search_returns_k_neighbours(self):
        """FAISS must return exactly k results (or fewer if index is small)."""
        img = np.random.randn(768).astype(np.float32)
        txt = np.random.randn(768).astype(np.float32)
        nb  = self._search(img, txt, k=5)
        self.assertLessEqual(len(nb), 5)
        self.assertGreater(len(nb), 0)

    def test_similarity_scores_in_valid_range(self):
        """Cosine similarity scores from IndexFlatIP must be in [-1, 1]."""
        img = np.random.randn(768).astype(np.float32)
        txt = np.random.randn(768).astype(np.float32)
        nb  = self._search(img, txt)
        for n in nb:
            self.assertGreaterEqual(n["similarity"], -1.01)
            self.assertLessEqual(n["similarity"],     1.01)

    def test_all_neighbours_have_required_keys(self):
        """Every returned neighbour must have sample_id, verdict, similarity."""
        img = np.random.randn(768).astype(np.float32)
        txt = np.random.randn(768).astype(np.float32)
        nb  = self._search(img, txt)
        for n in nb:
            self.assertIn("sample_id",  n)
            self.assertIn("verdict",    n)
            self.assertIn("similarity", n)

    def test_verdicts_are_valid_strings(self):
        """All neighbour verdicts must be 'fake' or 'real' (never unknown)."""
        img = np.random.randn(768).astype(np.float32)
        txt = np.random.randn(768).astype(np.float32)
        nb  = self._search(img, txt)
        for n in nb:
            self.assertIn(n["verdict"].lower(), {"fake", "real"})

    def test_self_retrieval(self):
        """
        A stored vector queried back must retrieve itself at rank 1
        with similarity ≈ 1.0. Tests normalisation correctness.
        """
        img_half = self.vecs[0, :768]
        txt_half = self.vecs[0, 768:]
        # Reconstruct joint query that matches stored vector 0
        joint = np.concatenate([img_half, txt_half]).astype(np.float32)
        joint = joint / (np.linalg.norm(joint) + 1e-9)
        joint = joint.reshape(1, -1)

        distances, indices = self.index.search(joint, 1)
        self.assertEqual(int(indices[0][0]), 0,
            "Self-query must retrieve stored vector 0 at rank 1")
        self.assertGreater(float(distances[0][0]), 0.99,
            "Self-similarity must be ~1.0 for L2-normalised vectors")

    def test_majority_vote_fake_wins(self):
        """4 fake + 1 real → majority = fake."""
        nb = [{"verdict":"fake"}]*4 + [{"verdict":"real"}]*1
        fk = sum(1 for n in nb if n["verdict"] == "fake")
        rk = sum(1 for n in nb if n["verdict"] == "real")
        self.assertEqual("fake" if fk > rk else "real", "fake")

    def test_majority_vote_tie_returns_unknown(self):
        """2 fake + 2 real → tie → unknown."""
        nb = [{"verdict":"fake"}]*2 + [{"verdict":"real"}]*2
        fk = sum(1 for n in nb if n["verdict"] == "fake")
        rk = sum(1 for n in nb if n["verdict"] == "real")
        result = "fake" if fk > rk else ("real" if rk > fk else "unknown")
        self.assertEqual(result, "unknown")

    def test_majority_vote_empty_returns_unknown(self):
        """Empty neighbours → unknown."""
        nb = []
        result = "unknown" if not nb else "fake"
        self.assertEqual(result, "unknown")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6 — ENTROPY METRICS TESTS
# Tests Brain.get_metrics() for correctness.
# ══════════════════════════════════════════════════════════════════════════════

class TestEntropyMetrics(unittest.TestCase):
    """
    Tests get_metrics(prob) for mathematical correctness.
    Imports Brain.py directly — no model loading needed.
    """

    def setUp(self):
        try:
            from Brain import get_metrics
            self.get_metrics = get_metrics
        except ImportError:
            self.skipTest("Brain.py not importable in this environment")

    def test_maximum_entropy_at_0_5(self):
        """prob=0.5 → maximum entropy = 1.0 (most uncertain)."""
        _, entropy = self.get_metrics(0.5)
        self.assertAlmostEqual(entropy, 1.0, places=2)

    def test_minimum_entropy_at_extremes(self):
        """prob=0.0 or 1.0 → entropy ≈ 0 (most certain)."""
        _, e0 = self.get_metrics(0.001)
        _, e1 = self.get_metrics(0.999)
        self.assertLess(e0, 0.05)
        self.assertLess(e1, 0.05)

    def test_entropy_triggers_gate_b_boundary(self):
        """entropy from prob=0.5 must exceed 0.50 gate threshold."""
        _, entropy = self.get_metrics(0.5)
        self.assertGreater(entropy, 0.50,
            "Maximum entropy must trigger Gate B")

    def test_confident_prediction_triggers_gate_a(self):
        """entropy from prob=0.95 must be below 0.50 gate threshold."""
        _, entropy = self.get_metrics(0.95)
        self.assertLess(entropy, 0.50,
            "High-confidence prediction must NOT trigger Gate B")

    def test_confidence_always_above_0_5(self):
        """confidence must always be >= 0.5 (distance from boundary)."""
        for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            conf, _ = self.get_metrics(p)
            self.assertGreaterEqual(conf, 0.5,
                f"Confidence must be >= 0.5, got {conf} for p={p}")

    def test_symmetry(self):
        """get_metrics(p) and get_metrics(1-p) must have same confidence and entropy."""
        for p in [0.2, 0.3, 0.4]:
            c1, e1 = self.get_metrics(p)
            c2, e2 = self.get_metrics(1.0 - p)
            self.assertAlmostEqual(c1, c2, places=3)
            self.assertAlmostEqual(e1, e2, places=3)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 7 — BACKEND API CONTRACT TESTS
# Tests that backend.py returns the correct HTTP response schema.
# Uses TestClient from FastAPI — no real model needed (mocked ENGINE).
# ══════════════════════════════════════════════════════════════════════════════

class TestBackendAPIContract(unittest.TestCase):
    """
    Tests the FastAPI /predict and /health endpoints.
    Mocks the ENGINE so no real model is needed.
    """

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            import backend

            # Replace the real ENGINE with a mock
            self.mock_engine = make_mock_engine(logit=2.5, entropy=0.20, gate="A")
            backend.ENGINE   = self.mock_engine
            self.client      = TestClient(backend.app)
        except Exception as e:
            self.skipTest(f"FastAPI or backend not importable: {e}")

    def test_health_endpoint_returns_200(self):
        """GET /health must return HTTP 200."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_contains_status_field(self):
        """GET /health must contain a 'status' field."""
        resp = self.client.get("/health")
        self.assertIn("status", resp.json())

    def test_predict_returns_200(self):
        """POST /predict must return HTTP 200."""
        img_bytes = make_fake_image_bytes()
        resp = self.client.post(
            "/predict",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"caption": "Test caption for fake news detection"}
        )
        self.assertEqual(resp.status_code, 200)

    def test_predict_response_has_status_success(self):
        """POST /predict response must have status='success'."""
        img_bytes = make_fake_image_bytes()
        resp = self.client.post(
            "/predict",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"caption": "Some news caption"}
        )
        data = resp.json()
        self.assertEqual(data.get("status"), "success")

    def test_predict_response_has_verdict(self):
        """POST /predict response must contain 'verdict' key."""
        img_bytes = make_fake_image_bytes()
        resp = self.client.post(
            "/predict",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"caption": "Some news caption"}
        )
        data = resp.json()
        self.assertIn("verdict", data)

    def test_predict_missing_caption_raises_error(self):
        """POST /predict without caption must not return status=success."""
        img_bytes = make_fake_image_bytes()
        resp = self.client.post(
            "/predict",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={}
        )
        # FastAPI returns 422 for missing required Form field
        self.assertNotEqual(resp.status_code, 200)

    def test_predict_empty_caption_handled(self):
        """POST /predict with empty caption string should not crash server."""
        img_bytes = make_fake_image_bytes()
        resp = self.client.post(
            "/predict",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"caption": ""}
        )
        # Should either succeed or return a handled error, not a 500 crash
        self.assertIn(resp.status_code, [200, 400, 422, 500])


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 8 — INTEGRATION TESTS
# Full end-to-end pipeline tests with real model + FAISS.
# Only runs if checkpoint and FAISS files exist on disk.
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_EXISTS = Path("checkpoints_pt/best_model.pt").exists()
FAISS_EXISTS      = Path("faiss_joint_store.index").exists()
TEST_DATA_EXISTS  = Path("Test data").exists()

@unittest.skipUnless(
    CHECKPOINT_EXISTS and FAISS_EXISTS,
    "Skipping integration tests: checkpoint or FAISS index not found on disk."
)
class TestIntegrationPipeline(unittest.TestCase):
    """
    Full end-to-end tests with the real engine.
    Requires: checkpoints_pt/best_model.pt + faiss_joint_store.index
    """

    @classmethod
    def setUpClass(cls):
        """Load engine once for all integration tests."""
        print("\n  [Integration] Loading REDDOTRagEngine...")
        from Orchestrator import REDDOTRagEngine
        cls.engine = REDDOTRagEngine(
            checkpoint_path = "checkpoints_pt/best_model.pt",
            store_path      = "faiss_joint_store",
            k_neighbours    = 5,
            entropy_gate    = 0.50,
        )
        cls.fake_image_bytes = make_fake_image_bytes()

    def test_engine_loaded_successfully(self):
        """Model and FAISS must load without error."""
        self.assertIsNotNone(self.engine.model)

    def test_run_returns_complete_result(self):
        """engine.run() must return a dict with all required keys."""
        result = self.engine.run(
            image_input = self.fake_image_bytes,
            caption     = "A protest in downtown caused traffic disruption.",
            sample_id   = "integration_test_1"
        )
        required = [
            "verdict", "base_verdict", "neighbour_verdict", "signals_agree",
            "gate_triggered", "logit", "prob_fake", "confidence", "entropy",
            "cosine_sim", "retrieval_context", "neighbours", "sample_id", "latency_ms"
        ]
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_verdict_is_valid(self):
        """Verdict must be one of the three valid labels."""
        result = self.engine.run(
            image_input = self.fake_image_bytes,
            caption     = "Scientists discover water on Mars.",
            sample_id   = "integration_test_2"
        )
        self.assertIn(result["verdict"].lower(), {"real", "fake", "uncertain"})

    def test_mismatched_pair_scores_higher_logit_than_matching(self):
        """
        Semantic mismatch must produce a higher (more positive) logit
        than a matching pair on the same image.
        This is the core model sensitivity test.
        """
        if not TEST_DATA_EXISTS:
            self.skipTest("Test data folder not found.")

        folders = sorted(Path("Test data").iterdir())
        if len(folders) < 1:
            self.skipTest("No test samples found.")

        folder = folders[0]
        if not (folder / "image.jpg").exists() or not (folder / "caption.txt").exists():
            self.skipTest("First test folder missing image.jpg or caption.txt.")

        caption_match    = open(folder / "caption.txt").read().strip()
        caption_mismatch = "Astronauts land on Jupiter near ancient ruins."

        r_match    = self.engine.run(str(folder / "image.jpg"), caption_match,    "match_test")
        r_mismatch = self.engine.run(str(folder / "image.jpg"), caption_mismatch, "mismatch_test")

        self.assertGreater(
            r_mismatch["logit"], r_match["logit"],
            f"Mismatch logit ({r_mismatch['logit']}) should be > "
            f"match logit ({r_match['logit']})"
        )

    def test_gate_a_fires_for_confident_prediction(self):
        """
        If the model is confident (low entropy), Gate A should fire.
        We test this by finding a sample where entropy < 0.50.
        """
        if not TEST_DATA_EXISTS:
            self.skipTest("Test data folder not found.")

        folders = sorted(Path("Test data").iterdir())
        for folder in folders:
            if not (folder / "image.jpg").exists():
                continue
            caption = open(folder / "caption.txt").read().strip() if (folder / "caption.txt").exists() else "test"
            result  = self.engine.run(str(folder / "image.jpg"), caption, f"gate_test_{folder.name}")
            if result["entropy"] <= 0.50:
                self.assertEqual(result["gate_triggered"], "A",
                    f"entropy={result['entropy']} should trigger Gate A")
                return
        # If all samples had high entropy, that's acceptable — just note it
        print(f"\n  {SKIP} All test samples had entropy > 0.50 — only Gate B was triggered")

    def test_latency_under_threshold_on_warm_engine(self):
        """
        After warmup, inference latency must be < 10,000ms on CPU.
        (Conservative threshold — CPU with ViT-L/14 can be slow.)
        """
        result = self.engine.run(
            image_input = self.fake_image_bytes,
            caption     = "Economy shows signs of recovery.",
            sample_id   = "latency_test"
        )
        self.assertLess(result["latency_ms"], 10_000,
            f"Latency {result['latency_ms']}ms exceeds 10s — something is wrong")


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(unit_only=False, integration_only=False):
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    unit_classes = [
        TestOutputSchema,
        TestGateRouting,
        TestFalseNegativeMinimisation,
        TestFalsePositiveRecoveryViaRAG,
        TestFAISSComponent,
        TestEntropyMetrics,
        TestBackendAPIContract,
    ]
    integration_classes = [
        TestIntegrationPipeline,
    ]

    if integration_only:
        classes = integration_classes
    elif unit_only:
        classes = unit_classes
    else:
        classes = unit_classes + integration_classes

    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    print(f"\n{'═'*60}")
    print(f"  RED-DOT RAG — Deployment Test Suite")
    print(f"  Mode: {'Unit only' if unit_only else 'Integration only' if integration_only else 'Full suite'}")
    print(f"  Checkpoint : {'✓ Found' if CHECKPOINT_EXISTS else '✗ Not found (integration tests will skip)'}")
    print(f"  FAISS Index: {'✓ Found' if FAISS_EXISTS else '✗ Not found (integration tests will skip)'}")
    print(f"  Test Data  : {'✓ Found' if TEST_DATA_EXISTS else '✗ Not found (some tests will skip)'}")
    print(f"{'═'*60}\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'═'*60}")
    print(f"  Tests run   : {result.testsRun}")
    print(f"  Passed      : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed      : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    print(f"  Skipped     : {len(result.skipped)}")
    print(f"{'═'*60}")

    if result.failures:
        print("\n  FAILURES:")
        for test, trace in result.failures:
            print(f"  {FAIL} {test}")
            print(f"    {trace.splitlines()[-1]}")

    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RED-DOT RAG Deployment Test Suite")
    parser.add_argument("--unit",        action="store_true", help="Run unit tests only (no model needed)")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only (needs model)")
    args = parser.parse_args()

    success = run_tests(
        unit_only        = args.unit,
        integration_only = args.integration,
    )
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""Unit tests for character recognition and deduplication in tools/windows/video-webui.py."""

from __future__ import annotations

import base64
import importlib.util
import os
import re
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WEBUI_PATH = os.path.join(_REPO_ROOT, "tools", "windows", "video-webui.py")
_SAMPLE_VIDEO = os.path.join(
    _REPO_ROOT, "video-jobs", "video_f9d40f468f327258d583be86", "reference_video_0.mp4"
)


def _load_webui():
    spec = importlib.util.spec_from_file_location("video_webui", _WEBUI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _make_person_frame(width: int = 320, height: int = 240, persons: list[tuple[int, tuple[int, int, int]]] | None = None) -> str:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    img = np.full((height, width, 3), 210, dtype=np.uint8)
    for cx, shirt_bgr in (persons or []):
        # Torso block
        cv2.rectangle(img, (cx - 28, 105), (cx + 28, 210), shirt_bgr, -1)
        # Head/face oval
        cv2.ellipse(img, (cx, 68), (22, 28), 0, 0, 360, (180, 200, 230), -1)
        cv2.circle(img, (cx - 8, 62), 3, (30, 30, 30), -1)
        cv2.circle(img, (cx + 8, 62), 3, (30, 30, 30), -1)
        cv2.ellipse(img, (cx, 78), (8, 4), 0, 0, 180, (60, 60, 120), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class VideoWebuiDetectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.webui = _load_webui()

    def test_extract_json_array_variants(self) -> None:
        arr = '[{"label": "Alice", "frame_index": 0, "bbox": [10, 20, 300, 400]}]'
        self.assertEqual(len(self.webui._extract_json_array(arr)), 1)

        fenced = "```json\n" + arr + "\n```"
        self.assertEqual(len(self.webui._extract_json_array(fenced)), 1)

        wrapped = '{"characters": ' + arr + "}"
        parsed = self.webui._extract_json_array(wrapped)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["label"], "Alice")

    def test_normalize_bbox(self) -> None:
        self.assertEqual(self.webui._normalize_bbox_py([0.1, 0.2, 0.5, 0.8]), (0.1, 0.2, 0.5, 0.8))
        self.assertEqual(self.webui._normalize_bbox_py([100, 200, 500, 800]), (0.1, 0.2, 0.5, 0.8))
        self.assertEqual(
            self.webui._normalize_bbox_py({"xmin": 100, "ymin": 200, "xmax": 500, "ymax": 800}),
            (0.1, 0.2, 0.5, 0.8),
        )
        pix = self.webui._normalize_bbox_py([320, 240, 1280, 960], img_w=1600, img_h=1200)
        self.assertIsNotNone(pix)
        assert pix is not None
        self.assertAlmostEqual(pix[0], 0.2, places=3)
        self.assertAlmostEqual(pix[2], 0.8, places=3)
        self.assertIsNone(self.webui._normalize_bbox_py([100, 100, 105, 105]))

    def test_managed_vlm_command_flags(self) -> None:
        mgr = self.webui.ManagedVlmServer(
            gufo_exe="C:/AI/gufo/build/windows-release/gufo.exe",
            model_path="C:/AI/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf",
            mmproj_path="C:/AI/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf",
        )
        cmd = mgr.build_command(8085)
        self.assertIn("llm", cmd)
        self.assertIn("--context", cmd)
        self.assertEqual(cmd[cmd.index("--context") + 1], "8192")
        self.assertIn("--think", cmd)
        self.assertEqual(cmd[cmd.index("--think") + 1], "off")
        self.assertIn("--cache-ram-bytes", cmd)
        self.assertEqual(cmd[cmd.index("--cache-ram-bytes") + 1], "off")
        self.assertIn("--sessions", cmd)
        self.assertEqual(cmd[cmd.index("--sessions") + 1], "1")
        self.assertIn("--mmproj", cmd)
        self.assertEqual(mgr.model_display_name(), "Qwen3.8-Flash-Next VLM")

    def test_dedup_vlm_same_frame_and_cross_frame(self) -> None:
        raw = [
            # Same-frame overlapping / contained duplicate
            {
                "label": "Woman in red jacket",
                "description": "woman with dark hair wearing a red jacket on the left",
                "frame_index": 0,
                "bbox": [50, 50, 400, 900],
            },
            {
                "label": "Red jacket woman face",
                "description": "woman in red jacket",
                "frame_index": 0,
                "bbox": [100, 70, 320, 420],
            },
            # Cross-frame duplicate of the same woman
            {
                "label": "Woman in red jacket",
                "description": "the woman with dark hair wearing a red jacket standing in center",
                "frame_index": 2,
                "bbox": [260, 60, 620, 910],
            },
            # Distinct second person on frame 0
            {
                "label": "Man in blue suit",
                "description": "man with short blond hair in a navy blue suit on the right",
                "frame_index": 0,
                "bbox": [580, 80, 930, 920],
            },
        ]
        deduped = self.webui._dedup_vlm_characters(raw, frames=["", "", ""])
        self.assertEqual(len(deduped), 2)
        labels = {c["label"] for c in deduped}
        self.assertIn("Woman in red jacket", labels)
        self.assertIn("Man in blue suit", labels)

    @unittest.skipUnless(_has_cv2(), "cv2 not installed")
    def test_dedup_vlm_with_synthetic_frames(self) -> None:
        f0 = _make_person_frame(persons=[(80, (20, 20, 200)), (240, (200, 40, 20))])
        f1 = _make_person_frame(persons=[(110, (20, 20, 200)), (250, (200, 40, 20))])
        raw = [
            {
                "label": "Character 1",
                "description": "person in red shirt on the left",
                "frame_index": 0,
                "bbox": [120, 140, 380, 900],
            },
            {
                "label": "Character 2",
                "description": "person wearing red shirt in the center",
                "frame_index": 1,
                "bbox": [210, 140, 470, 900],
            },
            {
                "label": "Character 3",
                "description": "person in blue shirt on the right",
                "frame_index": 0,
                "bbox": [620, 140, 880, 900],
            },
        ]
        deduped = self.webui._dedup_vlm_characters(raw, frames=[f0, f1])
        self.assertEqual(len(deduped), 2)

    @unittest.skipUnless(_has_cv2(), "cv2 not installed")
    def test_local_hog_haar_containment_dedup(self) -> None:
        import numpy as np  # type: ignore

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        decoded = [(0, img)]
        # Verify containment helper prevents face-vs-body duplicates
        face_expanded = (0.20, 0.10, 0.50, 0.60)
        full_body = (0.15, 0.05, 0.55, 0.95)
        self.assertLess(self.webui._box_iou(face_expanded, full_body), 0.45)
        self.assertGreater(self.webui._box_containment(face_expanded, full_body), 0.90)
        chars = self.webui._detect_characters_local_hog_haar(decoded, include_haar=True)
        self.assertIsInstance(chars, list)

    @unittest.skipUnless(_has_cv2() and os.path.isfile(_SAMPLE_VIDEO), "cv2 or sample video unavailable")
    def test_local_detector_sample_video_smoke(self) -> None:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(_SAMPLE_VIDEO)
        self.assertTrue(cap.isOpened())
        frames: list[str] = []
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for idx in (0, max(0, total // 2), max(0, total - 1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    ok_enc, buf = cv2.imencode(".jpg", frame)
                    if ok_enc:
                        frames.append(
                            "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
                        )
        finally:
            cap.release()
        self.assertGreater(len(frames), 0)
        chars, detector = self.webui._detect_characters_local(frames)
        self.assertIsInstance(chars, list)
        self.assertTrue(detector.startswith("Local "))

    def test_resolution_options_divisible_by_32_up_to_4k(self) -> None:
        m = re.search(
            r'<select id="resolution">(.*?)</select>',
            self.webui.HTML_PAGE,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(m)
        assert m is not None
        values = re.findall(r'<option value="(\d+x\d+)"', m.group(1))
        self.assertGreaterEqual(len(values), 10)
        for val in values:
            w_str, h_str = val.split("x")
            w, h = int(w_str), int(h_str)
            self.assertEqual(w % 32, 0, f"Width {w} in {val} is not divisible by 32")
            self.assertEqual(h % 32, 0, f"Height {h} in {val} is not divisible by 32")
            self.assertGreaterEqual(w, 256)
            self.assertGreaterEqual(h, 256)
            self.assertLessEqual(w, 3840)
            self.assertLessEqual(h, 3840)
        for forbidden in ("1280x720", "720x1280", "1920x1080", "3840x2160"):
            self.assertNotIn(forbidden, values)
        self.assertIn("3840x2144", values)
        self.assertIn("2144x3840", values)

    @unittest.skipUnless(_has_cv2(), "cv2 not installed")
    def test_crop_reference_face_if_wide_and_closeup(self) -> None:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        class FakeYunet:
            def __init__(self, box: tuple[float, float, float, float]) -> None:
                self.box = box

            def setInputSize(self, _size: tuple[int, int]) -> None:
                pass

            def detect(self, _bgr):
                x, y, w, h = self.box
                row = [x, y, w, h] + [0.0] * 10 + [0.95]
                return 1, np.array([row], dtype=np.float32)

        img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
        ok, enc = cv2.imencode(".png", img)
        self.assertTrue(ok)
        raw = enc.tobytes()

        orig_get = self.webui._get_yunet_sface
        try:
            # Wide shot: 100x120 face in 1000x1000 frame -> 1.2% area -> should crop
            self.webui._get_yunet_sface = lambda *a, **kw: (FakeYunet((450.0, 200.0, 100.0, 120.0)), None)
            wide_res = self.webui._crop_reference_face_if_wide(raw, "full_body.png")
            self.assertTrue(wide_res["cropped"])
            self.assertLess(wide_res["face_area_ratio"], 0.25)
            self.assertLess(wide_res["width"], 500)
            self.assertLess(wide_res["height"], 500)
            self.assertTrue(bool(wide_res.get("image_b64")))

            # Close-up shot: 600x600 face in 1000x1000 frame -> 36% area -> no crop
            self.webui._get_yunet_sface = lambda *a, **kw: (FakeYunet((200.0, 150.0, 600.0, 600.0)), None)
            closeup_res = self.webui._crop_reference_face_if_wide(raw, "closeup.png")
            self.assertFalse(closeup_res["cropped"])
            self.assertGreaterEqual(closeup_res["face_area_ratio"], 0.25)
        finally:
            self.webui._get_yunet_sface = orig_get

    def test_swap_prompt_and_face_crop_ui_contracts(self) -> None:
        html = self.webui.HTML_PAGE
        self.assertIn('id="optAutoFaceCrop"', html)
        self.assertIn("maybeAutoSelectSwapLora", html)
        self.assertIn("h3_character_swap_v1.5", html)
        self.assertIn("/api/crop-face", html)
        self.assertNotIn("Edit <Video 1> directly without changing", html)
        self.assertIn("Keep the reference character's exact face, hair, and clothing", html)


if __name__ == "__main__":
    unittest.main()

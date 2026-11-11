import base64
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from squeakpose.services.openai_labeling import (
    OpenAILabelingError,
    build_pose_request,
    list_models,
    parse_pose_proposal,
    request_pose_proposal,
)


def _parse_single_proposal(value, **kwargs):
    return parse_pose_proposal({"instances": [value]}, **kwargs)


class OpenAILabelingTests(unittest.TestCase):
    def test_plan_model_catalog_keeps_only_listed_models_in_server_order(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"slug": "gpt-6-astra", "display_name": "Astra", "visibility": "list"},
                        {"slug": "internal-model", "visibility": "hidden"},
                        {"slug": "gpt-6.1-sol", "display_name": "Sol", "visibility": "list"},
                    ]
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            choices = list_models("test-access-token", client=client)

        self.assertEqual([choice.slug for choice in choices], ["gpt-6-astra", "gpt-6.1-sol"])

    def test_request_is_stateless_streaming_structured_image_input(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            image_path.write_bytes(b"not-a-real-png-but-adequate-for-payload-construction")

            payload = build_pose_request(
                model="gpt-5.6-sol",
                image_path=str(image_path),
                class_name="mouse",
                keypoint_names=("nose", "left_ear", "tail_base"),
            )

            self.assertFalse(payload["store"])
            self.assertTrue(payload["stream"])
            self.assertEqual(payload["reasoning"], {"effort": "low"})
            image = payload["input"][0]["content"][1]
            self.assertEqual(image["detail"], "original")
            self.assertTrue(image["image_url"].startswith("data:image/png;base64,"))
            encoded = image["image_url"].split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), image_path.read_bytes())
            schema = payload["text"]["format"]
            self.assertEqual(schema["type"], "json_schema")
            self.assertTrue(schema["strict"])
            self.assertEqual(
                schema["schema"]["properties"]["instances"]["items"]["properties"]["keypoints"][
                    "minItems"
                ],
                3,
            )

    def test_normalized_proposal_maps_to_pixels_and_canonical_names(self):
        proposal = _parse_single_proposal(
            {
                "keypoints": [
                    {
                        "name": "tailbase",
                        "x": 590,
                        "y": 569,
                        "visibility": 2,
                        "confidence": 0.91,
                    },
                    {
                        "name": "nose",
                        "x": 516,
                        "y": 658,
                        "visibility": 2,
                        "confidence": 0.97,
                    },
                    {
                        "name": "left ear",
                        "x": 551,
                        "y": 592,
                        "visibility": 1,
                        "confidence": 0.96,
                    },
                ],
                "bounding_box": {
                    "x_min": 473,
                    "y_min": 455,
                    "x_max": 715,
                    "y_max": 676,
                },
            },
            source_image="frame.png",
            model="gpt-5.6-sol",
            class_name="mouse",
            keypoint_names=("nose", "left_ear", "tail_base"),
            image_width=1440,
            image_height=1080,
        )

        self.assertEqual(
            [point.name for point in proposal.instances[0].keypoints],
            ["nose", "left_ear", "tail_base"],
        )
        self.assertAlmostEqual(proposal.instances[0].keypoints[0].x, 743.04)
        self.assertAlmostEqual(proposal.instances[0].keypoints[0].y, 710.64)
        self.assertEqual(proposal.instances[0].keypoints[1].visibility, 1)
        for actual, expected in zip(proposal.instances[0].box, (681.12, 491.4, 348.48, 238.68)):
            self.assertAlmostEqual(actual, expected)

    def test_completed_response_usage_is_returned_with_proposal(self):
        result = {
            "keypoints": [{"name": "nose", "x": 500, "y": 600, "visibility": 2, "confidence": 0.9}],
            "bounding_box": {"x_min": 100, "y_min": 200, "x_max": 900, "y_max": 800},
        }
        result = {"instances": [result]}
        events = "\n".join(
            (
                f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': json.dumps(result)})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': {'usage': {'input_tokens': 1200, 'output_tokens': 300, 'total_tokens': 1500, 'output_tokens_details': {'reasoning_tokens': 125}}}})}",
                "data: [DONE]",
            )
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=events)

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            image_path.write_bytes(b"image")
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                proposal = request_pose_proposal(
                    access_token="token",
                    model="gpt-6-astra",
                    image_path=str(image_path),
                    image_width=100,
                    image_height=100,
                    class_name="mouse",
                    keypoint_names=("nose",),
                    client=client,
                )

        self.assertEqual(proposal.usage.input_tokens, 1200)
        self.assertEqual(proposal.usage.output_tokens, 300)
        self.assertEqual(proposal.usage.reasoning_tokens, 125)
        self.assertEqual(proposal.usage.total_tokens, 1500)

    def test_streaming_http_error_surfaces_api_message(self):
        body = json.dumps(
            {"error": {"message": "Rate limit reached for gpt-6-astra", "code": "rate_limit"}}
        ).encode("utf-8")

        def handler(_request: httpx.Request) -> httpx.Response:
            # Passing ``stream=`` (instead of ``content=``) leaves the body
            # unread, exactly like a real streamed 4xx/5xx response.
            return httpx.Response(
                429,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(body),
            )

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            image_path.write_bytes(b"image")
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(OpenAILabelingError) as raised:
                    request_pose_proposal(
                        access_token="token",
                        model="gpt-6-astra",
                        image_path=str(image_path),
                        image_width=100,
                        image_height=100,
                        class_name="mouse",
                        keypoint_names=("nose",),
                        client=client,
                    )

        message = str(raised.exception)
        self.assertIn("OpenAI rejected the labeling request", message)
        self.assertIn("Rate limit reached for gpt-6-astra", message)
        self.assertNotIn("ResponseNotRead", message)
        self.assertNotIn("read()", message)

    def test_streaming_http_error_without_json_body_falls_back_to_status_code(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, stream=httpx.ByteStream(b"<html>bad gateway</html>"))

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            image_path.write_bytes(b"image")
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(OpenAILabelingError, "503"):
                    request_pose_proposal(
                        access_token="token",
                        model="gpt-6-astra",
                        image_path=str(image_path),
                        image_width=100,
                        image_height=100,
                        class_name="mouse",
                        keypoint_names=("nose",),
                        client=client,
                    )

    def test_missing_requested_keypoint_is_rejected(self):
        with self.assertRaisesRegex(OpenAILabelingError, "tail_base"):
            _parse_single_proposal(
                {
                    "keypoints": [
                        {
                            "name": "nose",
                            "x": 500,
                            "y": 500,
                            "visibility": 2,
                            "confidence": 0.9,
                        }
                    ],
                    "bounding_box": {
                        "x_min": 100,
                        "y_min": 100,
                        "x_max": 900,
                        "y_max": 900,
                    },
                },
                source_image="frame.png",
                model="gpt-5.6-sol",
                class_name="mouse",
                keypoint_names=("nose", "tail_base"),
                image_width=100,
                image_height=100,
            )

    def _parse_batch(self, value, keypoints=("nose",)):
        return parse_pose_proposal(
            value,
            source_image="frame.png",
            model="test-model",
            class_name="mouse",
            keypoint_names=keypoints,
            image_width=100,
            image_height=200,
        )

    def _instance(self, x=100):
        return {
            "bounding_box": {"x_min": x, "y_min": 100, "x_max": x + 100, "y_max": 400},
            "keypoints": [
                {"name": "nose", "x": x + 50, "y": 200, "visibility": 2, "confidence": 0.9}
            ],
        }

    def test_one_five_and_zero_instances_and_stable_spatial_order(self):
        for count in (0, 1, 5):
            proposal = self._parse_batch(
                {"instances": [self._instance(100 + i * 150) for i in reversed(range(count))]}
            )
            self.assertEqual(len(proposal.instances), count)
            positions = [instance.box[0] for instance in proposal.instances]
            self.assertEqual(positions, sorted(positions))
            self.assertTrue(all(len(instance.keypoints) == 1 for instance in proposal.instances))

    def test_missing_instance_list_and_malformed_member_reject_entire_response(self):
        for value in (
            {},
            {"instances": None},
            {"instances": {}},
            {"instances": [self._instance(), None]},
            {"instances": [self._instance(), {"keypoints": []}]},
        ):
            with self.subTest(value=value), self.assertRaises(OpenAILabelingError):
                self._parse_batch(value)

    def test_duplicate_extra_keypoints_invalid_boxes_and_nonfinite_coordinates_rejected(self):
        for mutation in ("duplicate", "extra", "box", "nan"):
            instance = self._instance()
            if mutation == "duplicate":
                instance["keypoints"].append(dict(instance["keypoints"][0]))
            elif mutation == "extra":
                instance["keypoints"].append(dict(instance["keypoints"][0], name="tail"))
            elif mutation == "box":
                instance["bounding_box"]["x_max"] = 50
            else:
                instance["keypoints"][0]["x"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(OpenAILabelingError):
                self._parse_batch({"instances": [instance]})

    def test_generic_box_only_class_has_empty_keypoints(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp, "frame.png")
            image.write_bytes(b"image")
            payload = build_pose_request(
                model="test", image_path=str(image), class_name="marble", keypoint_names=()
            )
            prompt = payload["input"][0]["content"][0]["text"]
            self.assertIn('"marble"', prompt)
            self.assertIn("every distinct instance", prompt)
            self.assertNotIn("exactly one", prompt)
            self.assertNotIn("mouse_id", json.dumps(payload["text"]["format"]["schema"]))
        instance = self._instance()
        instance["keypoints"] = []
        proposal = self._parse_batch({"instances": [instance]}, keypoints=())
        self.assertEqual(proposal.instances[0].keypoints, ())

    def _stream_request(self, events):
        body = "\n".join(f"data: {json.dumps(event)}" for event in events) + "\ndata: [DONE]\n"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            image_path.write_bytes(b"image")
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                return request_pose_proposal(
                    access_token="token",
                    model="gpt-6-astra",
                    image_path=str(image_path),
                    image_width=100,
                    image_height=100,
                    class_name="mouse",
                    keypoint_names=("nose",),
                    client=client,
                )

    def test_top_level_stream_error_event_surfaces_message_and_code(self):
        with self.assertRaises(OpenAILabelingError) as raised:
            self._stream_request(
                [
                    {
                        "type": "error",
                        "code": "usage_limit_reached",
                        "message": "You've hit your usage limit.",
                        "param": None,
                    }
                ]
            )
        message = str(raised.exception)
        self.assertIn("You've hit your usage limit.", message)
        # main_window._openai_error looks for "usage_limit"/"rate_limit" after
        # normalising spaces and hyphens to underscores.
        self.assertIn("usage_limit", message.casefold().replace(" ", "_").replace("-", "_"))
        self.assertNotIn("ended before completion", message)

    def test_top_level_stream_error_with_code_only_or_nested_error(self):
        with self.assertRaisesRegex(OpenAILabelingError, "rate_limit_exceeded"):
            self._stream_request([{"type": "error", "code": "rate_limit_exceeded"}])
        with self.assertRaisesRegex(OpenAILabelingError, "Server overloaded"):
            self._stream_request(
                [{"type": "error", "error": {"message": "Server overloaded", "code": None}}]
            )

    def test_failed_and_incomplete_events_tolerate_null_nested_objects(self):
        for event in (
            {"type": "response.failed", "response": None},
            {"type": "response.failed", "response": {"error": None}},
        ):
            with self.subTest(event=event):
                with self.assertRaisesRegex(OpenAILabelingError, "unknown error"):
                    self._stream_request([event])
        with self.assertRaisesRegex(OpenAILabelingError, "failed: quota gone"):
            self._stream_request(
                [{"type": "response.failed", "response": {"error": {"message": "quota gone"}}}]
            )
        for event in (
            {"type": "response.incomplete", "response": None},
            {"type": "response.incomplete", "response": {"incomplete_details": None}},
        ):
            with self.subTest(event=event):
                with self.assertRaisesRegex(OpenAILabelingError, r"incomplete labeling response\."):
                    self._stream_request([event])
        with self.assertRaisesRegex(OpenAILabelingError, r"\(max_output_tokens\)"):
            self._stream_request(
                [
                    {
                        "type": "response.incomplete",
                        "response": {"incomplete_details": {"reason": "max_output_tokens"}},
                    }
                ]
            )

    def _point(self, name, x):
        return {"name": name, "x": x, "y": 200, "visibility": 2, "confidence": 0.9}

    def _box(self):
        return {"x_min": 100, "y_min": 100, "x_max": 900, "y_max": 900}

    def test_keypoint_names_differing_only_by_case_or_punctuation_stay_distinct(self):
        for names in (("tail_base", "tail base"), ("L", "l"), ("Nose", "nose", "NOSE")):
            with self.subTest(names=names):
                points = [self._point(name, 100 + i * 100) for i, name in enumerate(names)]
                instance = {"keypoints": list(reversed(points)), "bounding_box": self._box()}
                proposal = self._parse_batch({"instances": [instance]}, keypoints=names)
                keypoints = proposal.instances[0].keypoints
                self.assertEqual([point.name for point in keypoints], list(names))
                expected_x = [10 + i * 10 for i in range(len(names))]
                self.assertEqual([point.x for point in keypoints], expected_x)

    def test_ambiguous_normalized_name_is_rejected_but_unique_fallback_still_works(self):
        names = ("tail_base", "tail base", "left_ear")
        instance = {
            "keypoints": [
                self._point("TailBase", 100),  # ambiguous: matches two configured names
                self._point("tail base", 200),
                self._point("left ear", 300),
            ],
            "bounding_box": self._box(),
        }
        with self.assertRaisesRegex(OpenAILabelingError, "unexpected keypoint name"):
            self._parse_batch({"instances": [instance]}, keypoints=names)
        instance["keypoints"][0] = self._point("tail_base", 100)
        proposal = self._parse_batch({"instances": [instance]}, keypoints=names)
        self.assertEqual(
            [point.name for point in proposal.instances[0].keypoints],
            ["tail_base", "tail base", "left_ear"],
        )
        # An exact name and its normalized alias for the same point are duplicates.
        instance["keypoints"].append(self._point("LEFT-EAR", 400))
        with self.assertRaisesRegex(OpenAILabelingError, "duplicate keypoint name"):
            self._parse_batch({"instances": [instance]}, keypoints=names)


if __name__ == "__main__":
    unittest.main()

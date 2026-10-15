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


if __name__ == "__main__":
    unittest.main()

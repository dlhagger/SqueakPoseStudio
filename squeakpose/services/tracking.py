"""Shared tracking configuration for inference and downstream analysis."""

from __future__ import annotations

from dataclasses import dataclass

TRACKER_AUTO = "auto"
TRACKER_BYTETRACK = "bytetrack"
TRACKER_BOTSORT = "botsort"
TRACKER_NONE = "none"
TRACKER_CHOICES = (TRACKER_AUTO, TRACKER_BYTETRACK, TRACKER_BOTSORT)
DEFAULT_TRACKER_PROFILE = "fixed_camera_v1"
MIN_EXPECTED_ANIMALS = 1
MAX_EXPECTED_ANIMALS = 32


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Validated, reproducible tracker settings for one source video."""

    expected_animal_count: int
    requested_tracker: str
    resolved_tracker: str
    tracker_profile: str = DEFAULT_TRACKER_PROFILE
    enabled: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "expected_animal_count": self.expected_animal_count,
            "requested_tracker": self.requested_tracker,
            "resolved_tracker": self.resolved_tracker,
            "tracker_profile": self.tracker_profile,
        }


def normalize_tracker_choice(value: object, *, default: str = TRACKER_AUTO) -> str:
    """Normalize user- or manifest-provided tracker names."""
    normalized = str(value or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "auto": TRACKER_AUTO,
        "byte": TRACKER_BYTETRACK,
        "bytetrack": TRACKER_BYTETRACK,
        "botsort": TRACKER_BOTSORT,
    }
    return aliases.get(normalized, default)


def resolve_tracking_config(
    expected_animal_count: int = 1,
    requested_tracker: str = TRACKER_AUTO,
    *,
    enabled: bool = True,
    tracker_profile: str = DEFAULT_TRACKER_PROFILE,
) -> TrackingConfig:
    """Validate settings and resolve Auto using the app's centralized policy."""
    count = int(expected_animal_count)
    if not MIN_EXPECTED_ANIMALS <= count <= MAX_EXPECTED_ANIMALS:
        raise ValueError(
            f"expected_animal_count must be between {MIN_EXPECTED_ANIMALS} "
            f"and {MAX_EXPECTED_ANIMALS}"
        )
    requested = normalize_tracker_choice(requested_tracker, default="")
    if requested not in TRACKER_CHOICES:
        raise ValueError(f"unsupported tracker: {requested_tracker!r}")
    profile = str(tracker_profile or DEFAULT_TRACKER_PROFILE).strip()
    if not profile:
        raise ValueError("tracker_profile must not be empty")
    resolved = requested
    if not enabled:
        resolved = TRACKER_NONE
    elif requested == TRACKER_AUTO:
        resolved = TRACKER_BYTETRACK if count == 1 else TRACKER_BOTSORT
    return TrackingConfig(
        expected_animal_count=count,
        requested_tracker=requested,
        resolved_tracker=resolved,
        tracker_profile=profile,
        enabled=bool(enabled),
    )


__all__ = [
    "DEFAULT_TRACKER_PROFILE",
    "MAX_EXPECTED_ANIMALS",
    "MIN_EXPECTED_ANIMALS",
    "TRACKER_AUTO",
    "TRACKER_BOTSORT",
    "TRACKER_BYTETRACK",
    "TRACKER_CHOICES",
    "TRACKER_NONE",
    "TrackingConfig",
    "normalize_tracker_choice",
    "resolve_tracking_config",
]

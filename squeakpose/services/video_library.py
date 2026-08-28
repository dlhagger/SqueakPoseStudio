"""Safe project video-link management without copying source videos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
)


@dataclass(frozen=True, slots=True)
class VideoLibraryEntry:
    name: str
    path: str
    is_link: bool
    target: str
    target_exists: bool


def _require_library_child(videos_dir: str, name: str) -> str:
    clean_name = str(name).strip()
    if not clean_name or clean_name in {".", ".."} or os.path.basename(clean_name) != clean_name:
        raise ValueError("Video link names must be a single file name")
    directory = os.path.abspath(videos_dir)
    path = os.path.abspath(os.path.join(directory, clean_name))
    if os.path.commonpath((directory, path)) != directory or path == directory:
        raise ValueError("Video link path escapes the project videos folder")
    return path


def _legacy_macos_alias_target(videos_dir: str, path: str) -> str:
    """Resolve a copied Finder alias when its original Mac path is stale."""
    try:
        if os.path.getsize(path) > 64 * 1024:
            return ""
        with open(path, "rb") as handle:
            payload = handle.read(64 * 1024)
    except OSError:
        return ""
    if not payload.startswith(b"XSym\n"):
        return ""
    text = payload.decode("utf-8", errors="ignore")
    recorded = next(
        (line.strip() for line in text.splitlines() if line.strip().startswith("/")), ""
    )
    if not recorded:
        return ""
    if os.path.isfile(recorded):
        return os.path.abspath(recorded)

    # Finder aliases copied with a project retain the old absolute prefix. In
    # this project layout the source video remains beside the project folder,
    # under its animal/session directory. Require the expected filename and a
    # unique nearby match rather than searching arbitrary filesystem roots.
    filename = os.path.basename(recorded)
    if filename.casefold() != os.path.basename(path).casefold():
        return ""
    import_root = Path(videos_dir).resolve().parent.parent
    session_name = Path(recorded).parent.name
    direct = import_root / session_name / filename
    if direct.is_file():
        return str(direct)
    matches = [
        candidate
        for candidate in import_root.glob(f"*/{filename}")
        if candidate.is_file() and candidate.parent != Path(videos_dir)
    ]
    return str(matches[0]) if len(matches) == 1 else ""


def list_project_videos(videos_dir: str) -> list[VideoLibraryEntry]:
    """List video files and links, including broken video links."""
    directory = os.path.abspath(videos_dir)
    if not os.path.isdir(directory):
        return []
    entries: list[VideoLibraryEntry] = []
    with os.scandir(directory) as iterator:
        for item in iterator:
            if item.name.startswith(".") or not item.name.lower().endswith(VIDEO_EXTENSIONS):
                continue
            path = os.path.abspath(item.path)
            is_link = item.is_symlink()
            alias_target = ""
            if is_link:
                raw_target = os.readlink(path)
                target = os.path.abspath(os.path.join(directory, raw_target))
                target_exists = os.path.isfile(path)
            elif item.is_file(follow_symlinks=False):
                alias_target = _legacy_macos_alias_target(directory, path)
                target = alias_target or path
                target_exists = os.path.isfile(target)
            else:
                continue
            entries.append(
                VideoLibraryEntry(
                    name=item.name,
                    path=target if alias_target else path,
                    is_link=is_link or bool(alias_target),
                    target=target,
                    target_exists=target_exists,
                )
            )
    return sorted(entries, key=lambda entry: entry.name.casefold())


def resolve_project_video_paths(videos_dir: str, paths: list[str]) -> list[str]:
    """Resolve project-library links and imported Finder aliases in one scan."""
    directory = os.path.abspath(videos_dir)
    resolved_by_path: dict[str, str] = {}
    for entry in list_project_videos(directory):
        resolved = os.path.abspath(entry.target if entry.target_exists else entry.path)
        candidates = (
            os.path.join(directory, entry.name),
            entry.path,
            entry.target,
        )
        for candidate in candidates:
            if candidate:
                resolved_by_path[os.path.abspath(candidate)] = resolved
    return [
        resolved_by_path.get(os.path.abspath(os.fspath(path)), os.path.abspath(os.fspath(path)))
        for path in paths
    ]


def _available_link_name(videos_dir: str, preferred_name: str) -> str:
    stem, extension = os.path.splitext(preferred_name)
    candidate = preferred_name
    suffix = 2
    while os.path.lexists(_require_library_child(videos_dir, candidate)):
        candidate = f"{stem} {suffix}{extension}"
        suffix += 1
    return candidate


def add_video_links(videos_dir: str, source_paths: list[str]) -> list[VideoLibraryEntry]:
    """Create absolute symlinks and choose non-conflicting destination names."""
    directory = os.path.abspath(videos_dir)
    os.makedirs(directory, exist_ok=True)
    created: list[VideoLibraryEntry] = []
    for source_path in source_paths:
        source = os.path.abspath(os.fspath(source_path))
        if not os.path.isfile(source):
            raise ValueError(f"Video source is not a readable file: {source}")
        if not source.lower().endswith(VIDEO_EXTENSIONS):
            raise ValueError(f"Unsupported video filename: {os.path.basename(source)}")
        source = os.path.realpath(source)
        existing = next(
            (
                entry
                for entry in list_project_videos(directory)
                if os.path.realpath(entry.path) == source
            ),
            None,
        )
        if existing is not None:
            continue
        name = _available_link_name(directory, os.path.basename(source))
        destination = _require_library_child(directory, name)
        os.symlink(source, destination)
        created.append(VideoLibraryEntry(name, destination, True, source, target_exists=True))
    return created


def remove_video_link(videos_dir: str, name: str) -> None:
    """Remove a library symlink without ever deleting a regular video file."""
    path = _require_library_child(videos_dir, name)
    if not os.path.islink(path):
        raise ValueError("Only video links can be removed here; the original file was not changed")
    os.unlink(path)


def rename_video_link(videos_dir: str, old_name: str, new_name: str) -> str:
    """Rename a symlink while preserving its target."""
    source = _require_library_child(videos_dir, old_name)
    if not os.path.islink(source):
        raise ValueError("Only linked videos can be renamed here")
    clean_name = str(new_name).strip()
    if not os.path.splitext(clean_name)[1]:
        clean_name += os.path.splitext(old_name)[1]
    if not clean_name.lower().endswith(VIDEO_EXTENSIONS):
        raise ValueError("The link name must use a supported video extension")
    destination = _require_library_child(videos_dir, clean_name)
    if os.path.lexists(destination) and destination != source:
        raise FileExistsError(f"A video named '{clean_name}' already exists")
    os.rename(source, destination)
    return destination


def retarget_video_link(videos_dir: str, name: str, source_path: str) -> str:
    """Atomically change a symlink target without modifying either video."""
    link_path = _require_library_child(videos_dir, name)
    if not os.path.islink(link_path):
        raise ValueError("Only linked videos can have their source changed")
    source = os.path.realpath(os.path.abspath(os.fspath(source_path)))
    if not os.path.isfile(source):
        raise ValueError(f"Video source is not a readable file: {source}")
    if not source.lower().endswith(VIDEO_EXTENSIONS):
        raise ValueError(f"Unsupported video filename: {os.path.basename(source)}")
    temporary = _require_library_child(videos_dir, f".{name}.replacement")
    if os.path.lexists(temporary):
        raise FileExistsError(f"Temporary link already exists: {temporary}")
    try:
        os.symlink(source, temporary)
        os.replace(temporary, link_path)
    finally:
        if os.path.islink(temporary):
            os.unlink(temporary)
    return source

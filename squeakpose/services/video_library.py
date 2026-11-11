"""Safe project video-link management without copying source videos."""

from __future__ import annotations

import errno
import hashlib
import ntpath
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

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


# Samba/macOS "minshall-french" symlink files: a fixed 1067-byte text file that records
# the link target. SqueakPose writes them when the OS refuses real symlinks (Windows
# without administrator rights or Developer Mode, FAT/exFAT drives); macOS also treats
# them as symlinks on SMB shares.
XSYM_MAGIC = b"XSym\n"
XSYM_FILE_SIZE = 1067
_POINTER_READ_LIMIT = 64 * 1024
_WINDOWS_PRIVILEGE_NOT_HELD = 1314
_SYMLINK_UNSUPPORTED_ERRNOS = frozenset(
    {errno.EPERM, errno.EACCES, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}
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


def _xsym_recorded_target(path: str) -> str:
    """Return the target recorded in an XSym pointer file, or "" for any other file."""
    try:
        if os.path.getsize(path) > _POINTER_READ_LIMIT:
            return ""
        with open(path, "rb") as handle:
            payload = handle.read(_POINTER_READ_LIMIT)
    except OSError:
        return ""
    if not payload.startswith(XSYM_MAGIC):
        return ""
    lines = [line.strip() for line in payload.decode("utf-8", errors="ignore").splitlines()]
    # Layout: magic, 4-digit length, md5, target, padding. Older copies written by other
    # tools are not always exact, so fall back to the first absolute-looking line.
    if len(lines) >= 4 and lines[3]:
        return lines[3]
    return next((line for line in lines if line.startswith("/") or ntpath.isabs(line)), "")


def _is_pointer_file(path: str) -> bool:
    return os.path.isfile(path) and not os.path.islink(path) and bool(_xsym_recorded_target(path))


def _write_pointer_file(destination: str, target: str) -> None:
    encoded = target.encode("utf-8")
    digest = hashlib.md5(encoded, usedforsecurity=False).hexdigest().encode("ascii")
    body = XSYM_MAGIC + b"%04d\n%s\n%s\n" % (len(encoded), digest, encoded)
    with open(destination, "xb") as handle:
        handle.write(body.ljust(XSYM_FILE_SIZE, b" "))


def _symlink_unsupported(exc: OSError) -> bool:
    if isinstance(exc, FileExistsError):
        return False
    if getattr(exc, "winerror", None) == _WINDOWS_PRIVILEGE_NOT_HELD:
        return True
    return exc.errno in _SYMLINK_UNSUPPORTED_ERRNOS


def _create_link(source: str, destination: str) -> bool:
    """Link ``destination`` to ``source``; return False when a pointer file was written."""
    try:
        os.symlink(source, destination)
        return True
    except OSError as exc:
        if not _symlink_unsupported(exc):
            raise
    _write_pointer_file(destination, source)
    return False


def _is_library_link(path: str) -> bool:
    return os.path.islink(path) or _is_pointer_file(path)


def _legacy_macos_alias_target(videos_dir: str, path: str) -> str:
    """Resolve an XSym pointer file: SqueakPose's Windows fallback or a copied Mac alias."""
    recorded = _xsym_recorded_target(path)
    if not recorded:
        return ""
    if os.path.isfile(recorded):
        return os.path.abspath(recorded)

    # Finder aliases copied with a project retain the old absolute prefix. In
    # this project layout the source video remains beside the project folder,
    # under its animal/session directory. Require the expected filename and a
    # unique nearby match rather than searching arbitrary filesystem roots.
    recorded_path = PureWindowsPath(recorded) if "\\" in recorded else Path(recorded)
    filename = recorded_path.name
    if filename.casefold() != os.path.basename(path).casefold():
        return ""
    # Preserve the caller's absolute path spelling. On macOS, ``Path.resolve``
    # rewrites /var to /private/var, which made an otherwise valid relocated
    # alias differ from the rest of the library's abspath-based results.
    import_root = Path(os.path.abspath(videos_dir)).parent.parent
    session_name = recorded_path.parent.name
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
                recorded = _xsym_recorded_target(path)
                if recorded:
                    # A pointer file is a link; an unresolvable one is a broken link,
                    # never a playable video.
                    is_link = True
                    alias_target = _legacy_macos_alias_target(directory, path)
                    target = alias_target or recorded
                    target_exists = bool(alias_target)
                else:
                    target = path
                    target_exists = True
            else:
                continue
            entries.append(
                VideoLibraryEntry(
                    name=item.name,
                    path=target if alias_target else path,
                    is_link=is_link,
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
    """Link source videos (symlink, or XSym pointer file when symlinks are refused)."""
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
        is_symlink = _create_link(source, destination)
        created.append(
            VideoLibraryEntry(
                name,
                destination if is_symlink else source,
                True,
                source,
                target_exists=True,
            )
        )
    return created


def remove_video_link(videos_dir: str, name: str) -> None:
    """Remove a library link without ever deleting a regular video file."""
    path = _require_library_child(videos_dir, name)
    if not _is_library_link(path):
        raise ValueError("Only video links can be removed here; the original file was not changed")
    os.unlink(path)


def rename_video_link(videos_dir: str, old_name: str, new_name: str) -> str:
    """Rename a library link while preserving its target."""
    source = _require_library_child(videos_dir, old_name)
    if not _is_library_link(source):
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
    """Atomically change a library link target without modifying either video."""
    link_path = _require_library_child(videos_dir, name)
    if not _is_library_link(link_path):
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
        _create_link(source, temporary)
        os.replace(temporary, link_path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return source

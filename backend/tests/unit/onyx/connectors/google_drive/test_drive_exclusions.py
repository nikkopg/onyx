"""Excluded folders and excluded file name patterns in the Google Drive connector.

Both indexing and pruning (slim retrieval) go through ``crawl_folders_for_files``
and ``_checkpointed_retrieval``, so excluded files are also removed by pruning.
"""

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

from googleapiclient.discovery import Resource

from onyx.connectors.google_drive.connector import GoogleDriveConnector
from onyx.connectors.google_drive.file_retrieval import (
    DriveFileFieldType,
    crawl_folders_for_files,
)
from onyx.connectors.google_drive.models import (
    DriveRetrievalStage,
    GoogleDriveCheckpoint,
    RetrievedDriveFile,
    StageCompletion,
)
from onyx.utils.threadpool_concurrency import ThreadSafeDict

_FILE_RETRIEVAL_MODULE = "onyx.connectors.google_drive.file_retrieval"
_USER = "user@example.com"

# root
# ├── report.docx
# ├── docs/          -> notes.docx
# └── logs/          -> run.csv
#     └── deep/      -> deep.csv
_FOLDERS: dict[str, list[str]] = {
    "root": ["docs", "logs"],
    "docs": [],
    "logs": ["deep"],
    "deep": [],
}
_FILES: dict[str, list[str]] = {
    "root": ["report.docx"],
    "docs": ["notes.docx"],
    "logs": ["run.csv"],
    "deep": ["deep.csv"],
}


def _fake_folders_in_parent(
    service: Resource,  # noqa: ARG001
    parent_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    for folder_id in _FOLDERS[cast(str, parent_id)]:
        yield {"id": folder_id, "name": folder_id}


def _fake_files_in_parent(
    service: Resource,  # noqa: ARG001
    parent_id: str,
    **_kwargs: object,
) -> Iterator[dict[str, Any]]:
    for name in _FILES[parent_id]:
        yield {"id": name, "name": name}


def _crawl(excluded_folder_ids: frozenset[str]) -> list[str]:
    traversed: set[str] = set()
    with (
        patch(
            f"{_FILE_RETRIEVAL_MODULE}._get_folders_in_parent",
            side_effect=_fake_folders_in_parent,
        ),
        patch(
            f"{_FILE_RETRIEVAL_MODULE}._get_files_in_parent",
            side_effect=_fake_files_in_parent,
        ),
    ):
        return [
            retrieved.drive_file["name"]
            for retrieved in crawl_folders_for_files(
                service=cast(Resource, object()),
                parent_id="root",
                field_type=DriveFileFieldType.STANDARD,
                user_email=_USER,
                traversed_parent_ids=traversed,
                update_traversed_ids_func=traversed.add,
                excluded_folder_ids=excluded_folder_ids,
            )
        ]


def test_crawl_skips_excluded_folder_and_its_subfolders() -> None:
    assert sorted(_crawl(frozenset())) == [
        "deep.csv",
        "notes.docx",
        "report.docx",
        "run.csv",
    ]
    assert sorted(_crawl(frozenset({"logs"}))) == ["notes.docx", "report.docx"]


def test_crawl_skips_everything_when_root_is_excluded() -> None:
    assert _crawl(frozenset({"root"})) == []


def test_connector_parses_excluded_folder_urls() -> None:
    connector = GoogleDriveConnector(
        shared_folder_urls="https://drive.google.com/drive/folders/root",
        exclude_folder_urls=(
            "https://drive.google.com/drive/u/0/folders/logs?usp=sharing, "
            "https://drive.google.com/drive/folders/tmp"
        ),
    )
    assert connector._excluded_folder_ids == frozenset({"logs", "tmp"})


def _retrieved(name: str) -> RetrievedDriveFile:
    return RetrievedDriveFile(
        completion_stage=DriveRetrievalStage.FOLDER_FILES,
        drive_file={
            "id": name,
            "name": name,
            "modifiedTime": "2024-06-01T00:00:00Z",
            "webViewLink": f"https://drive.google.com/file/d/{name}",
        },
        user_email=_USER,
        parent_id="root",
    )


def test_checkpointed_retrieval_drops_files_matching_patterns() -> None:
    connector = GoogleDriveConnector(
        shared_folder_urls="https://drive.google.com/drive/folders/root",
        exclude_file_patterns="log_*.csv, *.LOG",
    )
    checkpoint = GoogleDriveCheckpoint(
        has_more=True,
        retrieved_folder_and_drive_ids=set(),
        completion_stage=DriveRetrievalStage.FOLDER_FILES,
        completion_map=ThreadSafeDict(
            {
                _USER: StageCompletion(
                    stage=DriveRetrievalStage.FOLDER_FILES,
                    completed_until=0,
                    current_folder_or_drive_id=None,
                )
            }
        ),
    )
    names = ["report.docx", "Log_Run_01.csv", "server.log", "logbook.csv"]

    def retrieval_method(
        field_type: DriveFileFieldType,  # noqa: ARG001
        checkpoint: GoogleDriveCheckpoint,  # noqa: ARG001
        start: float | None = None,  # noqa: ARG001
        end: float | None = None,  # noqa: ARG001
    ) -> Iterator[RetrievedDriveFile]:
        for name in names:
            yield _retrieved(name)

    kept = [
        retrieved.drive_file["name"]
        for retrieved in connector._checkpointed_retrieval(
            retrieval_method=retrieval_method,
            field_type=DriveFileFieldType.SLIM,
            checkpoint=checkpoint,
        )
    ]

    assert kept == ["report.docx", "logbook.csv"]
    # The frontier still moves past the skipped files.
    assert checkpoint.completion_map[_USER].completed_until > 0

REQUEST_CONTENT_IDS: tuple[str, ...] = (
    "send_report",
    "review_document",
    "schedule_meeting",
    "update_file",
    "share_notes",
    "confirm_date",
    "revise_draft",
    "submit_form",
    "explain_result",
    "correct_record",
    "inspect_sample",
    "approve_request",
    "archive_message",
    "compare_versions",
    "annotate_chart",
    "verify_entry",
)

ASSERTION_CONTENT_IDS: tuple[str, ...] = (
    "result_difference",
    "system_unstable",
    "sample_contaminated",
    "meeting_delayed",
    "report_complete",
    "estimate_inaccurate",
    "record_incomplete",
    "method_effective",
    "schedule_conflict",
    "document_outdated",
    "measurement_error",
    "procedure_changed",
    "sample_ready",
    "model_consistent",
    "request_valid",
    "timeline_changed",
)

ALL_CONTENT_IDS = REQUEST_CONTENT_IDS + ASSERTION_CONTENT_IDS

CONTENT_CODE_BY_ID: dict[str, str] = {
    content_id: f"<CONTENT_{index:02d}>"
    for index, content_id in enumerate(ALL_CONTENT_IDS, start=1)
}

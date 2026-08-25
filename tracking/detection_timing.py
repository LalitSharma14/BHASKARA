"""Timing policy for asynchronous detector results."""


def calculate_result_age(current_frame_number, source_frame_number):
    """Return a non-negative detector-result age measured in video frames."""

    return max(0, current_frame_number - source_frame_number)


def should_accept_result(result_age_frames, maximum_age_frames):
    """Return whether an asynchronous result is recent enough to reconcile."""

    return 0 <= result_age_frames <= maximum_age_frames


def select_reconciled_box(detector_box, current_track_box, result_age_frames):
    """Choose geometry without moving a current track back to a stale box.

    A fresh result is authoritative. A delayed result may confirm identity and
    contribute a label vote, but optical flow already represents the track at
    the current video time and therefore keeps control of its position.
    """

    if result_age_frames <= 0:
        return detector_box

    return current_track_box

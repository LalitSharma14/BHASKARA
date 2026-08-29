"""Deterministic detector scheduling helpers."""


def should_schedule_evaluation_frame(frame_number, interval, total_frames=None):
    """Return whether this frame belongs to the fixed evaluation schedule.

    The final frame is skipped when the video length is known so every
    synchronous result has a following frame on which it can be reconciled.
    """

    if interval <= 0 or frame_number <= 0:
        return False
    if total_frames and frame_number >= total_frames:
        return False
    return frame_number % interval == 0


def scheduled_evaluation_frames(total_frames, interval):
    """Return the exact source-frame schedule for a complete video."""

    if total_frames <= 0 or interval <= 0:
        return []
    return list(range(interval, total_frames, interval))

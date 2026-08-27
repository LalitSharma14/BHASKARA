"""Identity-preserving lifecycle helpers for optical-flow failures."""


def prepare_track_for_lost_pool(track, refreshed_points, frame_number):
    """Preserve identity and last box while transitioning out of active flow."""

    track["points"] = refreshed_points
    track["fresh_detection"] = False
    track["lost_at_frame"] = frame_number
    return track

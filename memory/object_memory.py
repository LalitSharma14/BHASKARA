# --------------------------------------------------
# BHASKARA
# Trusted Object Memory
# --------------------------------------------------

from datetime import datetime


# --------------------------------------------------
# Main in-memory store
#
# Key:
#   track_id
#
# Value:
#   trusted information about that physical object
# --------------------------------------------------

object_memory = {}


# --------------------------------------------------
# Update memory
# --------------------------------------------------

def update_memory(
    trusted_tracks,
    frame_number
):
    """
    Update BHASKARA's memory using only trusted tracks.

    A trusted track should already have:
    - repeated detector confirmations
    - a valid bounding box
    - a stable track ID
    """

    current_time = datetime.now()


    for track in trusted_tracks:

        track_id = track["id"]

        object_name = track["object"]

        confidence = track["confidence"]

        box = track["box"]

        confirmations = track.get(
            "confirmations",
            0
        )


        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1


        # Ignore broken / collapsed boxes
        if width < 10 or height < 10:
            continue


        # --------------------------------------------------
        # IMPORTANT:
        #
        # Memory is indexed by track_id.
        #
        # Therefore if:
        #
        # ID 12:
        # desk -> bed
        #
        # the old desk memory is overwritten rather than
        # producing a second physical object.
        # --------------------------------------------------

        object_memory[track_id] = {

            "object": object_name,

            "confidence": confidence,

            "box": box,

            "confirmations": confirmations,

            "label_votes": track.get(
                "label_votes",
                {}
            ),

            "last_seen_frame": frame_number,

            "last_seen_time": current_time
        }


# --------------------------------------------------
# Get complete memory
# --------------------------------------------------

def get_memory():

    return object_memory


# --------------------------------------------------
# Find all remembered instances of an object
# --------------------------------------------------

def find_object(object_name):

    matches = []


    for track_id, data in object_memory.items():

        if (
            data["object"].lower()
            == object_name.lower()
        ):

            matches.append({

                "track_id": track_id,

                **data
            })


    return matches


# --------------------------------------------------
# Get most recently seen object instance
# --------------------------------------------------

def get_last_seen(object_name):

    matches = find_object(
        object_name
    )


    if not matches:
        return None


    return max(
        matches,
        key=lambda item:
            item["last_seen_time"]
    )


# --------------------------------------------------
# Clear memory
# --------------------------------------------------

def clear_memory():

    object_memory.clear()
from collections import namedtuple


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "returns", "goal_id"],
)

import enum


class EndCause(enum.Enum):
    NONE = 0
    TIMEOUT = 1
    ALL_AGENTS_INACTIVE = 2


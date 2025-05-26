"""Access control system for tools.

This module provides access control mechanisms for tools, including the AccessLevel
enum for specifying tool permission levels.
"""

from enum import Enum


class AccessLevel(Enum):
    """Access level for tools to manage read/write operations.

    Levels:
        READ: Tool only reads state, no side effects
        WRITE: Tool modifies state or has side effects (default)
        ADMIN: Tool has system-level access (fork, spawn, goto)
    """

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

    @classmethod
    def from_string(cls, value: str) -> "AccessLevel":
        """Convert a string representation to AccessLevel enum.

        Args:
            value: String representation of access level ("read", "write", "admin")

        Returns:
            AccessLevel enum value

        Raises:
            ValueError: If the string is empty/None or doesn't match a valid level
        """
        if value is None or value == "":
            # Empty or None value is not valid because the caller should
            # explicitly choose an access level or rely on higher‑level
            # defaults (e.g. register_tool() falls back to WRITE).
            raise ValueError("Access level string cannot be empty – expected 'read', 'write', or 'admin'.")

        value = value.lower()
        if value == "read":
            return cls.READ
        elif value == "write":
            return cls.WRITE
        elif value == "admin":
            return cls.ADMIN
        else:
            raise ValueError(f"Invalid access level: {value}. Expected 'read', 'write', or 'admin'")

    def compare_to(self, other: "AccessLevel") -> int:
        """Compare this access level to another access level.

        Args:
            other: The access level to compare with

        Returns:
            -1 if self < other, 0 if self == other, 1 if self > other

        Examples:
            >>> AccessLevel.READ.compare_to(AccessLevel.WRITE)
            -1
            >>> AccessLevel.ADMIN.compare_to(AccessLevel.READ)
            1
            >>> AccessLevel.WRITE.compare_to(AccessLevel.WRITE)
            0
        """
        # Define the hierarchy of access levels
        access_order = {AccessLevel.READ: 0, AccessLevel.WRITE: 1, AccessLevel.ADMIN: 2}

        self_val = access_order[self]
        other_val = access_order[other]

        if self_val < other_val:
            return -1
        elif self_val > other_val:
            return 1
        else:
            return 0

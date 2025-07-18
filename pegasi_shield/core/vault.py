from typing import List, Tuple


class Vault:
    """
    A utility class for storing tuples, often used as placeholder values for anonymization
    that need to be retained for later decoding.

    Provides methods for adding, removing, and retrieving tuples.

    Attributes:
        _tuples (List[Tuple]): The list of stored tuples.
    """

    def __init__(self):
        self._tuples: List[Tuple] = []

    def add(self, new_tuple: Tuple):
        """
        Adds a tuple to the vault.

        Args:
            new_tuple (Tuple): The tuple to be added.
        """
        self._tuples.append(new_tuple)

    def add_multiple(self, new_tuples: List[Tuple]):
        """
        Adds multiple tuples to the vault.

        Args:
            new_tuples (List[Tuple]): A list of tuples to be added.
        """
        self._tuples.extend(new_tuples)

    def remove(self, tuple_to_remove: Tuple):
        """
        Removes a specific tuple from the vault.

        Args:
            tuple_to_remove (Tuple): The tuple to be removed.
        """
        self._tuples.remove(tuple_to_remove)

    def get_all(self) -> List[Tuple]:
        """
        Retrieves all stored tuples from the vault.

        Returns:
            List[Tuple]: A list containing all the stored tuples.
        """
        return self._tuples

    def clear(self):
        """
        Resets the vault by clearing all stored tuples.
        """
        self._tuples.clear()

import logging
from pegasi_shield_safeguards.vault import Vault
from pegasi_shield_safeguards.output_detectors.base_detector import Detector

log = logging.getLogger(__name__)


class Deanonymize(Detector):
    """
    A class for replacing placeholders in the model's output with real values from a secure vault.

    This class uses the Vault class to access stored values and replaces any placeholders
    in the model's output with their corresponding values from the vault.
    """

    def __init__(self, vault: Vault):
        """
        Initializes an instance of the DeanonymizationDetector class.

        Parameters:
            vault (Vault): An instance of the Vault class which stores the real values.
        """
        self._vault = vault

    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        vault_items = self._vault.get_all()
        if len(vault_items) == 0:
            log.warning("No items found in the Vault")

        for vault_item in vault_items:
            log.debug(f"Replaced placeholder ${vault_item[0]} with real value")
            output = output.replace(vault_item[0], vault_item[1])

        self._vault.clear()

        return output, True, 0.0

import json
import logging
import os
import re
from typing import Tuple, List, Optional, Pattern, Dict

from ..vault import Vault
from .base_detector import Detector

SENSITIVE_PATTERNS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "patterns",
    "sensitive_patterns.json",
)

log = logging.getLogger(__name__)


class Reidentify(Detector):
    """
    A class for replacing fake/redacted data in the model's output with the original values from a secure vault.

    This detector works in conjunction with the Deidentifier input detector. It searches for placeholder
    patterns or fake data in the model's output and replaces them with the original values stored in the vault.
    
    It supports:
    1. Placeholder formats like [REDACTED_TYPE] and <TYPE>
    2. Faker-generated replacements
    3. Custom regex patterns from sensitive_patterns.json
    4. Numbered placeholders like [REDACTED_TYPE_1]
    """

    def __init__(
        self, 
        vault: Vault,
        placeholder_pattern: Optional[str] = None,
        restore_all: bool = True,
        entity_types: Optional[List[str]] = None,
        regex_pattern_path: str = SENSITIVE_PATTERNS_PATH
    ):
        """
        Initialize the Reidentify detector.
        
        Args:
            vault (Vault): A vault instance containing the original sensitive data.
            placeholder_pattern (Optional[str]): Custom regex pattern to identify placeholders.
                If None, will use default patterns to match [REDACTED_TYPE] and <TYPE> formats.
            restore_all (bool): If True, attempt to restore all entities in the vault.
                If False, only restore entities that match the placeholder pattern.
            entity_types (Optional[List[str]]): List of entity types to restore.
                If None, all entity types will be restored.
            regex_pattern_path (str): Path to a JSON file with regex pattern groups.
                If not provided, defaults to sensitive_patterns.json.
        """
        self._vault = vault
        self._entity_types = entity_types
        self._restore_all = restore_all
        self._regex_patterns = self._load_regex_patterns(regex_pattern_path)
        
        # Compile regex patterns for placeholder detection
        if placeholder_pattern:
            self._placeholder_pattern = re.compile(placeholder_pattern)
        else:
            # Default patterns to match [REDACTED_TYPE] and <TYPE> formats
            # Also match numbered placeholders like [REDACTED_TYPE_1]
            self._placeholder_pattern = re.compile(r'(\[REDACTED_[A-Z_]+(_(\d+))?\]|<[A-Z_]+>)')
    
    def _load_regex_patterns(self, json_path: str) -> Dict[str, Pattern]:
        """
        Load regex patterns from a specified JSON file and compile them for use in reidentification.
        
        Args:
            json_path (str): Path to the JSON file containing regex patterns.
            
        Returns:
            Dict[str, Pattern]: Dictionary mapping entity types to compiled regex patterns.
        """
        compiled_patterns = {}
        try:
            with open(json_path, "r") as myfile:
                pattern_groups_raw = json.load(myfile)
                
            for group in pattern_groups_raw:
                entity_type = group["name"].upper()
                # Initialize a list for this entity type if it doesn't exist
                if entity_type not in compiled_patterns:
                    compiled_patterns[entity_type] = []
                    
                for expression in group["expressions"]:
                    # Compile the regex pattern and add it to the list
                    compiled_patterns[entity_type].append(re.compile(expression))
                    log.debug(f"Loaded regex pattern for {entity_type}: {expression}")
                    
        except FileNotFoundError:
            log.warning(f"Could not find {json_path}")
        except json.decoder.JSONDecodeError as json_error:
            log.warning(f"Could not parse {json_path}: {json_error}")
            
        return compiled_patterns
        
    def _find_sensitive_patterns(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find sensitive patterns in the text using regex patterns.
        
        Args:
            text (str): The text to search for sensitive patterns.
            
        Returns:
            List[Tuple[str, str, int, int]]: List of tuples containing (matched_text, entity_type, start, end).
        """
        matches = []
        
        for entity_type, patterns in self._regex_patterns.items():
            # Skip if entity_types is specified and this type is not in the list
            if self._entity_types and entity_type not in self._entity_types:
                continue
                
            # Process each pattern for this entity type
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group(0)
                    start, end = match.span()
                    matches.append((matched_text, entity_type, start, end))
                    log.debug(f"Found sensitive pattern match: {matched_text} ({entity_type})")
                
        return matches
    
    def scan(self, prompt: str, output: str, context: str = None) -> Tuple[str, bool, float]:
        """
        Process the model's output to replace placeholders or fake data with original values.
        
        Args:
            prompt (str): The original prompt sent to the model.
            output (str): The model's output to process.
            context (str, optional): Additional context for processing.
            
        Returns:
            Tuple[str, bool, float]: A tuple containing:
                - str: The processed output with original values restored.
                - bool: Always True (this detector doesn't invalidate outputs).
                - float: Risk score (always 0.0 as this detector doesn't assess risk).
        """
        try:
            # If the vault is empty, return the original output
            vault_items = self._vault.get_all()
            if not vault_items:
                log.warning("No items found in the Vault for reidentification")
                return output, True, 0.0
            
            log.debug(f"Found {len(vault_items)} items in the vault for reidentification")
            
            # First pass: restore entities based on the restore_all setting
            if self._restore_all:
                # Replace all fake/redacted data with original values
                restored_output = self._restore_all_entities(output, vault_items)
            else:
                # Only replace placeholders that match the pattern
                restored_output = self._restore_placeholders(output, vault_items)
            
            # Second pass: look for any sensitive patterns that might still be in the output
            # This is helpful for catching patterns that weren't explicitly stored in the vault
            # but match known sensitive data patterns (like credit card numbers, SSNs, etc.)
            sensitive_matches = self._find_sensitive_patterns(restored_output)
            
            # Sort matches in reverse order by start position to avoid offset issues when replacing
            sensitive_matches.sort(key=lambda x: x[2], reverse=True)
            
            # If we found sensitive patterns, try to restore them from the vault
            if sensitive_matches:
                log.debug(f"Found {len(sensitive_matches)} sensitive patterns in the output")
                for matched_text, entity_type, start, end in sensitive_matches:
                    # First check for direct matches in the vault
                    direct_matches = [item for item in vault_items if matched_text == item[2]]  # Check if pattern matches a fake replacement
                    
                    if direct_matches:
                        # We found a direct match for this pattern in the vault
                        original_text = direct_matches[0][0]
                        restored_output = restored_output[:start] + original_text + restored_output[end:]
                        log.debug(f"Restored direct pattern match '{matched_text}' to original '{original_text}'")
                    else:
                        # Look for matching entity types in the vault
                        matching_items = [item for item in vault_items if item[1] == entity_type]
                        
                        # If no direct matches, check for related entity types
                        if not matching_items:
                            related_types = []
                            if entity_type == 'SSN':
                                related_types.append('US_SSN')
                            elif entity_type == 'US_SSN':
                                related_types.append('SSN')
                            elif entity_type == 'PHONE':
                                related_types.append('PHONE_NUMBER')
                            elif entity_type == 'PHONE_NUMBER':
                                related_types.append('PHONE')
                            elif entity_type == 'CREDIT_CARD':
                                related_types.append('CREDIT_CARD_NUMBER')
                            elif entity_type == 'CREDIT_CARD_NUMBER':
                                related_types.append('CREDIT_CARD')
                            
                            for related_type in related_types:
                                related_items = [item for item in vault_items if item[1] == related_type]
                                if related_items:
                                    matching_items = related_items
                                    break
                        
                        if matching_items:
                            # Replace the matched text with the original value from the vault
                            original_text = matching_items[0][0]
                            restored_output = restored_output[:start] + original_text + restored_output[end:]
                            log.debug(f"Restored sensitive pattern '{matched_text}' to original '{original_text}'")
            
            # Return the restored output
            return restored_output, True, 0.0
            
        except Exception as e:
            # If there's an error, log it and return the original output
            log.error(f"Error in Reidentify.scan: {e}")
            return output, True, 0.0
    
    def _restore_all_entities(self, output: str, vault_items: List[Tuple]) -> str:
        """
        Restore all entities from the vault in the output.
        
        Args:
            output (str): The model's output to process.
            vault_items (List[Tuple]): List of tuples from the vault.
            
        Returns:
            str: The processed output with original values restored.
        """
        restored_output = output
        
        # Process each item in the vault
        for item in vault_items:
            original_text, entity_type, fake_replacement = item
            
            # Skip if entity_types is specified and this type is not in the list
            if self._entity_types and entity_type not in self._entity_types:
                continue
            
            # Replace the fake replacement with the original text
            if fake_replacement in restored_output:
                restored_output = restored_output.replace(fake_replacement, original_text)
                log.debug(f"Restored '{fake_replacement}' to original '{original_text}'")
            
            # Also check for placeholder formats
            placeholder_format1 = f"[REDACTED_{entity_type}]"
            placeholder_format2 = f"<{entity_type}>"
            
            # Handle related entity types for placeholders
            related_placeholders = []
            if entity_type == 'SSN':
                related_placeholders.extend(["[REDACTED_US_SSN]", "<US_SSN>"])
            elif entity_type == 'US_SSN':
                related_placeholders.extend(["[REDACTED_SSN]", "<SSN>"])
            elif entity_type == 'PHONE':
                related_placeholders.extend(["[REDACTED_PHONE_NUMBER]", "<PHONE_NUMBER>"])
            elif entity_type == 'PHONE_NUMBER':
                related_placeholders.extend(["[REDACTED_PHONE]", "<PHONE>"])
            elif entity_type == 'CREDIT_CARD':
                related_placeholders.extend(["[REDACTED_CREDIT_CARD_NUMBER]", "<CREDIT_CARD_NUMBER>"])
            elif entity_type == 'CREDIT_CARD_NUMBER':
                related_placeholders.extend(["[REDACTED_CREDIT_CARD]", "<CREDIT_CARD>"])
            
            # Check for numbered placeholders too (e.g., [REDACTED_PERSON_1])
            for i in range(1, 10):
                numbered_placeholder = f"[REDACTED_{entity_type}_{i}]"
                if numbered_placeholder in restored_output:
                    restored_output = restored_output.replace(numbered_placeholder, original_text)
                    log.debug(f"Restored numbered placeholder '{numbered_placeholder}' to original '{original_text}'")
            
            if placeholder_format1 in restored_output:
                restored_output = restored_output.replace(placeholder_format1, original_text)
                log.debug(f"Restored placeholder '{placeholder_format1}' to original '{original_text}'")
                
            if placeholder_format2 in restored_output:
                restored_output = restored_output.replace(placeholder_format2, original_text)
                log.debug(f"Restored placeholder '{placeholder_format2}' to original '{original_text}'")
                
            # Check for related placeholders
            for related_placeholder in related_placeholders:
                if related_placeholder in restored_output:
                    restored_output = restored_output.replace(related_placeholder, original_text)
                    log.debug(f"Restored related placeholder '{related_placeholder}' to original '{original_text}'")
        
        return restored_output
    
    def _restore_placeholders(self, output: str, vault_items: List[Tuple]) -> str:
        """
        Restore only placeholders that match the pattern.
        
        Args:
            output (str): The model's output to process.
            vault_items (List[Tuple]): List of tuples from the vault.
            
        Returns:
            str: The processed output with original values restored.
        """
        restored_output = output
        
        # Find all placeholders in the output
        placeholders = self._placeholder_pattern.findall(restored_output)
        
        if not placeholders:
            log.debug("No placeholders found in the output")
            return restored_output
        
        # Extract just the placeholder strings (not the tuple from findall)
        placeholder_strings = [p[0] for p in placeholders]
        
        # Process each placeholder
        for placeholder in placeholder_strings:
            # Extract the entity type from the placeholder
            if placeholder.startswith('[REDACTED_'):
                # Handle [REDACTED_TYPE] format
                entity_type = placeholder.strip('[]').split('_', 1)[1]
                # Remove any trailing numbers (e.g., PERSON_1 -> PERSON)
                if '_' in entity_type:
                    entity_type = entity_type.rsplit('_', 1)[0]
            elif placeholder.startswith('<'):
                # Handle <TYPE> format
                entity_type = placeholder.strip('<>')
            else:
                # Skip if the placeholder doesn't match expected formats
                continue
            
            # Skip if entity_types is specified and this type is not in the list
            if self._entity_types and entity_type not in self._entity_types:
                continue
            
            # Check for direct matches first
            matching_items = [item for item in vault_items if item[1] == entity_type]
            
            # If no direct matches, check for related entity types
            if not matching_items:
                related_types = []
                if entity_type == 'SSN':
                    related_types.append('US_SSN')
                elif entity_type == 'US_SSN':
                    related_types.append('SSN')
                elif entity_type == 'PHONE':
                    related_types.append('PHONE_NUMBER')
                elif entity_type == 'PHONE_NUMBER':
                    related_types.append('PHONE')
                elif entity_type == 'CREDIT_CARD':
                    related_types.append('CREDIT_CARD_NUMBER')
                elif entity_type == 'CREDIT_CARD_NUMBER':
                    related_types.append('CREDIT_CARD')
                
                for related_type in related_types:
                    related_items = [item for item in vault_items if item[1] == related_type]
                    if related_items:
                        matching_items = related_items
                        break
            
            if matching_items:
                # Use the first matching item (could be enhanced to find best match)
                original_text = matching_items[0][0]
                restored_output = restored_output.replace(placeholder, original_text)
                log.debug(f"Restored placeholder '{placeholder}' to original '{original_text}'")
        
        return restored_output

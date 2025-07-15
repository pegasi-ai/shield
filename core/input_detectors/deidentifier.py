import logging
import os
from typing import List, Optional, Tuple

from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer
from presidio_anonymizer import AnonymizerEngine

from pegasi_shield_safeguards.vault import Vault
from pegasi_shield_safeguards.input_detectors.base_detector import Detector

# Configure logging
log = logging.getLogger(__name__)

# Initialize Faker with a seed for reproducibility
fake = Faker(seed=42)

# Entity mapping for GLiNER recognizer
ENTITY_MAPPING = {
    # Basic Personal Identifiers
    "person": "PERSON",
    "first name": "FIRST_NAME",
    "last name": "LAST_NAME",

    # Contact Details
    "phone number": "PHONE",
    "mobile phone": "PHONE",
    "work phone": "PHONE",
    "email": "EMAIL",
    "work email": "EMAIL",
    "address": "ADDRESS",
    "zipcode": "ZIP",
    "postal code": "ZIP",

    # Organizational & Job Details
    "organization": "ORGANIZATION",
    "company": "ORGANIZATION",
    "department": "DEPARTMENT",
    "manager": "MANAGER",
    "office location": "OFFICE_LOCATION",
    "employee id": "EMPLOYEE_ID",
    "job title": "JOB_TITLE",
    "hire date": "HIRE_DATE",
    "termination date": "TERMINATION_DATE",
    "employee type": "EMPLOYEE_TYPE",
    "job level": "JOB_LEVEL",
    "internal identifier": "INTERNAL_ID",  # Company-specific internal code

    # Compensation & Financial Data
    "salary": "SALARY",
    "compensation": "COMPENSATION",
    "bonus": "BONUS",
    "salary band": "SALARY_BAND",
    "time off accrual": "TIME_OFF_ACCRUAL",
    "bank account number": "BANK_ACCOUNT",
    "bank routing number": "BANK_ROUTING",
    "tax id": "TAX_ID",
    "credit card number": "CREDIT_CARD",

    # Personal & Demographic Information
    "date of birth": "DOB",
    "birthdate": "DOB",
    "ssn": "SSN",
    "social security number": "SSN",
    "gender": "GENDER",
    "ethnicity": "ETHNICITY",
    "marital status": "MARITAL_STATUS",

    # Medical / Sensitive Data
    "medical diagnosis": "DIAGNOSIS",
    "doctor": "DOCTOR",
    "medical record number": "MRN",
    "immunization records": "IMMUNIZATION",
    "allergies": "ALLERGIES",
    "primary care physician": "PRIMARY_CARE_PHYSICIAN",
    "physical examination": "PHYSICAL_EXAM",
    "blood type": "BLOOD_TYPE",
    "mental health diagnosis": "MENTAL_HEALTH_DIAGNOSIS",
    "chronic condition": "CHRONIC_CONDITION",
    "medical leave": "MEDICAL_LEAVE",
    "health insurance": "HEALTH_INSURANCE",
    "health plan": "HEALTH_PLAN",
    "sensitive data": "SENSITIVE_DATA",

    # Work Eligibility & Legal
    "visa status": "VISA_STATUS",
    "work authorization": "WORK_AUTHORIZATION",
    "security clearance": "SECURITY_CLEARANCE",

    # Performance & Evaluation
    "performance review": "PERFORMANCE_REVIEW",
    "performance rating": "PERFORMANCE_RATING",

    # Emergency & Benefits
    "emergency contact": "EMERGENCY_CONTACT",
    "benefits": "BENEFITS",

    # Education & Certifications
    "education": "EDUCATION",
    "degree": "DEGREE",
    "university": "UNIVERSITY",
    "certifications": "CERTIFICATIONS",

    # Security questionnaires
    "car model": "CAR_MODEL",
    "car make": "CAR_MAKE",
    "car year": "CAR_YEAR",
    "car color": "CAR_COLOR",
    "elementary school": "ELEMENTARY_SCHOOL",
    "high school": "HIGH_SCHOOL",
    "university": "UNIVERSITY",
    "college": "COLLEGE",
    "work experience": "WORK_EXPERIENCE",
    "pet names": "PET_NAMES",
    "blood type": "BLOOD_TYPE",
    "allergies": "ALLERGIES",
}

# Faker mapping for entity types
ENTITY_FAKER_MAP = {
    "PERSON": fake.name,
    "FIRST_NAME": fake.first_name,
    "LAST_NAME": fake.last_name,
    "PHONE": fake.phone_number,
    "EMAIL": fake.email,
    "ADDRESS": fake.address,
    "ZIP": fake.zipcode,
    "ORGANIZATION": fake.company,
    "DEPARTMENT": lambda: f"{fake.job()[:10]} Department",
    "MANAGER": fake.name,
    "OFFICE_LOCATION": fake.city,
    "EMPLOYEE_ID": lambda: f"EMP{fake.random_number(digits=6)}",
    "JOB_TITLE": fake.job,
    "HIRE_DATE": fake.date,
    "TERMINATION_DATE": fake.date,
    "EMPLOYEE_TYPE": lambda: fake.random_element(elements=("Full-time", "Part-time", "Contract")),
    "JOB_LEVEL": lambda: f"Level {fake.random_int(min=1, max=10)}",
    "INTERNAL_ID": lambda: f"ID-{fake.random_number(digits=8)}",
    "SALARY": lambda: f"${fake.random_number(digits=5)}",
    "COMPENSATION": lambda: f"${fake.random_number(digits=6)}",
    "BONUS": lambda: f"${fake.random_number(digits=4)}",
    "SALARY_BAND": lambda: f"Band {fake.random_letter()}-{fake.random_int(min=1, max=5)}",
    "TIME_OFF_ACCRUAL": lambda: f"{fake.random_int(min=1, max=30)} days",
    "BANK_ACCOUNT": fake.bban,
    "BANK_ROUTING": lambda: fake.random_number(digits=9),
    "TAX_ID": lambda: f"TAX-{fake.random_number(digits=9)}",
    "CREDIT_CARD": fake.credit_card_number,
    "DOB": fake.date_of_birth,
    "SSN": fake.ssn,
    "GENDER": lambda: fake.random_element(elements=("Male", "Female", "Non-binary")),
    "ETHNICITY": lambda: fake.random_element(elements=("White", "Black", "Asian", "Hispanic", "Other")),
    "MARITAL_STATUS": lambda: fake.random_element(elements=("Single", "Married", "Divorced", "Widowed")),
    "DIAGNOSIS": lambda: f"Diagnosis: {fake.word()}",
    "DOCTOR": lambda: f"Dr. {fake.name()}",
    "MRN": lambda: f"MRN-{fake.random_number(digits=8)}",
    "IMMUNIZATION": lambda: f"{fake.word()} Vaccine",
    "ALLERGIES": lambda: f"Allergic to {fake.word()}",
    "PRIMARY_CARE_PHYSICIAN": lambda: f"Dr. {fake.name()}",
    "PHYSICAL_EXAM": lambda: f"Physical Exam on {fake.date()}",
    "BLOOD_TYPE": lambda: fake.random_element(elements=("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-")),
    "MENTAL_HEALTH_DIAGNOSIS": lambda: f"{fake.word()} disorder",
    "CHRONIC_CONDITION": lambda: f"Chronic {fake.word()}",
    "MEDICAL_LEAVE": lambda: f"Medical leave from {fake.date()} to {fake.date()}",
    "HEALTH_INSURANCE": lambda: f"{fake.company()} Health Insurance",
    "HEALTH_PLAN": lambda: f"{fake.company()} Health Plan {fake.random_letter()}{fake.random_int(min=1, max=99)}",
    "SENSITIVE_DATA": fake.text,
    "VISA_STATUS": lambda: fake.random_element(elements=("H1B", "L1", "F1", "Green Card", "EAD")),
    "WORK_AUTHORIZATION": lambda: f"Authorized until {fake.date()}",
    "SECURITY_CLEARANCE": lambda: fake.random_element(elements=("Secret", "Top Secret", "Confidential", "Public Trust")),
    "PERFORMANCE_REVIEW": lambda: f"Review on {fake.date()}: {fake.random_element(elements=('Excellent', 'Good', 'Average', 'Needs Improvement'))}",
    "PERFORMANCE_RATING": lambda: f"{fake.random_int(min=1, max=5)}/5",
    "EMERGENCY_CONTACT": lambda: f"{fake.name()}, {fake.phone_number()}",
    "BENEFITS": lambda: f"{fake.random_element(elements=('401k', 'Health Insurance', 'Dental', 'Vision', 'Life Insurance'))}",
    "EDUCATION": lambda: f"{fake.random_element(elements=('Bachelor', 'Master', 'PhD', 'Associate'))} in {fake.word()}",
    "DEGREE": lambda: f"{fake.random_element(elements=('BS', 'BA', 'MS', 'MA', 'PhD', 'MBA'))} in {fake.word()}",
    "UNIVERSITY": lambda: f"{fake.company()} University",
    "CERTIFICATIONS": lambda: f"{fake.word()} Certification",
    "CAR_MODEL": lambda: fake.random_element(elements=("Civic", "Accord", "Camry", "Corolla", "F-150", "Silverado")),
    "CAR_MAKE": lambda: fake.random_element(elements=("Honda", "Toyota", "Ford", "Chevrolet", "BMW", "Mercedes")),
    "CAR_YEAR": lambda: str(fake.random_int(min=1990, max=2023)),
    "CAR_COLOR": lambda: fake.safe_color_name(),
    "ELEMENTARY_SCHOOL": lambda: f"{fake.word().capitalize()} Elementary School",
    "HIGH_SCHOOL": lambda: f"{fake.word().capitalize()} High School",
    "COLLEGE": lambda: f"{fake.company()} College",
    "WORK_EXPERIENCE": lambda: f"{fake.random_int(min=1, max=20)} years at {fake.company()}",
    "PET_NAMES": fake.first_name,
}


class Deidentifier:
    """
    Deidentifier detector that uses Presidio Analyzer and Faker to identify and redact sensitive information.
    
    This detector uses GLiNER for entity recognition and Faker to replace sensitive data with realistic fake data.
    All original sensitive data is stored in a Vault for potential later retrieval.
    """
    
    def __init__(
        self,
        vault: Vault,
        entity_types: Optional[List[str]] = None,
        allowed_strings: Optional[List[str]] = None,
        pretext: str = "",
        use_faker: bool = False,
    ):
        """
        Initialize the Deidentifier detector.
        
        Args:
            vault (Vault): A vault instance to store the original sensitive data.
            entity_types (Optional[List[str]]): List of entity types to detect. If None, all supported types are used.
            allowed_strings (Optional[List[str]]): List of strings that should not be redacted.
            pretext (str): Text to prepend to the sanitized output.
            use_faker (bool): Whether to use Faker to generate realistic replacements instead of placeholders.
                              If False (default), placeholders like [REDACTED_PERSON] will be used.
        """
        # Disable tokenizers parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Create analyzer and anonymizer engines
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        
        # Set up GLiNER recognizer
        self._gliner_recognizer = GLiNERRecognizer(
            model_name="urchade/gliner_multi_pii-v1",
            entity_mapping=ENTITY_MAPPING,
            flat_ner=False,
            multi_label=True,
            map_location="cpu",
        )
        
        # Add the GLiNER recognizer to the registry
        self._analyzer.registry.add_recognizer(self._gliner_recognizer)
        
        # Remove the spaCy recognizer to avoid NER coming from spaCy
        self._analyzer.registry.remove_recognizer("SpacyRecognizer")
        
        # Store configuration
        self._vault = vault
        self._entity_types = entity_types
        self._allowed_strings = allowed_strings
        self._pretext = pretext
        self._use_faker = use_faker
    
    def scan(self, prompt: str) -> Tuple[str, bool, float]:
        """
        Process the input prompt to identify and redact sensitive information.
        
        Args:
            prompt (str): The input text to scan for sensitive information.
            
        Returns:
            Tuple[str, bool, float]: A tuple containing:
                - str: The sanitized text with sensitive information replaced with fake data.
                - bool: False if sensitive information was found and redacted, True otherwise.
                - float: Risk score where 0 means no risk and 1 means high risk.
        """
        # Return early if prompt is empty
        if not prompt or prompt.strip() == "":
            return prompt, True, 0.0
        
        try:
            # Analyze text to identify entities
            results = self._analyzer.analyze(
                text=prompt, 
                language="en",
                allow_list=self._allowed_strings,
                entities=self._entity_types
            )
            
            # If no entities were found, return the original prompt
            if not results:
                return prompt, True, 0.0
            
            # Calculate risk score (highest confidence score among detected entities)
            risk_score = max(result.score for result in results) if results else 0.0
            
            # If we're using Faker, we need to do our own replacement logic
            # Otherwise, use the anonymizer to redact the text
            if self._use_faker:
                # Make a copy of the prompt to modify
                sanitized_prompt = prompt
                
                # Sort results by start index (descending) to avoid offset issues when replacing text
                sorted_results = sorted(results, key=lambda x: x.start, reverse=True)
                
                # Replace each entity with a fake value
                for result in sorted_results:
                    original_text = prompt[result.start:result.end]
                    entity_type = result.entity_type
                    
                    # Generate fake replacement using Faker
                    fake_replacement = self._get_fake_value(entity_type)
                    
                    # Store the original text, entity type, and fake replacement in the vault
                    self._vault.add((original_text, entity_type, fake_replacement))
                    
                    # Replace the original text with the fake replacement
                    sanitized_prompt = sanitized_prompt[:result.start] + fake_replacement + sanitized_prompt[result.end:]
                    
                    # Log the redaction (for debugging)
                    log.debug(f"Redacted {entity_type}: '{original_text}' → '{fake_replacement}'")
            else:
                # Use the anonymizer to redact the text with placeholders
                anonymized_result = self._anonymizer.anonymize(
                    text=prompt, 
                    analyzer_results=results
                )
                
                # Store original values in the vault
                for result in results:
                    original_text = prompt[result.start:result.end]
                    entity_type = result.entity_type
                    
                    # Use placeholder format
                    fake_replacement = f"[REDACTED_{entity_type}]"
                    
                    # Store the original text, entity type, and fake replacement in the vault
                    self._vault.add((original_text, entity_type, fake_replacement))
                    
                    # Log the redaction (for debugging)
                    log.debug(f"Redacted {entity_type}: '{original_text}' → '{fake_replacement}'")
                
                # Get the anonymized text
                sanitized_prompt = anonymized_result.text
            
            # If the text was modified, return it with the pretext
            if prompt != sanitized_prompt:
                log.info(f"Found and redacted sensitive data. Risk score: {risk_score}")
                return self._pretext + sanitized_prompt, False, risk_score
            
            # If no changes were made, return the original prompt
            return prompt, True, 0.0
            
        except Exception as e:
            # If there's an error, log it and return the original prompt
            log.error(f"Error in Deidentifier.scan: {e}")
            return prompt, True, 0.0
    
    def _get_fake_value(self, entity_type: str) -> str:
        """
        Generate a fake value for the given entity type using Faker.
        
        Args:
            entity_type (str): The type of entity to generate a fake value for.
            
        Returns:
            str: A fake value appropriate for the entity type.
        """
        # Get the faker function for this entity type, or use a generic placeholder
        faker_func = ENTITY_FAKER_MAP.get(entity_type)
        
        if faker_func:
            try:
                return str(faker_func())
            except Exception as e:
                log.warning(f"Error generating fake value for {entity_type}: {e}")
                return f"[REDACTED_{entity_type}]"
        else:
            return f"[REDACTED_{entity_type}]"

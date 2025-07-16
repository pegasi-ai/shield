import spacy
import logging
from scipy.spatial import distance
from datetime import datetime
from dateutil import parser
import re

from pegasi_shield.utils import device

from .base_detector import Detector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TemporalMismatchDetector(Detector):
    """
    TemporalMismatchDetector Class:

    Detects and analyzes temporal expressions to identify mismatches using spaCy.
    """

    def __init__(self, model_name="en_core_web_sm"):
        """
        Initializes an instance with a spaCy model for NLP tasks.

        Parameters:
            model_name (str): The name of the spaCy model to load.
        """
        self._nlp = spacy.load(model_name)
        log.info(f"Loaded spaCy model {model_name}")

        self.quarter_mapping = {
            "Q1": {"start": (1, 1), "end": (3, 31)},
            "Q2": {"start": (4, 1), "end": (6, 30)},
            "Q3": {"start": (7, 1), "end": (9, 30)},
            "Q4": {"start": (10, 1), "end": (12, 31)},
            "first quarter": {"start": (1, 1), "end": (3, 31)},
            "second quarter": {"start": (4, 1), "end": (6, 30)},
            "third quarter": {"start": (7, 1), "end": (9, 30)},
            "fourth quarter": {"start": (10, 1), "end": (12, 31)},
            "1st quarter": {"start": (1, 1), "end": (3, 31)},
            "2nd quarter": {"start": (4, 1), "end": (6, 30)},
            "3rd quarter": {"start": (7, 1), "end": (9, 30)},
            "4th quarter": {"start": (10, 1), "end": (12, 31)},
            "first half": {"start": (1, 1), "end": (6, 30)},
            "second half": {"start": (7, 1), "end": (12, 31)},
            "H1": {"start": (1, 1), "end": (6, 30)},
            "H2": {"start": (7, 1), "end": (12, 31)},
            "first semester": {"start": (1, 1), "end": (6, 30)},
            "second semester": {"start": (7, 1), "end": (12, 31)},
            "1st half": {"start": (1, 1), "end": (6, 30)},
            "2nd half": {"start": (7, 1), "end": (12, 31)},
            "beginning of the year": {"start": (1, 1), "end": (6, 30)},
            "end of the year": {"start": (7, 1), "end": (12, 31)},
            "end of year": {"start": (7, 1), "end": (12, 31)},
            "EOY": {"start": (7, 1), "end": (12, 31)},
            "early months": {"start": (1, 1), "end": (4, 30)},
            "mid-year": {"start": (5, 1), "end": (8, 31)},
            "late months": {"start": (9, 1), "end": (12, 31)},
            "spring": {"start": (3, 20), "end": (6, 20)},
            "summer": {"start": (6, 21), "end": (9, 22)},
            "fall": {"start": (9, 23), "end": (12, 20)},
            "autumn": {"start": (9, 23), "end": (12, 20)},
            "winter": {
                "start": (12, 21),
                "end": (3, 19),
            },  # Note: Winter spans the end of one year and the start of the next
        }

    def extract_temporal_entities(self, text: str):
        """
        Extracts and normalizes temporal entities from text, including specific dates and abstract temporal expressions.
        Dynamically adjusts the year based on explicit year mentions in the text.

        Parameters:
            text (str): The text to process.

        Returns:
            A list of datetime objects and ranges.
        """
        # fix dateutil bug
        text = text.replace(",", ", ")

        doc = self._nlp(text)
        entities = []

        current_year = datetime.now().year  # Current year for reference
        year_reference = current_year  # Default to current year

        # Enhanced pattern to include two-digit year representations
        year_pattern = r"\b(\d{2,4})\b"
        years_found = re.findall(year_pattern, text)

        for year_str in years_found:
            if len(year_str) == 2:
                # Assuming '00' to '99' refer to 2000 to 2099
                year_int = 2000 + int(year_str)
                year_reference = min(
                    year_int, current_year
                )  # Use the detected year if it's not in the future
            elif len(year_str) == 4:
                year_reference = int(year_str)
                break  # Assume the first 4-digit year found is the reference

        # Extract specific dates and times, adjust year_reference dynamically
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                try:
                    normalized_date = parser.parse(ent.text, fuzzy=True)
                    entities.append(normalized_date)
                    # Adjust year_reference if a year is explicitly mentioned
                    if normalized_date.year != datetime.now().year:
                        year_reference = normalized_date.year
                except ValueError:
                    log.warning(f"Could not parse date: {ent.text}")

        # Update the normalization process using the dynamically adjusted year_reference
        for expression, dates in self.quarter_mapping.items():
            if re.search(r"\b" + expression.lower() + r"\b", text.lower()):
                start_date = datetime(year_reference, *dates["start"])
                end_date = datetime(year_reference, *dates["end"])
                if start_date > end_date:  # For expressions spanning years
                    end_date = datetime(year_reference + 1, *dates["end"])
                entities.append((start_date, end_date))

        return entities

    def calculate_distance(self, date_or_range1, date_or_range2):
        """
        Calculate the minimum distance in days between two dates or date ranges.
        Both date_or_range1 and date_or_range2 can be a single date (datetime object) or a date range (tuple of datetimes).
        Returns 0 if there's an overlap or the date is within the range.
        Returns a positive value for the number of days if they are adjacent or non-overlapping.
        """
        if isinstance(date_or_range1, tuple):
            start1, end1 = date_or_range1
        else:
            start1, end1 = date_or_range1, date_or_range1

        if isinstance(date_or_range2, tuple):
            start2, end2 = date_or_range2
        else:
            start2, end2 = date_or_range2, date_or_range2

        # Adjust for adjacent, non-overlapping ranges
        if end1 == start2:  # end of range1 is the start of range2
            return 1  # Considered a minimal but significant distance
        elif start1 == end2:  # start of range1 is the end of range2
            return 1  # Similarly considered a minimal but significant distance

        if end1 < start2:  # range1 entirely before range2
            return (start2 - end1).days
        elif start1 > end2:  # range1 entirely after range2
            return (start1 - end2).days
        else:  # ranges overlap
            return 0

    def compare_temporal_entities(self, prompt_dates, output_dates, context_dates):
        """
        Compares output dates against context and prompt dates,
        calculating a score where 1.00 represents no mismatch.
        """
        # Initialize counters for matches and total comparisons
        matches = 0
        total_comparisons = 0

        # Function to compare a set of dates to a set of reference dates, with special handling for context
        def compare_dates(source_dates, reference_dates, context=False):
            nonlocal matches, total_comparisons
            local_matches = 0
            local_comparisons = 0

            for source_date in source_dates:
                for ref_date in reference_dates:
                    distance = self.calculate_distance(source_date, ref_date)
                    local_comparisons += 1
                    if distance == 0:
                        local_matches += 1  # Increment local matches for zero distances

            # Adjusting total comparisons and matches based on context and overlap
            if context and local_matches > 0:
                # If there's some overlap, adjust total comparisons and matches
                # cap it at one, so if there's 10+ dates in context doesn't change
                matches += 1
                # Only count comparisons that resulted in a match to avoid penalizing extra context dates
                total_comparisons += 1
            else:
                # Standard behavior: add all comparisons and matches
                matches += local_matches
                total_comparisons += local_comparisons

        # Compare using the refined logic
        compare_dates(output_dates, context_dates, context=True)
        compare_dates(prompt_dates, context_dates, context=True)
        compare_dates(prompt_dates, output_dates, context=False)

        # Calculate mismatch score based on the proportion of matches
        mismatch_score = matches / total_comparisons if total_comparisons > 0 else 1.0

        # Inverting the logic so that 1.0 represents no mismatch and closer to 0.0 represents higher mismatch
        return mismatch_score

    def scan(self, prompt: str, output: str, context: str) -> (str, bool, float):
        """
        Evaluates likelihood of temporal mismatches.

        Parameters:
            context, prompt, output (str): Text segments to analyze.

        Returns:
            Tuple indicating mismatch detection and score.
        """

        # Extract and normalize temporal entities.
        prompt_dates = self.extract_temporal_entities(prompt)
        print("PROMPT: ", prompt_dates)
        output_dates = self.extract_temporal_entities(output)
        print("OUTPUT: ", output_dates)
        context_dates = self.extract_temporal_entities(context)
        print("CONTEXT: ", context_dates)

        # Compare temporal entities across segments.
        mismatch_score = 1 - self.compare_temporal_entities(
            prompt_dates, output_dates, context_dates
        )

        # Define mismatch detection threshold.
        threshold = 0.5
        mismatch_detected = mismatch_score > threshold

        return output, mismatch_detected, mismatch_score

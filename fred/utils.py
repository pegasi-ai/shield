import re
import string
from dateutil.parser import parse as date_parse
from dateutil.parser import ParserError
import json
import json_repair
from json_repair import repair_json
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def contain_nested_tags(tags, text):
    """
    Check if any tags in the given text are nested within each other.

    This function looks for improper nesting among a predefined set of tags in the input text.
    It returns `True` if any tag is found to be nested within another tag of a different type.

    Args:
        tags (list of str): A list of tag names (e.g., ["entity", "numerical", "relation"]).
        text (str): The input text that may contain tagged content.

    Returns:
        bool: True if nested tags are detected, otherwise False.

    Example:
        >>> text = <contradictory><delete>This calculation shows that Project K actually contributed positively to the 2013 SGA, increasing
        >>>        it by <numerical><mark>2%</mark><delete>3.5%</delete></numerical>.</delete><contradictory>
        >>> contain_nested_tags(tags, text)
        True
    """
    tag_pattern = re.compile(r"</?(" + "|".join(tags) + r")>")
    stack = []
    errors = []

    for match in tag_pattern.finditer(text):
        tag = match.group(0)
        tag_name = match.group(1)
        position = match.start()

        if tag.startswith("</"):
            if stack:
                open_tag, open_pos = stack.pop()
                if any(inner_tag != open_tag for inner_tag, _ in stack):
                    # Found a nested tag
                    errors.append((open_tag, tag_name, open_pos, position))
        else:
            stack.append((tag_name, position))

    return errors!=[]


def recover_original_string(text):
    """
    Recover the original text by removing annotated error tags.

    This function processes annotated text that contains custom XML-like tags indicating various types 
    of hallucination or factual error annotations (e.g., <contradictory>, <unverifiable>, <entity>, etc.).
    It recovers the original form of the text by:
      - Removing sections marked as <contradictory> or <unverifiable> entirely.
      - Replacing <entity>, <numerical>, <temporal>, and <relation> tags with the original (deleted) content 
        inside <delete> tags, effectively discarding the marked content.

    Args:
        text (str): The input text containing error annotations with custom tags.

    Returns:
        str: The cleaned text with the original content recovered and annotations removed.

    Example:
        >>> input_text = '<entity><delete>Revenue</delete><mark>Profits</mark></entity> increased.'
        >>> recover_original_string(input_text)
        'Revenue increased.'
    """
    # Step 1: Remove contradictory sections entirely
    text = re.sub(r"<contradictory>.*?</contradictory>", "", text, flags=re.DOTALL)

    # Step 2: Remove unverifiable sections entirely
    text = re.sub(r"<unverifiable>.*?</unverifiable>", "", text, flags=re.DOTALL)
    
    # Step 3: Replace <entity><delete>...</delete><mark>...</mark></entity> with just the <delete> content
    text = re.sub(r"<entity><delete>(.*?)</delete><mark>.*?</mark></entity>", r"\1", text, flags=re.DOTALL)

    # Step 4: Replace <numerical><delete>...</delete><mark>...</mark></numerical> with just the <delete> content
    text = re.sub(r"<numerical><delete>(.*?)</delete><mark>.*?</mark></numerical>", r"\1", text, flags=re.DOTALL)

    # Step 5: Replace <temporal><delete>...</delete><mark>...</mark></temporal> with just the <delete> content
    text = re.sub(r"<temporal><delete>(.*?)</delete><mark>.*?</mark></temporal>", r"\1", text, flags=re.DOTALL)

    # Step 6: Replace <relation><delete>...</delete><mark>...</mark></relation> with just the <delete> content
    text = re.sub(r"<relation><delete>(.*?)</delete><mark>.*?</mark></relation>", r"\1", text, flags=re.DOTALL)
    
    return text.strip()


def have_same_word_sequence(str1, str2):

    def normalize_text(text):
        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        # Normalize whitespace and strip
        text = re.sub(r"\s+", " ", text).strip()
        return text
    words1 = normalize_text(str1).split()
    words2 = normalize_text(str2).split()
    return words1 == words2

def replace_tag_with_new_tag(text, old_tag, new_tag):
    """
    Replace all occurrences of <old_tag> and </old_tag> with <new_tag> and </new_tag>.

    Args:
        text (str): The input text containing tags.
        old_tag (str): The tag to be replaced.
        new_tag (str): The new tag to use.

    Returns:
        str: Text with tags replaced.
    """
    pattern = fr"</?{old_tag}>"
    return re.sub(pattern, lambda m: f"</{new_tag}>" if m.group(0).startswith("</") else f"<{new_tag}>", text)

def contains_only_nouns_or_phrases(text):
    # Process the text
    doc = nlp(text)

    # Check if all tokens are either nouns or noun phrases
    for token in doc:
        if token.pos_ not in ["NOUN", "PROPN"] and not token.dep_ == "nsubj":
            return False  # If any token is not a noun or proper noun or part of a noun phrase
    
    return True

def contains_only_verbs_adj_adv(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Check if every token is a verb, adjective, or adverb
    for token in doc:
        if token.pos_ not in ["VERB", "ADJ", "ADV"]:
            return False
    return True

def contains_only_articles_or_demonstratives(text):
    doc = nlp(text)

    for token in doc:
        # Keep only tokens that are determiners (DET) with tag DT (i.e., articles or demonstratives)
        if token.pos_ != "DET" or token.tag_ != "DT":
            return False
    return True

def replace_tagged_with_mark(text):
    # This regex matches any tag like <relation>, <entity>, etc. 
    # and replaces the entire tag with only the content of <mark>
    pattern = r"<(relation|entity|numerical|temporal)><mark>(.*?)</mark><delete>.*?</delete></\1>"
    return re.sub(pattern, r"\2", text)


def is_temporal(text):
    """
    Checks if the string contains temporal information such as years, quarters, months, or dates.
    """
    text = text.strip()

    # Match exact 4-digit years from 1900 to 2099
    if re.fullmatch(r'(19|20)\d{2}', text):
        return True

    # Match quarter + year
    if re.search(r'\bQ[1-4]\s?(19|20)\d{2}\b', text, re.IGNORECASE):
        return True

    # Match fiscal year
    if re.search(r'\bFY\s?(19|20)\d{2}\b', text, re.IGNORECASE):
        return True

    # Full or abbreviated month names only
    if re.fullmatch(r'(January|February|March|April|May|June|July|August|September|October|November|December|'
                    r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)', text, re.IGNORECASE):
        return True


    return False


def is_numerical(text):
    """
    Checks if the string represents a numerical value, such as integers, floats, percentages, or currency.
    Ignores pure temporal values like '2022'.
    """
    """
    Checks if the string represents a numerical value, such as numbers, percentages, currency,
    or values expressed with multipliers like million, billion, thousand.
    """
    text = text.strip()

    # If it's a standalone year, treat it as temporal
    if re.fullmatch(r'(19|20)\d{2}', text):
        return False

    # Match numerical words like 8 million, 2.5 billion, etc.
    if re.match(r'^[\d,.]+(\.\d+)?\s*(thousand|million|billion|trillion|k|m|bn)?$', text, re.IGNORECASE):
        return True

    # Remove formatting characters
    cleaned = re.sub(r'[,\$%()]', '', text)
    try:
        float(cleaned)
        return True
    except ValueError:
        return False

def match_lower_precision(a, b):
    # Convert floats to string to count decimal digits
    def decimal_places(x):
        s = format(x, 'f').rstrip('0')  # remove trailing zeros
        if '.' in s:
            return len(s.split('.')[1])
        return 0

    # Determine the lower precision (fewer decimal places)
    precision = min(decimal_places(a), decimal_places(b))

    # Round both numbers to that precision
    return round(a, precision), round(b, precision)


## for converting format
# removes <mark>, <delete>, and other error tokens from passage
def remove_error_tags(token_passage):
    error_tokens = ['<entity>', '<relation>', '<contradictory>', '<unverifiable>', '<invented>', '<subjective>', '<temporal>', '<numerical>', '<mark>',
                    '</entity>', '</relation>', '</contradictory>', '</unverifiable>', '</invented>', '</subjective>', '</temporal>', '</numerical>', '</mark>']
    for tok in error_tokens:
        token_passage = token_passage.replace(tok, "")
    if "<delete>" not in token_passage:
        return token_passage
    else:
        count = 0
        while "<delete>" in token_passage and "</delete>" in token_passage and count < 10:
            next_delete_start_index = token_passage.index("<delete>")
            next_delete_end_index = token_passage.index("</delete>")
            deleted_part = token_passage[next_delete_start_index:(next_delete_end_index + 9)]
            token_passage = token_passage.replace(deleted_part, "")
            count += 1

    token_passage = token_passage.replace("</s>", "")
    return token_passage
    
def swap_error_tags(token_passage):
    token_passage = token_passage.replace("<mark>", "<d>")
    token_passage = token_passage.replace("</mark>", "</d>")
    token_passage = token_passage.replace("<delete>", "<mark>")
    token_passage = token_passage.replace("</delete>", "</mark>")
    token_passage = token_passage.replace("<d>", "<delete>")
    token_passage = token_passage.replace("</d>", "</delete>")
    token_passage = token_passage.replace("<contradictory>", "<contradictory><delete>")
    token_passage = token_passage.replace("</contradictory>", "</delete><contradictory>")
  #  print(token_passage)
    token_passage = token_passage.replace("</s>", "")
    return token_passage

## for recovering text from response
def remove_tagged_spans(text):
    # Define patterns for all tags
    tags = ["unverifiable", "contradictory", "invented", "subjective"]
    for tag in tags:
        # Pattern to match <tag>...</tag>, non-greedy match in between
        pattern = fr"<{tag}>.*?</{tag}>"
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text

## for tag statistics
def contains_error_tags(error_tags, text):
    """
    Checks if the given text contains any of the defined error tags.

    Args:
        text (str): The input text.

    Returns:
        bool: True if any error tag is present, False otherwise.
    """
    # Combine tags into a regex pattern
    pattern = re.compile("|".join(re.escape(tag) for tag in error_tags))
    return bool(pattern.search(text))

## for postprocessing from inference
def extract_numerical_value(text):
    """
    Extracts the first numerical value from a string.
    Assumes there is only one numerical value in the text.
    Supports integers, floats, negative numbers, and percentage formats.
    
    Example:
    >>> extract_numerical_value("The growth was 3.82% last year.")
    3.82
    """
    match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if match:
        number_str = match.group(0).replace(',', '')  # remove commas for large numbers
        return float(number_str)
    return None

# for error insertion
def extract_failed_generation_json(error_message):
    match = re.search(r"'failed_generation': '(.*)'", error_message, re.DOTALL)
    if match:
        json_part = match.group(1)
        # Clean the extracted part to ensure it's valid JSON
        json_part = json_part.replace(
            "'", '"'
        )  # Replace single quotes with double quotes

        # Additional cleaning if necessary (escaping any unescaped double quotes, etc.)
        json_part = re.sub(r"\\n", "", json_part)  # Remove new line characters
        json_part = re.sub(r"\s+", " ", json_part)  # Remove extra spaces

        json_repair = repair_json(json_part)

        try:
            # Attempt to parse the cleaned JSON part
            _ = json.loads(json_repair)
            return json_repair
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse the failed_generation JSON: {str(e)}")
    else:
        raise ValueError("Malformed error message, missing 'failed_generation' part.")

def extract_wrapped_json(text: str) -> str:
    """Extract JSON content from a response that's wrapped in markdown code blocks.

    Args:
        text (str): The text containing JSON content wrapped in ```json code blocks

    Returns:
        str: The extracted JSON content if found, otherwise returns the original text
    """
    if "```json" in text:
        json_pattern = r"```json\n([\s\S]*?)```"
        if match := re.search(json_pattern, text):
            return match.group(1).strip()
    return text

# from prompt
def extract_references_and_passage(text: str):
    pattern = r"Read the following references:(.*?)(?=Please identify all the errors)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reference = match.group(1).strip()

    # Find where the passage starts
    passage_split_marker = "Text:"
    if passage_split_marker in text:
        passage = text.split(passage_split_marker, 1)[1].strip()
    else:
        passage = ""

    return reference, passage


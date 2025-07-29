from typing import Dict, List
import os
import spacy

# Load spaCy English model once at module level
spacy_model_path = os.path.join(os.path.dirname(__file__), "..", "model_weights", "spacy", "en_core_web_sm","en_core_web_sm-3.8.0")
nlp = spacy.load(spacy_model_path)


def extract_named_entities(text: str) -> List[Dict[str, str]]:
    """
    Extract named entities from text using spaCy NER.
    Args:
        text (str): The input text to analyze.
    Returns:
        List[Dict[str, str]]: List of entities with their label and text.
    """
    doc = nlp(text)
    return [{"entity": ent.label_, "text": ent.text} for ent in doc.ents]

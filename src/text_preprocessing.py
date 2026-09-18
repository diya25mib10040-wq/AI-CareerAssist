import re


def clean_text(text):
    """
    Clean text by converting it to lowercase,
    removing special characters, and removing extra spaces.
    """

    text = str(text)
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
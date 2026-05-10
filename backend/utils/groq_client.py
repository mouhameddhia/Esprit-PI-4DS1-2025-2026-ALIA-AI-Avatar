# Re-export the shared Groq client from the alia_nlp package so that
# backend routes can import it via a relative path without duplicating logic.
from alia_nlp.utils.groq_client import get_groq_client, get_model  # noqa: F401

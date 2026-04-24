from NLP.pipeline.language import detect_language


def test_detect_language_english_medical_query():
    text = "Can you summarize the efficacy and safety profile for this treatment?"
    assert detect_language(text) == "en"


def test_detect_language_french_without_accents():
    text = "Bonjour docteur pouvez vous resumer les donnees d efficacite"
    assert detect_language(text) == "fr"


def test_detect_language_arabic_script():
    text = "مرحبا دكتور هل يمكنك شرح الجرعة الموصى بها"
    assert detect_language(text) == "ar"


def test_detect_language_arabizi_pattern():
    text = "marhba docteur, chno hiya jar3a li khasni n9oul"
    assert detect_language(text) == "ar"


def test_detect_language_ambiguous_short_text_returns_unknown():
    text = "ok thanks"
    assert detect_language(text) in {"en", "unknown"}

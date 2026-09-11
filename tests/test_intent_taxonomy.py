import pytest

from src.intent import taxonomy


def test_taxonomy_version_exists():
    assert hasattr(taxonomy, "TAXONOMY_VERSION") and taxonomy.TAXONOMY_VERSION


def test_required_fields_and_uniqueness():
    ids = []
    examples = []
    required = {"id", "name", "definition", "inclusion_criteria", "exclusion_criteria", "examples"}

    for intent in taxonomy.TAXONOMY:
        assert required.issubset(set(intent.keys())), f"Missing fields in intent {intent.get('id')!r}"
        assert intent["id"] not in ids, f"Duplicate intent id: {intent['id']}"
        ids.append(intent["id"])

        # definition, inclusion and exclusion must be non-empty strings
        assert isinstance(intent["definition"], str) and intent["definition"].strip()
        assert isinstance(intent["inclusion_criteria"], str) and intent["inclusion_criteria"].strip()
        assert isinstance(intent["exclusion_criteria"], str) and intent["exclusion_criteria"].strip()

        # examples list
        assert isinstance(intent["examples"], list)
        for ex in intent["examples"]:
            assert isinstance(ex, str) and ex.strip()
            examples.append(ex)

    # 'other' and 'unclear' must exist
    assert "other" in ids
    assert "unclear" in ids

    # unique example texts across intents
    assert len(examples) == len(set(examples)), "Duplicate example text across intents"


def test_primary_intent_count():
    primary = [i for i in taxonomy.TAXONOMY if i["id"] not in ("other", "unclear")]
    # requirement: approximately 10-12 primary intents
    assert 10 <= len(primary) <= 12, f"Primary intent count is {len(primary)}, expected 10-12"


def test_intent_ids_unique():
    ids = [i["id"] for i in taxonomy.TAXONOMY]
    assert len(ids) == len(set(ids))

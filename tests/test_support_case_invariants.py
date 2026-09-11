from scripts.build_support_cases import TARGET_BRAND, split_thread_issue_components, validate_case_invariants


def test_split_thread_issue_components_keeps_legitimate_escalation_chain():
    tweet_rows = {
        "c1": {
            "tweet_id": "c1",
            "author_id": "customer_a",
            "inbound": True,
            "created_at": "Mon Jan 01 00:00:00 +0000 2024",
            "text": "first issue",
            "in_response_to_tweet_id": None,
        },
        "u1": {
            "tweet_id": "u1",
            "author_id": TARGET_BRAND,
            "inbound": False,
            "created_at": "Mon Jan 01 00:05:00 +0000 2024",
            "text": "we can help",
            "in_response_to_tweet_id": "c1",
        },
        "c2": {
            "tweet_id": "c2",
            "author_id": "customer_a",
            "inbound": True,
            "created_at": "Mon Jan 01 00:10:00 +0000 2024",
            "text": "follow up",
            "in_response_to_tweet_id": "u1",
        },
    }

    components = split_thread_issue_components(["c1", "u1", "c2"], tweet_rows)

    assert components == [["c1", "u1", "c2"]]


def test_split_thread_issue_components_splits_new_issue_when_thread_structure_supports_it():
    tweet_rows = {
        "c1": {
            "tweet_id": "c1",
            "author_id": "customer_a",
            "inbound": True,
            "created_at": "Mon Jan 01 00:00:00 +0000 2024",
            "text": "issue one",
            "in_response_to_tweet_id": None,
        },
        "u1": {
            "tweet_id": "u1",
            "author_id": TARGET_BRAND,
            "inbound": False,
            "created_at": "Mon Jan 01 00:05:00 +0000 2024",
            "text": "issue one follow-up",
            "in_response_to_tweet_id": "c1",
        },
        "c2": {
            "tweet_id": "c2",
            "author_id": "customer_a",
            "inbound": True,
            "created_at": "Mon Jan 01 00:20:00 +0000 2024",
            "text": "new issue",
            "in_response_to_tweet_id": None,
        },
        "u2": {
            "tweet_id": "u2",
            "author_id": TARGET_BRAND,
            "inbound": False,
            "created_at": "Mon Jan 01 00:25:00 +0000 2024",
            "text": "handling new issue",
            "in_response_to_tweet_id": "c2",
        },
    }

    components = split_thread_issue_components(["c1", "u1", "c2", "u2"], tweet_rows)

    assert sorted(tuple(item) for item in components) == sorted((("c1", "u1"), ("c2", "u2")))


def test_validate_case_invariants_allows_external_parent_reference_and_rejects_bad_links():
    valid_external_parent_case = {
        "case_id": "case_0001",
        "customer_id": "customer_a",
        "full_thread": [
            {"tweet_id": "t1", "author_id": "customer_a", "parent_tweet_id": "old_customer_tweet"},
            {"tweet_id": "u1", "author_id": TARGET_BRAND, "parent_tweet_id": "t1", "text": "we have your order info"},
        ],
        "customer_messages": [{"tweet_id": "t1"}],
        "uber_messages": [{"tweet_id": "u1", "parent_tweet_id": "t1", "text": "we have your order info"}],
    }

    invalid_customer_parent_case = {
        "case_id": "case_0002",
        "customer_id": "customer_a",
        "full_thread": [
            {"tweet_id": "t1", "author_id": "customer_a", "parent_tweet_id": None},
            {"tweet_id": "u1", "author_id": TARGET_BRAND, "parent_tweet_id": "t2", "text": "please message us"},
            {"tweet_id": "t2", "author_id": "customer_b", "parent_tweet_id": None},
        ],
        "customer_messages": [{"tweet_id": "t1"}],
        "uber_messages": [{"tweet_id": "u1", "parent_tweet_id": "t2", "text": "please message us"}],
    }

    duplicate_case = {
        "case_id": "case_0003",
        "customer_id": "customer_a",
        "full_thread": [
            {"tweet_id": "t1", "author_id": "customer_a", "parent_tweet_id": None},
            {"tweet_id": "u1", "author_id": TARGET_BRAND, "parent_tweet_id": "t1", "text": "we resolved it"},
        ],
        "customer_messages": [{"tweet_id": "t1"}],
        "uber_messages": [{"tweet_id": "u1", "parent_tweet_id": "t1", "text": "we resolved it"}],
    }

    mixed_customer_case = {
        "case_id": "case_0004",
        "customer_id": "customer_a",
        "full_thread": [
            {"tweet_id": "t1", "author_id": "customer_a", "parent_tweet_id": None},
            {"tweet_id": "t2", "author_id": "customer_b", "parent_tweet_id": None},
            {"tweet_id": "u1", "author_id": TARGET_BRAND, "parent_tweet_id": "t1", "text": "we can help"},
        ],
        "customer_messages": [{"tweet_id": "t1"}, {"tweet_id": "t2"}],
        "uber_messages": [{"tweet_id": "u1", "parent_tweet_id": "t1", "text": "we can help"}],
    }

    misaddressed_case = {
        "case_id": "case_0005",
        "customer_id": "customer_a",
        "full_thread": [
            {"tweet_id": "t1", "author_id": "customer_a", "parent_tweet_id": None},
            {"tweet_id": "t2", "author_id": "customer_a", "parent_tweet_id": "t1"},
            {"tweet_id": "u1", "author_id": TARGET_BRAND, "parent_tweet_id": "t3", "text": "please message us"},
            {"tweet_id": "t3", "author_id": "customer_b", "parent_tweet_id": None},
        ],
        "customer_messages": [{"tweet_id": "t1"}, {"tweet_id": "t2"}],
        "uber_messages": [{"tweet_id": "u1", "parent_tweet_id": "t3", "text": "please message us"}],
    }

    issues = validate_case_invariants([
        valid_external_parent_case,
        invalid_customer_parent_case,
        duplicate_case,
        mixed_customer_case,
        misaddressed_case,
        duplicate_case,
    ])

    assert "case_0001:empty_case" not in issues
    assert any("case_0002:misaddressed_uber_response" in issue for issue in issues)
    assert any("case_0004:mixed_customer_ids" in issue for issue in issues)
    assert any("duplicate_tweet_id_across_cases:t1" in issue for issue in issues)
    assert any("case_0005:misaddressed_uber_response" in issue for issue in issues)
    assert "case_0001:broken_parent_reference" not in "\n".join(issues)

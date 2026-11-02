"""Retrieval quality: the right help-center section must be found for realistic questions."""

import pytest

from support_agent.kb import chunk_markdown, tokenize

# (question, expected chunk id) -- the expected section must be the top hit.
TOP1 = [
    ("How long does standard shipping take to Canada?", "shipping#international-shipping"),
    ("How much does standard shipping cost in the US?", "shipping#domestic-shipping-united-states"),
    ("My package says delivered but I can't find it", "shipping#delayed-or-lost-packages"),
    ("how long do refunds take?", "returns-and-exchanges#refund-timing"),
    ("Can I exchange for a different size?", "returns-and-exchanges#exchanges"),
    ("warranty claim broken zipper", "warranty#what-the-warranty-covers"),
    ("do you accept paypal?", "payments-and-pricing#accepted-payment-methods"),
    ("Do you price match?", "payments-and-pricing#price-adjustments"),
    ("I need to cancel my order", "order-changes-and-cancellations#cancellation-window"),
    ("How do I wash my down jacket?", "product-care#down-jackets-and-sleeping-bags"),
    ("do gift cards expire?", "loyalty-and-gift-cards#gift-cards"),
    ("what are your store opening hours", "contact-and-store-hours#flagship-store"),
]

# (question, expected chunk id) -- must appear in the top 3 (recall@3).
TOP3 = [
    ("Can I return boots I wore outside?", "returns-and-exchanges#return-window"),
    ("what size jacket for a 100 cm chest?", "sizing-guide#men-s-tops-and-jackets"),
    ("how many points do I need for a reward?", "loyalty-and-gift-cards#trailhead-rewards-membership"),
    ("can I change the shipping address after ordering", "order-changes-and-cancellations#changing-the-shipping-address"),
]


@pytest.mark.parametrize("question,expected", TOP1)
def test_top1(kb, question, expected):
    assert kb.search(question, top_k=1)[0].chunk.id == expected


@pytest.mark.parametrize("question,expected", TOP3)
def test_recall_at_3(kb, question, expected):
    assert expected in [r.chunk.id for r in kb.search(question, top_k=3)]


@pytest.mark.parametrize("question", ["Do you ship to Brazil?", "tell me a joke", "can I bring my dog to the store"])
def test_off_topic_questions_are_low_confidence(tools, question):
    assert tools.run("search_knowledge_base", {"query": question})["confidence"] == "low"


def test_all_documents_are_indexed(kb):
    assert len(kb.documents) == 9
    assert all("#" in c.id for c in kb.chunks)
    assert len({c.id for c in kb.chunks}) == len(kb.chunks), "chunk ids must be unique"


def test_chunking_splits_on_h2_and_keeps_title():
    md = "# Title\n\nIntro text.\n\n## First Part\n\nAlpha.\n\n## Second Part\n\nBeta."
    chunks = chunk_markdown("doc.md", md)
    assert [c.id for c in chunks] == ["doc#overview", "doc#first-part", "doc#second-part"]
    assert {c.title for c in chunks} == {"Title"}


def test_tokenizer_stems_and_drops_stopwords():
    assert tokenize("Where are my returns shipping?") == ["return", "ship"]
    assert tokenize("The delivery was broken") == ["deliver", "break"]

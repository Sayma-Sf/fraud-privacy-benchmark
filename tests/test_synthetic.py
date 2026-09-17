import pytest

from src.ingest.load import RAW_COLUMNS, TARGET
from src.privacy.synthetic import ClassConditionalSynthesizer, fidelity_scores


@pytest.fixture(scope="module")
def generator(transactions):
    return ClassConditionalSynthesizer.fit(transactions, "gaussian_copula", seed=0)


def test_sample_keeps_schema_and_fraud_rate(generator, transactions):
    synthetic = generator.sample(2000, seed=0)
    assert list(synthetic.columns) == [*RAW_COLUMNS, TARGET]
    assert len(synthetic) == 2000
    assert synthetic[TARGET].sum() == round(2000 * transactions[TARGET].mean())
    assert (synthetic["Amount"] >= 0).all()


def test_max_rows_per_class_never_drops_fraud(transactions):
    capped = ClassConditionalSynthesizer.fit(
        transactions, "gaussian_copula", max_rows_per_class=500, seed=0
    )
    # The fraud rate comes from the full training set, not the capped sample.
    assert capped.fraud_rate == pytest.approx(transactions[TARGET].mean())


def test_unknown_synthesizer_is_rejected(transactions):
    with pytest.raises(ValueError, match="Unknown synthesizer"):
        ClassConditionalSynthesizer.fit(transactions, "gan9000")


def test_fidelity_scores_are_bounded(generator, transactions):
    scores = fidelity_scores(transactions, generator.sample(1000, seed=1), max_rows=1000)
    assert set(scores) == {"overall", "column_shapes", "column_pair_trends"}
    assert all(0.0 <= s <= 1.0 for s in scores.values())

import pytest
import torch

from pie_utils.metric.weak_span_based_f1 import (
    SpanBasedF1WeakMeasure,
    compute_metrics,
    get_weak_match,
    has_weak_overlap,
    increase_span_end_index,
)

# Predicted Tag Sequence :
# [[O,B-own_claim,I-own_claim,I-own_claim,I-own_claim,I-own_claim],[I-own_claim,I-own_claim,I-own_claim,B-data,I-data,I-data]]
LOGITS = torch.tensor(
    [
        [
            [
                10.351420402526855,
                -3.2028937339782715,
                -2.790834665298462,
                -2.648576021194458,
                -3.457158327102661,
                -2.2753586769104004,
                -3.092430591583252,
            ],
            [
                -0.728147566318512,
                -0.3243664503097534,
                -3.784968137741089,
                -0.566190779209137,
                -3.6368443965911865,
                11.714195251464844,
                0.874237596988678,
            ],
            [
                -2.2986810207366943,
                -3.2249557971954346,
                -2.213132381439209,
                -3.176085948944092,
                -1.201798439025879,
                -2.5730767250061035,
                10.175483703613281,
            ],
            [
                -2.399078845977783,
                -3.7054524421691895,
                -2.732715606689453,
                -3.1169371604919434,
                -1.3757351636886597,
                -2.5311129093170166,
                10.664148330688477,
            ],
            [
                -1.685034155845642,
                -3.0339670181274414,
                -2.960148572921753,
                -3.0828864574432373,
                -1.1170631647109985,
                -2.528775215148926,
                10.603050231933594,
            ],
            [
                -2.307305097579956,
                -3.424372673034668,
                -2.1860435009002686,
                -3.231368064880371,
                -1.3228000402450562,
                -2.4107933044433594,
                10.90400218963623,
            ],
        ],
        [
            [
                -2.7964279651641846,
                -2.7935304641723633,
                -2.4361226558685303,
                -2.8135015964508057,
                -1.101932406425476,
                0.03466780483722687,
                10.032923698425293,
            ],
            [
                -2.841414451599121,
                -2.925645351409912,
                -2.197348117828369,
                -3.377918004989624,
                -1.2031382322311401,
                -1.7914681434631348,
                10.230854034423828,
            ],
            [
                -2.645219087600708,
                -3.736269235610962,
                -2.1186366081237793,
                -3.4355688095092773,
                -0.8101659417152405,
                -2.3279330730438232,
                10.49844741821289,
            ],
            [
                -2.3094680309295654,
                -3.7092173099517822,
                -1.942619800567627,
                10.281001091003418,
                -3.1347908973693848,
                -1.1283811330795288,
                -2.2268154621124268,
            ],
            [
                -3.747760772705078,
                -2.015471935272217,
                -3.379420757293701,
                -2.101996421813965,
                10.566914558410645,
                -3.555516242980957,
                -2.556119441986084,
            ],
            [
                -3.747760772705078,
                -2.015471935272217,
                -3.379420757293701,
                -2.101996421813965,
                10.566914558410645,
                -3.555516242980957,
                -2.556119441986084,
            ],
        ],
        [
            [
                -2.7964279651641846,
                -2.7935304641723633,
                -2.4361226558685303,
                -2.8135015964508057,
                -1.101932406425476,
                0.03466780483722687,
                10.032923698425293,
            ],
            [
                -2.841414451599121,
                -2.925645351409912,
                -2.197348117828369,
                -3.377918004989624,
                -1.2031382322311401,
                -1.7914681434631348,
                10.230854034423828,
            ],
            [
                -2.645219087600708,
                -3.736269235610962,
                -2.1186366081237793,
                -3.4355688095092773,
                -0.8101659417152405,
                -2.3279330730438232,
                10.49844741821289,
            ],
            [
                -2.3094680309295654,
                -3.7092173099517822,
                -1.942619800567627,
                10.281001091003418,
                -3.1347908973693848,
                -1.1283811330795288,
                -2.2268154621124268,
            ],
            [
                -3.747760772705078,
                -2.015471935272217,
                -3.379420757293701,
                -2.101996421813965,
                10.566914558410645,
                -3.555516242980957,
                -2.556119441986084,
            ],
            [
                -3.747760772705078,
                -2.015471935272217,
                -3.379420757293701,
                -2.101996421813965,
                10.566914558410645,
                -3.555516242980957,
                -2.556119441986084,
            ],
        ],
    ]
)

# Original Tag sequence:
# [[PAD,O,B-own_claim,I-own_claim,I-own_claim,PAD],[PAD,PAD,B-background_claim,I-background_claim,B-data,PAD]]
TARGETS = torch.tensor(
    [
        [
            -100,
            0,
            5,
            6,
            6,
            -100,
        ],
        [
            -100,
            -100,
            1,
            2,
            3,
            -100,
        ],
        [
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
        ],
    ]
)

MASKS = TARGETS != -100

LABEL_TO_ID = {
    "O": 0,
    "B-background_claim": 1,
    "I-background_claim": 2,
    "B-data": 3,
    "I-data": 4,
    "B-own_claim": 5,
    "I-own_claim": 6,
}


@pytest.mark.parametrize(
    "masks",
    [MASKS, None],
)
def test_update(masks):
    """Given instance contains four spans, two of which are own_claim and one each for data and
    background_claim. Model predicts one span labelled as own_claim correctly. Moreover, it
    predicts a span as data falsely. Therefore, resulting metric should contain 1 true positive
    count for 'own_claim', 1 false positive count for 'data'. and 1 false negative count for each
    label.

    predicted spans : [own_claim (1,5)] , [own_claim (0,2), data (3,5)]
    gold spans : [own_claim (2,4)] , [background_claim (2,3), data (4)]
    tp : own_claim=1, data=1, background_claim=0
    fp : own_claim=1, data=0, background_claim=0
    fn : own_claim=0, data=0, background_claim=1
    """

    metric = SpanBasedF1WeakMeasure(label_to_id=LABEL_TO_ID, return_metric="micro/f1")
    assert torch.equal(metric.tp, torch.zeros([3], dtype=torch.int64))
    assert torch.equal(metric.fp, torch.zeros([3], dtype=torch.int64))
    assert torch.equal(metric.fn, torch.zeros([3], dtype=torch.int64))
    metric.update(preds=torch.tensor(LOGITS), targets=torch.tensor(TARGETS), masks=masks)
    expected_true_positives = torch.zeros([3], dtype=torch.int64)
    expected_true_positives[metric._span_classes.index("own_claim")] = 1
    expected_true_positives[metric._span_classes.index("data")] = 1
    assert torch.equal(metric.tp, expected_true_positives)

    expected_false_positives = torch.zeros([3], dtype=torch.int64)
    expected_false_positives[metric._span_classes.index("own_claim")] = 1
    assert torch.equal(metric.fp, expected_false_positives)

    expected_false_negatives = torch.zeros([3], dtype=torch.int64)
    expected_false_negatives[metric._span_classes.index("background_claim")] = 1
    assert torch.equal(metric.fn, expected_false_negatives)


def test_update_with_prediction_map():
    """This test is similar to the last test but uses the prediction_map to obtain correct label in
    each batch of sequence.

    In our case, the label classes are same for all batch sequences. So prediction_map doesn't really make much
    difference. Accurate use case can be found here:
    https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/tests/training/metrics/span_based_f1_measure_test.py#L39
    """
    prediction_map = torch.tensor(
        [list(LABEL_TO_ID.values()), list(LABEL_TO_ID.values()), list(LABEL_TO_ID.values())]
    )
    metric = SpanBasedF1WeakMeasure(label_to_id=LABEL_TO_ID, return_metric="micro/f1")
    assert torch.equal(metric.tp, torch.zeros([3], dtype=torch.int64))
    assert torch.equal(metric.fp, torch.zeros([3], dtype=torch.int64))
    assert torch.equal(metric.fn, torch.zeros([3], dtype=torch.int64))
    metric.update(
        preds=torch.tensor(LOGITS), targets=torch.tensor(TARGETS), prediction_map=prediction_map
    )
    expected_true_positives = torch.zeros([3], dtype=torch.int64)
    expected_true_positives[metric._span_classes.index("own_claim")] = 1
    expected_true_positives[metric._span_classes.index("data")] = 1
    assert torch.equal(metric.tp, expected_true_positives)

    expected_false_positives = torch.zeros([3], dtype=torch.int64)
    expected_false_positives[metric._span_classes.index("own_claim")] = 1
    assert torch.equal(metric.fp, expected_false_positives)

    expected_false_negatives = torch.zeros([3], dtype=torch.int64)
    expected_false_negatives[metric._span_classes.index("background_claim")] = 1
    assert torch.equal(metric.fn, expected_false_negatives)


@pytest.mark.parametrize(
    "return_metric",
    ["own_claim/f1", "own_claim/precision", "own_claim/recall", "macro/f1", "micro/f1"],
)
def test_compute(return_metric):
    """Given instance contains four spans, two of which are own_claim and one each for data and
    background_claim. Model predicts one span labelled as own_claim correctly. Moreover, it
    predicts a span as data falsely. Based on these predictions, scores will be calculated for
    different return metrics.

    own_claim/f1 : 2*1*0.5/(0.5+1) = 0.667
    own_claim/precision : 1/(1+1) = 0.5
    own_claim/recall : 1/(1+0) = 1
    macro/f1 : (0.667 + 1 + 0)/3 = 0.556
    micro/f1 : (2*0.667*0.667)/(0.667+0.667) = 0.667
    """
    metric = SpanBasedF1WeakMeasure(label_to_id=LABEL_TO_ID, return_metric=return_metric)
    metric.update(
        preds=torch.tensor(LOGITS), targets=torch.tensor(TARGETS), masks=torch.tensor(MASKS)
    )
    scores = metric.compute()
    if return_metric == "own_claim/f1":
        assert pytest.approx(scores) == torch.tensor(0.6666666865348816)
    if return_metric == "own_claim/precision":
        assert torch.eq(scores, torch.tensor(0.5))
    if return_metric == "own_claim/recall":
        assert torch.eq(scores, torch.tensor(1))
    if return_metric == "macro/f1":
        assert pytest.approx(scores) == torch.tensor(0.5555555820465088)
    if return_metric == "micro/f1":
        assert pytest.approx(scores) == torch.tensor(0.6666666865348816)


def test_has_weak_overlap():
    #  checks if overlap in span is at least half of the length of the shorter span.

    weak_overlapping_indices = ((0, 5), (3, 6))
    no_weak_overlapping_indices = ((0, 5), (4, 9))
    touching_indices = ((0, 5), (5, 7))
    non_touching_indices = ((0, 5), (6, 7))
    containing_indices = ((0, 9), (3, 6))

    assert has_weak_overlap(weak_overlapping_indices[0], weak_overlapping_indices[1])
    assert not has_weak_overlap(no_weak_overlapping_indices[0], no_weak_overlapping_indices[1])
    assert not has_weak_overlap(touching_indices[0], touching_indices[1])
    assert not has_weak_overlap(non_touching_indices[0], non_touching_indices[1])
    assert has_weak_overlap(containing_indices[0], containing_indices[1])

    assert has_weak_overlap(weak_overlapping_indices[1], weak_overlapping_indices[0])
    assert not has_weak_overlap(no_weak_overlapping_indices[1], no_weak_overlapping_indices[0])
    assert not has_weak_overlap(touching_indices[1], touching_indices[0])
    assert not has_weak_overlap(non_touching_indices[1], non_touching_indices[0])
    assert has_weak_overlap(containing_indices[1], containing_indices[0])


def test_compute_metrics():
    true_positives = 10
    false_positives = 8
    false_negatives = 4
    true_precision = 10 / (10 + 8)
    true_recall = 10 / (10 + 4)
    true_f1 = (2 * true_recall * true_precision) / (true_recall + true_precision)
    precision, recall, f1 = compute_metrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    assert precision == true_precision
    assert recall == true_recall
    assert f1 == true_f1


def test_increase_span_end_index():
    # Resulting span should have end index adjusted by the value of the offset, i.e. the offset value should be added
    # to the end index of the span
    span = ("city", (0, 0))
    new_span = increase_span_end_index(span=span, offset=1)
    assert new_span == ("city", (0, 1))

    original_span = increase_span_end_index(span=new_span, offset=-1)
    assert original_span == span


def test_get_weak_match():
    # Since predicted span is contained in first span of the gold span list, it will be considered as the weak match.
    span = ("city", (0, 0))
    spans = [("city", (0, 3)), ("person", (3, 5)), ("person", (6, 9))]
    match = get_weak_match(span=span, gold_spans=spans)
    assert match == ("city", (0, 3))

    # Predicted span is not contained in the second gold span, therefore it is not considered as a match.
    span = ("person", (3, 5))
    spans = [("city", (0, 3)), ("person", (5, 9))]
    match = get_weak_match(span=span, gold_spans=spans)
    assert match is None

    # Here predicted span is partly inside the second gold span, there we have a weak match
    span = ("person", (3, 5))
    spans = [("city", (0, 3)), ("person", (4, 7))]
    match = get_weak_match(span=span, gold_spans=spans)
    assert match == ("person", (4, 7))

    # Here predicted span is partly inside the second gold span but not enough to be considered as a weak match.
    span = ("person", (3, 6))
    spans = [("city", (0, 3)), ("person", (5, 9))]
    match = get_weak_match(span=span, gold_spans=spans)
    assert match is None


@pytest.mark.parametrize(
    "inclusive_end_index",
    [True, False],
)
def test_get_weak_match_with_inclusive_end_index(inclusive_end_index):
    # Here in the given span if we consider end index to be part of span then we have weak match with the first span
    # in the list of gold spans. However, if we do not consider end index as inclusive to the span, then there is no
    # overlap and hence no weak match.
    span = ("city", (0, 2))
    spans = [("city", (2, 3)), ("person", (3, 5))]

    match = get_weak_match(span=span, gold_spans=spans, inclusive_end_index=inclusive_end_index)
    if inclusive_end_index:
        assert match == ("city", (2, 3))
    else:
        assert match is None

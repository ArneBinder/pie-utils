from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torchmetrics import Metric

from pie_utils.sequence_tagging import tag_sequence_to_token_spans
from pie_utils.span.slice import get_overlap_len

TypedStringSpan = Tuple[str, Tuple[int, int]]
TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]


def has_weak_overlap(indices_1: Tuple[int, int], indices_2: Tuple[int, int]) -> bool:
    # checks if overlap in span is at least half of the length of the shorter span
    min_len = min(indices_1[1] - indices_1[0], indices_2[1] - indices_2[0])
    overlap_len = get_overlap_len(indices_1, indices_2)
    return 2 * overlap_len >= min_len


def increase_span_end_index(
    span: Tuple[str, Tuple[int, int]], offset: int
) -> Tuple[str, Tuple[int, int]]:
    # format of span is (label,(start,end))
    return span[0], (span[1][0], span[1][1] + offset)


def get_weak_match(
    span: Tuple[str, Tuple[int, int]],
    gold_spans: List[Tuple[str, Tuple[int, int]]],
    inclusive_end_index: bool = False,
) -> Optional[Tuple[str, Tuple[int, int]]]:
    """This method checks if the predicted span is weakly matched with any of the gold spans. If
    predicted type and gold type matches then we check if their respective indices are weakly
    overlapping or not. Weak overlap between gold and predicted span is defined in Lauscher et al.
    (2018) as overlap which should be at least half of the length of shorter span. If they are
    weakly overlapping as well then we return the matched span. In addition to this, we use
    inclusive_end_index boolean which if set adds an offset to the end index of each span in gold
    spans list and also to predicted span. Once a match is found we revert back changes to end
    index of matched span. Due to AllenNLP token based predictions, a span containing single token
    would have length of 0 since start and end index of span would be same. That is why we add an
    offset to end index.

    :param span: Predicted span instance as a tuple with span label and indices(start and end) of span.
    :param gold_spans: List of gold span instances as tuple with span label and indices(start and end) of span.
    :param inclusive_end_index: if set adds an offset to the end index of each span in gold spans list and also to
    predicted span. Once a match is found we revert back changes to end index of matched span.
    :return: gold span instance if matched with predicted span instance else None
    """
    if inclusive_end_index:
        span = increase_span_end_index(span, offset=1)
        gold_spans = [increase_span_end_index(gold_span, offset=1) for gold_span in gold_spans]

    match_found = None
    predicted_type, predicted_indices = span
    for gold_type, gold_indices in gold_spans:
        if predicted_type == gold_type and has_weak_overlap(predicted_indices, gold_indices):
            match_found = gold_type, gold_indices
            if inclusive_end_index:
                match_found = increase_span_end_index(match_found, offset=-1)
            break
    return match_found


def get_span_classes(label_vocabulary):
    return {
        label.split("-")[1]
        for label in list(label_vocabulary.values())
        if len(label.split("-")) == 2
    }


class SpanBasedF1WeakMeasure(Metric):
    def __init__(
        self,
        label_to_id: Dict[str, int],
        weak: bool = True,
        return_metric: str = "micro/f1",
        label_encoding: str = "IOB2",
        ignore_classes: List[str] = None,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._label_encoding = label_encoding
        self._ignore_classes: List[str] = ignore_classes or []
        self._weak = weak
        self._label_vocabulary = dict(zip(label_to_id.values(), label_to_id.keys()))
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
        self._span_classes = list(get_span_classes(self._label_vocabulary))
        self._num_classes = len(self._span_classes)
        self._return_metric = return_metric
        self._span_classes_to_index = {c: i for i, c in enumerate(self._span_classes)}

        def default():
            return torch.zeros([self._num_classes], dtype=torch.long)

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default(), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.BoolTensor] = None,
        prediction_map: Optional[torch.Tensor] = None,
    ):
        self.calculate_span_based_metric(
            preds=preds, targets=targets, masks=masks, prediction_map=prediction_map
        )

    def calculate_span_based_metric(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.BoolTensor] = None,
        prediction_map: Optional[torch.Tensor] = None,
    ):
        if masks is None:
            # masks = torch.ones_like(targets).bool() This will result in a tensor with all values True.
            # It will result in error since targets contain -100 as value which has no label.
            masks = targets != -100
        """
           If you actually passed gradient-tracking Tensors to a Metric, there will be
           a huge memory leak, because it will prevent garbage collection for the computation
           graph. This method ensures the tensors are detached.
           Check if it's actually a tensor in case something else was passed.
        """
        predictions, gold_labels, mask, prediction_map = (
            x.detach() if isinstance(x, torch.Tensor) else x
            for x in (preds, targets, masks, prediction_map)
        )

        sequence_lengths = masks.sum(-1)
        argmax_predictions = preds.argmax(dim=2)

        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        argmax_predictions = argmax_predictions.float()

        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]
            mask = masks[i, :]
            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            predicted_string_labels = [
                self._label_vocabulary[label_id] for label_id in sequence_prediction[mask].tolist()
            ]
            gold_string_labels = [
                self._label_vocabulary[label_id] for label_id in sequence_gold_label[mask].tolist()
            ]

            predicted_spans = tag_sequence_to_token_spans(
                tag_sequence=predicted_string_labels,
                coding_scheme=self._label_encoding,
                classes_to_ignore=self._ignore_classes,
            )
            gold_spans = tag_sequence_to_token_spans(
                tag_sequence=gold_string_labels,
                coding_scheme=self._label_encoding,
                classes_to_ignore=self._ignore_classes,
            )

            # Sorting spans so that it is deterministic all the time (handle_continued_spans may not maintain the order)
            predicted_spans = sorted(predicted_spans)
            gold_spans = sorted(gold_spans)
            for span in predicted_spans:
                span_original = span
                if self._weak:
                    span = get_weak_match(span, gold_spans, inclusive_end_index=True)
                if (not self._weak and span in gold_spans) or (self._weak and span):
                    self.tp[self._span_classes_to_index[span[0]]] += 1
                    gold_spans.remove(span)
                else:
                    if self._weak:
                        span = span_original
                    self.fp[self._span_classes.index(span[0])] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self.fn[self._span_classes.index(span[0])] += 1

    def compute(self):
        """Scores is a matrix of dimensions num_span_classes + 2 x 3.

        Here 3 signifies precision, recall and f1 and 2 signifies micro and macro averaged metric
        scores
        """

        scores = torch.zeros([self._num_classes + 2, 3])
        for i, tag in enumerate(self._span_classes):
            scores[i] = compute_metrics(self.tp[i], self.fp[i], self.fn[i])
            scores[self._num_classes + 1] += scores[i]

        # macro averaged metrics
        scores[self._num_classes + 1] = scores[self._num_classes + 1] / 3

        # micro averaged metrics
        scores[self._num_classes] = compute_metrics(
            true_positives=sum(self.tp),
            false_positives=sum(self.fp),
            false_negatives=sum(self.fn),
        )

        return self.get_return_metric(scores)

    def get_return_metric(self, scores):
        """It returns metric based on return_metric parameter.

        return_metric parameter is defined as TAG/METRIC.
        """
        tag_name, _metric_name = self._return_metric.split("/")
        metric_to_idx = {"precision": 0, "recall": 1, "f1": 2}
        if tag_name == "micro":
            index = self._num_classes
        elif tag_name == "macro":
            index = self._num_classes + 1
        else:
            index = self._span_classes.index(tag_name)
        return scores[index, metric_to_idx[_metric_name]]


def compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
    precision = true_positives / (true_positives + false_positives + 1e-13)
    recall = true_positives / (true_positives + false_negatives + 1e-13)
    f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
    return torch.tensor([precision, recall, f1_measure])

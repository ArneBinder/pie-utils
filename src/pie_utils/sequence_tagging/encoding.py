import logging
import warnings
from typing import Callable, List, Optional, Set, Tuple, TypeVar

from pie_utils.span.ill_formed_tag_sequence import (
    fix_ill_formed_tag_sequence,
    remove_ill_formed_tag_sequence,
)

logger = logging.getLogger(__name__)

# from allennlp.common.checks import ConfigurationError
# from allennlp.data.tokenizers import Token


TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]


# T = TypeVar("T", str, Token)
T = TypeVar("T")


def iob2_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: List[str] = None,
) -> List[TypedStringSpan]:
    """Given a sequence corresponding to BIO or IOB2 tags, extracts spans. Spans are inclusive and
    can be of zero length, representing a single word span.

    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "B":
            start = index
            current_span_label = label.partition("-")[2]
            index += 1
            if index < len(tag_sequence):
                label = tag_sequence[index]
            else:
                # if B is last tag in the sequence
                spans.append((current_span_label, (start, start)))
                continue
            if label[0] == "B":
                # if Ba Bb or Ba Ba
                spans.append((current_span_label, (start, start)))
                continue
            while label[0] == "I" and index < len(tag_sequence):
                # loop can end if it encounters another B or O or end of tag_sequence
                index += 1
                if index < len(tag_sequence):
                    label = tag_sequence[index]
            index -= 1
            end = index
            spans.append((current_span_label, (start, end)))
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def bioul_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: List[str] = None,
) -> List[TypedStringSpan]:
    """Given a sequence corresponding to BIOUL tags, extracts spans. Spans are inclusive and can be
    of zero length, representing a single word span. Ill-formed spans are not allowed and will
    raise `InvalidTagSequence`. This function works properly when the spans are unlabeled (i.e.,
    your labels are simply "B", "I", "O", "U", and "L").

    # Parameters

    tag_sequence : `List[str]`, required.
        The tag sequence encoded in BIOUL, e.g. ["B-PER", "L-PER", "O"].
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    # Returns

    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
    """

    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            spans.append((label.partition("-")[2], (index, index)))
        elif label[0] == "B":
            start = index
            end = index
            current_span_label = label.partition("-")[2]
            while label[0] != "L" and index < len(tag_sequence):
                index += 1
                label = tag_sequence[index]
                if label[0] == "L":
                    end = index
            spans.append((current_span_label, (start, end)))
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def to_bioul(tag_sequence: List[str], encoding: str = "IOB1") -> List[str]:
    """Given a tag sequence encoded with IOB1 labels, recode to BIOUL.

    In the IOB1 scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of span immediately following another
    span of the same type.

    In the BIO or IBO2 scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of a span.

    # Parameters

    tag_sequence : `List[str]`, required.
        The tag sequence encoded in IOB1, e.g. ["I-PER", "I-PER", "O"].
    encoding : `str`, optional, (default = `"IOB1"`).
        The encoding type to convert from. Must be either "IOB1" or "IBO2".

    # Returns

    bioul_sequence : `List[str]`
        The tag sequence encoded in IOB1, e.g. ["B-PER", "L-PER", "O"].
    """
    if encoding not in {"IOB1", "IOB2"}:
        # raise ConfigurationError(f"Invalid encoding {encoding} passed to 'to_bioul'.")
        raise ValueError(f"Invalid encoding {encoding} passed to 'to_bioul'.")

    def replace_label(full_label, new_label):
        # example: full_label = 'I-PER', new_label = 'U', returns 'U-PER'
        parts = list(full_label.partition("-"))
        parts[0] = new_label
        return "".join(parts)

    def pop_replace_append(in_stack, out_stack, new_label):
        # pop the last element from in_stack, replace the label, append
        # to out_stack
        tag = in_stack.pop()
        new_tag = replace_label(tag, new_label)
        out_stack.append(new_tag)

    def process_stack(stack, out_stack):
        # process a stack of labels, add them to out_stack
        if len(stack) == 1:
            # just a U token
            pop_replace_append(stack, out_stack, "U")
        else:
            # need to code as BIL
            recoded_stack = []
            pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                pop_replace_append(stack, recoded_stack, "I")
            pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            out_stack.extend(recoded_stack)

    # Process the tag_sequence one tag at a time, adding spans to a stack,
    # then recode them.
    bioul_sequence = []
    stack: List[str] = []

    for label in tag_sequence:
        # need to make a dict like
        # token = {'token': 'Matt', "labels": {'conll2003': "B-PER"}
        #                   'gold': 'I-PER'}
        # where 'gold' is the raw value from the CoNLL data set
        if label is None and len(stack) == 0:
            bioul_sequence.append(label)
        elif label is None and len(stack) > 0:
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label == "O" and len(stack) == 0:
            bioul_sequence.append(label)
        elif label == "O" and len(stack) > 0:
            # need to process the entries on the stack plus this one
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label[0] == "I":
            # check if the previous type is the same as this one
            # if it is then append to stack
            # otherwise this start a new entity if the type
            # is different
            if len(stack) == 0:
                if encoding == "IOB2":
                    raise InvalidTagSequence(tag_sequence)
                stack.append(label)
            else:
                # check if the previous type is the same as this one
                this_type = label.partition("-")[2]
                prev_type = stack[-1].partition("-")[2]
                if this_type == prev_type:
                    stack.append(label)
                else:
                    if encoding == "IOB2":
                        raise InvalidTagSequence(tag_sequence)
                    # a new entity
                    process_stack(stack, bioul_sequence)
                    stack.append(label)
        elif label[0] == "B":
            if len(stack) > 0:
                process_stack(stack, bioul_sequence)
            stack.append(label)
        else:
            raise InvalidTagSequence(tag_sequence)

    # process the stack
    if len(stack) > 0:
        process_stack(stack, bioul_sequence)

    return bioul_sequence


def token_spans_to_tag_sequence(
    labeled_spans: List[TypedStringSpan],
    base_sequence_length: int,
    coding_scheme: str = "IOB2",
    offset: int = 0,
    include_ill_formed: bool = True,
) -> List[str]:
    # create IOB2 encoding
    tags = ["O"] * base_sequence_length
    labeled_spans = sorted(labeled_spans, key=lambda span_annot: span_annot[1][0])
    for i, (label, (start, end)) in enumerate(labeled_spans):
        _start = start - offset
        _end = end - offset

        previous_tags = tags[_start:_end]
        assert previous_tags == ["O"] * len(
            previous_tags
        ), f"tags already set [{previous_tags}], i.e. there is an annotation overlap"
        # create IOB2 encoding
        tags[_start] = f"B-{label}"
        tags[_start + 1 : _end] = [f"I-{label}"] * (len(previous_tags) - 1)

    if include_ill_formed:
        tags = fix_encoding(tags, "IOB2")
    else:
        tags = remove_encoding(tags, "IOB2")

    # Recode the labels if necessary.
    if coding_scheme == "BIOUL":
        coded_tags = to_bioul(tags, encoding="IOB2") if tags is not None else None
    elif coding_scheme == "BOUL":
        coded_tags = to_boul(tags, encoding="IOB2") if tags is not None else None
    elif coding_scheme == "IOB2":
        coded_tags = tags
    else:
        raise ValueError(f"Unknown Coding scheme {coding_scheme}.")

    return coded_tags


def tag_sequence_to_token_spans(
    tag_sequence: List[str],
    coding_scheme: str = "IOB2",
    classes_to_ignore: List[str] = None,
    include_ill_formed: bool = True,
):

    if include_ill_formed:
        new_tag_sequence = fix_ill_formed_tag_sequence(tag_sequence, coding_scheme)
    else:
        new_tag_sequence = remove_ill_formed_tag_sequence(tag_sequence, coding_scheme)
    if coding_scheme == "BIOUL":
        labeled_spans = bioul_tags_to_spans(
            new_tag_sequence,
            classes_to_ignore=classes_to_ignore,
        )
    elif coding_scheme == "BOUL":
        labeled_spans = boul_tags_to_spans(
            new_tag_sequence,
            classes_to_ignore=classes_to_ignore,
        )
    elif coding_scheme == "IOB2":
        labeled_spans = iob2_tags_to_spans(
            tag_sequence,
            classes_to_ignore=classes_to_ignore,
        )
    else:
        raise ValueError(f"Unknown Coding scheme {coding_scheme}.")

    return labeled_spans


def to_boul(tag_sequence: List[str], encoding: str = "IOB2") -> List[str]:
    bioul_tags = to_bioul(tag_sequence=tag_sequence, encoding=encoding)
    return bioul_to_boul(bioul_tags)


def bioul_to_boul(bioul_tags: List[str]) -> List[str]:
    return ["O" if tag.startswith("I-") else tag for tag in bioul_tags]


def boul_to_bioul(tag_sequence: List[str]) -> List[str]:
    bioul_tags = []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label is None:
            index += 1
            continue
        elif label[0] == "B":
            bioul_tags.append(label)
            tag_type = label[2:]
            index += 1
            label = tag_sequence[index]
            while label[0] != "L" and index < len(tag_sequence):
                bioul_tags.append(f"I-{tag_type}")
                index += 1
                label = tag_sequence[index]
            bioul_tags.append(label)  # append L
        else:
            bioul_tags.append(label)  # append O
        index += 1
    return bioul_tags


def boul_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: List[str] = None,
):
    bioul_tags = boul_to_bioul(tag_sequence=tag_sequence)
    return bioul_tags_to_spans(
        tag_sequence=bioul_tags,
        classes_to_ignore=classes_to_ignore,
    )

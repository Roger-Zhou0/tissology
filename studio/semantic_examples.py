"""Task-level review budget validation, independent of frozen v1 scoring."""


def validate_examples(store, positive_ids, negative_ids):
    for identifiers in (positive_ids, negative_ids):
        if (not isinstance(identifiers, list) or any(not isinstance(x, str) for x in identifiers)
                or len(identifiers) > 40 or len(set(identifiers)) != len(identifiers)):
            raise ValueError("Choose up to 40 distinct library examples in total")
        for identifier in identifiers:
            store.row(identifier)
    if set(positive_ids) & set(negative_ids):
        raise ValueError("An example cannot both match and not match")
    if len(positive_ids) + len(negative_ids) > 40:
        raise ValueError("Choose up to 40 distinct library examples in total")
    return positive_ids, negative_ids

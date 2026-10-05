def calculate_recall(
    retrieved_pages: list[int],
    relevant_pages: list[int],
) -> float:

    if not relevant_pages:
        return 0.0

    retrieved = set(retrieved_pages)
    relevant = set(relevant_pages)

    intersection = retrieved.intersection(
        relevant
    )

    return len(intersection) / len(relevant)



def calculate_mrr(
    retrieved_pages: list[int],
    relevant_pages: list[int],
) -> float:

    relevant = set(relevant_pages)

    for index, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page in relevant:
            return 1 / index

    return 0.0
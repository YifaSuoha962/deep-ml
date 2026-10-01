def wl_refine(adj, colors):
    """
    Perform one round of 1-WL colour refinement.

    Args:
        adj: adjacency list, adj[i] is the neighbours of node i
        colors: list of integer colours, one per node

    Returns:
        New list of integer colours.
    """

    signatures = []

    for i in range(len(adj)):
        neighbour_colors = tuple(
            sorted(colors[j] for j in adj[i])
        )

        signature = (
            colors[i],
            neighbour_colors
        )

        signatures.append(signature)

    # Sort distinct signatures and assign deterministic integer ids
    unique_signatures = sorted(set(signatures))

    signature_to_color = {
        sig: idx
        for idx, sig in enumerate(unique_signatures)
    }

    new_colors = [
        signature_to_color[sig]
        for sig in signatures
    ]

    return new_colors


def wl_iterate(adj, colors=None):
    """
    Repeatedly apply WL refinement until the number of distinct
    colours stops increasing.

    Returns:
        (colors, rounds)

        colors:
            colours from the final refinement round

        rounds:
            number of rounds that increased the number of
            distinct colours
    """

    n = len(adj)

    if colors is None:
        colors = [0] * n
    else:
        colors = list(colors)

    rounds = 0

    while True:
        old_num_colors = len(set(colors))

        new_colors = wl_refine(adj, colors)

        new_num_colors = len(set(new_colors))

        # The problem asks us to return the result of the final round,
        # even when that round does not increase the number of colours.
        colors = new_colors

        if new_num_colors == old_num_colors:
            break

        rounds += 1

    return colors, rounds


def color_histogram(colors):
    """
    Return the sorted sizes of colour classes.

    Example:
        colors = [0, 1, 0, 2, 1]
        -> [1, 2, 2]
    """

    counts = {}

    for color in colors:
        counts[color] = counts.get(color, 0) + 1

    return sorted(counts.values())


def disjoint_union(adj1, adj2):
    """
    Build the disjoint union of two graphs.

    Nodes of adj2 are offset by len(adj1).
    """

    n1 = len(adj1)

    result = []

    # First graph keeps its original indices
    for neighbours in adj1:
        result.append(list(neighbours))

    # Second graph gets index offset
    for neighbours in adj2:
        result.append([
            j + n1
            for j in neighbours
        ])

    return result


def wl_test(adj1, adj2):
    """
    Run 1-WL jointly on the disjoint union.

    Returns:
        True  -> 1-WL cannot distinguish the two graphs
        False -> 1-WL distinguishes them
    """

    n1 = len(adj1)
    n2 = len(adj2)

    # Different numbers of nodes => immediately distinguishable
    if n1 != n2:
        return False

    union_adj = disjoint_union(adj1, adj2)

    # Start all nodes with the same colour
    initial_colors = [0] * (n1 + n2)

    final_colors, _ = wl_iterate(
        union_adj,
        initial_colors
    )

    # Compare colour multisets of the two halves
    colors1 = sorted(final_colors[:n1])
    colors2 = sorted(final_colors[n1:])

    return colors1 == colors2
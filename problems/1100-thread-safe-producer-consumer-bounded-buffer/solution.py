def run_producer_consumer(capacity, producer_items, num_consumers):
    # Coordinate producers and consumers over a shared bounded buffer.
    # Return the sorted list of all consumed items.

    if capacity < 1:
        raise ValueError("capacity must be >= 1")

    if num_consumers < 1:
        raise ValueError("num_consumers must be >= 1")

    buffer = []
    consumed = []

    # 每个 producer 当前生产到了哪个位置
    positions = [0] * len(producer_items)

    # 总共需要生产 / 消费的元素数量
    total_items = 0
    for items in producer_items:
        total_items += len(items)

    # 所有元素都消费完后结束
    while len(consumed) < total_items:

        # Producers:
        # 每个生产者每轮尝试放入一个元素
        for i in range(len(producer_items)):

            # 当前 producer 已经生产完
            if positions[i] >= len(producer_items[i]):
                continue

            # bounded buffer 已满
            if len(buffer) >= capacity:
                break

            buffer.append(
                producer_items[i][positions[i]]
            )

            positions[i] += 1

        # Consumers:
        # 每个 consumer 每轮最多消费一个元素
        for _ in range(num_consumers):

            if not buffer:
                break

            item = buffer.pop(0)
            consumed.append(item)

    return sorted(consumed)
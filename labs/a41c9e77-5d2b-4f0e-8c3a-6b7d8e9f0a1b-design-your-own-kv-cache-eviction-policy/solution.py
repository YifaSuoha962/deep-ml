# --- cell 1 ---
import math


class EvictionPolicy:
    def __init__(self, capacity_blocks):
        self.capacity = max(0, int(capacity_blocks))

        # 驻留块 -> [频率计数, 最近访问的请求编号, 深度]
        self.resident = {}

        # 尚未准入块的缺失次数，用于判断深层块是否重复出现。
        # 值为 [缺失次数, 最近一次缺失的序号]。
        self.seen = {}
        self.admit_calls = 0

        # 频率每经过一个 epoch 衰减一次。
        self.epoch_size = 256
        self.current_epoch = None
        self.clock = 0

    def admit(self, block, depth, prompt_blocks):
        # 没有容量时不准入，避免后续要求从空缓存中选 victim。
        if self.capacity <= 0:
            return False

        # 记录该块遇到缺失的次数。计数封顶，避免计数无限增长。
        self.admit_calls += 1
        count, _ = self.seen.get(block, (0, 0))
        count = min(count + 1, 8)
        self.seen[block] = (count, self.admit_calls)

        # 定期清理很久以前只出现过一次的块，限制 ghost 元数据大小。
        prune_interval = max(128, self.capacity * 4)
        if self.admit_calls % prune_interval == 0:
            keep_after = self.admit_calls - max(256, self.capacity * 8)
            self.seen = {
                key: value
                for key, value in self.seen.items()
                if value[1] >= keep_after
            }

        # 浅层通常属于共享前缀，第一次见到也值得缓存。
        # 深层块要再次出现才准入，过滤大部分一次性文档和 prompt。
        if depth < 4:
            return True
        return count >= 2

    def _decay_frequencies(self, t):
        """按请求编号推进 epoch，并衰减缓存内块的频率。"""
        epoch = t // self.epoch_size

        if self.current_epoch is None:
            self.current_epoch = epoch
            return

        elapsed = epoch - self.current_epoch
        if elapsed <= 0:
            return

        # 多个 epoch 一次跳过时，避免执行大量重复移位。
        shift = min(elapsed, 8)
        for meta in self.resident.values():
            # 频率至少保留为 1；近期是否被访问由 last_t 另行反映。
            meta[0] = max(1, meta[0] >> shift)

        self.current_epoch = epoch

    def touch(self, block, depth, prompt_blocks, t, hit):
        self._decay_frequencies(t)
        self.clock = t

        if hit and block in self.resident:
            # 命中提高频率并更新最近访问时间。
            meta = self.resident[block]
            meta[0] = min(meta[0] + 1, 255)
            meta[1] = t
            return

        # miss 时，cache 已先插入该块再调用 touch。
        # 使用此前记录的缺失次数初始化频率。
        seen_count, _ = self.seen.get(block, (1, 0))
        self.resident[block] = [
            max(1, seen_count),
            t,
            depth,
        ]

    def victim(self):
        # 正常情况下，cache 只在已满时调用 victim，因此 resident 不为空。
        if not self.resident:
            return None

        def utility(item):
            block, (frequency, last_t, depth) = item
            age = max(0, self.clock - last_t)

            # 高频块更值得保留；较浅的块有轻微优先级；
            # 长时间未访问的块会因 age 惩罚更容易被淘汰。
            score = (
                4.0 * math.log2(frequency + 1)
                + 2.0 / (depth + 1)
                - 0.02 * age
            )
            return score, last_t

        # 淘汰效用最低的驻留块；同分时淘汰更久未访问的块。
        victim_block = min(self.resident.items(), key=utility)[0]
        del self.resident[victim_block]
        return victim_block
def process_operations(operations):
    output_list = []

    # 队列：用两个 list 模拟（in_stack 负责入队，out_stack 负责出队）
    # 模拟 dequeue() -- from collections import dequeue
    in_stack = []
    # out_stack = []

    # 最小栈：一个 list 存元素，另一个 list 同步存当前最小值
    stack = []
    min_stack = []

    for op in operations:
        tag = op[0]

        if tag == "enqueue":
            # queue.append(v)
            v = op[1]
            in_stack.append(v)    

        elif tag == "dequeue":
            # 如果 out_stack 为空，将 in_stack 全部倒入 out_stack（顺序反转）-- queue.popleft()
            output_list.append(in_stack.pop(0))
            # if not out_stack:
            #     while in_stack:
            #         out_stack.append(in_stack.pop())  
            # output_list.append(out_stack.pop())

        elif tag == "mpush":
            v = op[1]
            stack.append(v)
            # 维护最小栈：如果新元素比当前最小值小或相等，则压入新元素；否则重复压入当前最小值
            if not min_stack or v <= min_stack[-1]:
                min_stack.append(v)
            else:
                min_stack.append(min_stack[-1])

        elif tag == "mpop":
            output_list.append(stack.pop())
            min_stack.pop()

        elif tag == "mtop":
            output_list.append(stack[-1])

        elif tag == "mmin":
            output_list.append(min_stack[-1])

    return output_list
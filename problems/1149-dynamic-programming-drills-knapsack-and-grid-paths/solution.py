def dp_drills(weights, values, capacity, m, n, s1, s2):
    # Return a dict with keys: 'knapsack', 'grid_paths', 'lcs', 'complexity'
    n_items = len(weights)
    
    # 1. 0/1 Knapsack
    knapsack_dp = [[0] * (capacity + 1) for _ in range(n_items + 1)] 
    for i in range(1, n_items + 1):
        item_w = weights[i-1]
        item_v = values[i-1]
        for w in range(capacity + 1):
            # don't choose it 
            knapsack_dp[i][w] = knapsack_dp[i-1][w]
            # choose it when satisfying the constraint
            if item_w <= w:
                knapsack_dp[i][w] = max(knapsack_dp[i-1][w], knapsack_dp[i-1][w-item_w] + item_v)
    knapsack_res = knapsack_dp[n_items][capacity]

    # 2. unique gridd path
    grid_dp = [[1] * (n + 1) for _ in range(m + 1)]
    # for j in range(n + 1):
    #     grid_dp[0][j] = 1
    # for i in range(m + 1):
    #     grid_dp[i][0] = 1 
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            grid_dp[i][j] = grid_dp[i-1][j] + grid_dp[i][j-1]
    path_res = grid_dp[m-1][n-1]

    # 3. Longest Common Subsequence
    #
    # lcs_dp[i][j]:
    # LCS length between
    # s1[:i] and s2[:j]
    # ============================================================
    len1 = len(s1)
    len2 = len(s2)

    lcs_dp = [
        [0] * (len2 + 1)
        for _ in range(len1 + 1)
    ]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            if s1[i - 1] == s2[j - 1]:
                lcs_dp[i][j] = (
                    lcs_dp[i - 1][j - 1] + 1
                )
            else:
                lcs_dp[i][j] = max(
                    lcs_dp[i - 1][j],
                    lcs_dp[i][j - 1]
                )

    lcs_result = lcs_dp[len1][len2]

    # ============================================================
    # 4. Complexity strings
    # Must match the required format exactly
    # ============================================================
    complexity = {
        "knapsack": "O(n*W) time, O(n*W) space",
        "grid_paths": "O(m*n) time, O(m*n) space",
        "lcs": "O(m*n) time, O(m*n) space"
    }

    return {
        "knapsack": knapsack_res,
        "grid_paths": path_res,
        "lcs": lcs_result,
        "complexity": complexity
    }



    

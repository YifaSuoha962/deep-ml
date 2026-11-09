import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        if smoothing <= 0:
            raise ValueError("smoothing must be greater than 0")

        self.smoothing = float(smoothing)
        self.log_class_priors_ = None
        self.feature_probs_ = None
        self.n_features_in_ = None

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("training data must not be empty")
        if not np.all((X == 0) | (X == 1)):
            raise ValueError("X must contain only binary values 0 and 1")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only class labels 0 and 1")

        # 使用浮点数计算概率。
        X = X.astype(float)
        y = y.astype(int)

        self.n_features_in_ = X.shape[1]
        self.log_class_priors_ = np.empty(2, dtype=float)
        self.feature_probs_ = np.empty((2, self.n_features_in_), dtype=float)

        for class_label in (0, 1):
            class_mask = (y == class_label)
            class_count = np.sum(class_mask)
            if class_count == 0:
                # 未在训练数据中出现的类别，其先验概率为 0。
                self.log_class_priors_[class_label] = -np.inf

                # 该类别不会被预测到；仍以平滑值填充，避免无效概率。
                self.feature_probs_[class_label] = 0.5
                continue
            # 类别先验 P(C)
            self.log_class_priors_[class_label] = np.log(class_count / y.size)
            # 似然 P(x|C)
            ones_per_feature = X[class_mask].sum(axis=0)
            self.feature_probs_[class_label] = (ones_per_feature + self.smoothing) / (class_count + 2 * self.smoothing)
        
        return self

    def predict(self, X):
        # Predict class labels for test set X
        if self.log_class_priors_ is None:
            raise RuntimeError("model has not been trained; call forward first")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as training data")
        if not np.all((X == 0) | (X == 1)):
            raise ValueError("X must contain only binary values 0 and 1")

        X = X.astype(float)

        # 每个类别、每个特征取值为 1 的概率，以及取值为 0 的概率。
        p1 = self.feature_probs_
        p0 = 1.0 - p1

        # 计算 log P(X | C) + log P(C)。
        log_likelihoods = (
            X[:, None, :] * np.log(p1)[None, :, :]
            + (1.0 - X[:, None, :]) * np.log(p0)[None, :, :]
        ).sum(axis=2)

        log_scores = log_likelihoods + self.log_class_priors_[None, :]

        # argmax 返回分数最高的类别；类别分数相同则返回 0。
        return np.argmax(log_scores, axis=1)



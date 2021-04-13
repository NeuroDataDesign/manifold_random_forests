
class TransformationMixin:
    def apply_transform(self, X, proj_mat):
        return X @ proj_mat
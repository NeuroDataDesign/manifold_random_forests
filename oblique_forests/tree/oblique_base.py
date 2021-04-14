class BaseObliqueSplitter:
    """Abstract base class for oblique splitters."""

    def sample_proj_mat(self, sample_inds):
        raise NotImplementedError("All oblique splitters must implement this function.")

    def project_data(self, sample_inds):
        raise NotImplementedError("")

    def split(self, sample_inds):
        raise NotImplementedError("")


class Whole:

    def __init__(self, number_of_traces):

        if number_of_traces == 1:
            raise ValueError

        self._traces = self._collect_traces()
        self.p_mean = sum([trace.p for trace in self._traces])/number_of_traces
        self.k_mean = sum([trace.k for trace in self._traces]) / number_of_traces

    def _collect_traces(self):
        raise NotImplemented()

    def _compute_all(self):
        for trace in range(len(self._traces)):
            self._traces[trace].calculate_work()
            self._traces[trace].rupture_forces()

    def plot_all(self):
        for trace in range(len(self._traces)):
            self._traces[trace].plot()

    def save_all(self):
        for trace in range(len(self._traces)):
            self._traces[trace].results_to_latex(self._traces[trace].histo_data)
            self._traces[trace].results_to_txt()

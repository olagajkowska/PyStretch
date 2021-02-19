from Tools import *
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path


class Trajectory:
    """

    Creates an object representing a single protein stretching trajectory (theoretical or experimental).

    Attributes
    ----------

        self._path : str
        self._case : int
        self._data : DataFrame
        self._data_inverse : DataFrame
        self._smooth_data : DataFrame
        self._histo_data : DataFrame
        self.bounds : tuple
        self.initial_guess : list
        self.residues : int
        self.bound_length : float
        self.p : float
        self.k : float

    Examples
    --------
    >>> trace = Trajectory("data.xlsx", 2, 0.38, 240, [0.7, 0.005], ((0.3, 0.8), (0.0009, 0.009)), p=0.6, k=0.005)
    >>> trace.histo_data

        heights means widths
        0.021 7.547 1.15
        0.013 31.555 5.017
        0.055 46.961 0.937
        0.031 77.696 1.06
        0.289 90.739 0.692


    """

    def __init__(self, filename: str, case: int, bond_length: float, residues: int, initial_guess: list,
                 bounds: tuple, **kwargs) -> object:
        """

        Parameters
        ---------

        filename : str
            Name of file with input data.
        case : int
            Case number.
        bond_length : float
            Bond length.
        residues: int
            Number of residues.
        initial guess : ndarray
            List with initial params (in order - p, then k)
        bounds : tuple
            Tuple of tuples with parameters bounds.
        **kwargs
            Arbitrary keyword arguments.

        Keyword arguments
        -----------------
        p : float
        k : float

        """

        self._path = os.path.join(os.path.dirname(__file__), "data_exp/", filename)
        self._case = case

        self.bond_length = bond_length
        self.residues = residues
        self.initial_guess = initial_guess
        self.bounds = bounds

        if self._path.endswith(".afm"):
            self._data = copy.deepcopy(load_data(self._path))
            file = Path(os.path.join(os.path.dirname(__file__), "data/", filename[:4] + "_inverse.afm"))

            if file.is_file():
                self._data_inverse = copy.deepcopy(load_data(file, inverse=True))
        else:
            self._data = copy.deepcopy(read_excel(self._path, None, [5 * (case - 1), 5 * case - 4]))
            self._data_inverse = copy.deepcopy(read_excel(self._path, None, [5 * (case - 1) + 2, 5 * case - 2]))

        self._smooth_data = copy.deepcopy(smooth_data(self._data))
        self._histo_data = self._find_peaks()

        if "p" in kwargs:
            self.p = kwargs["p"]
        if "k" in kwargs:
            self.k = kwargs["k"]

    @property
    def histo_data(self):
        return self._histo_data

    def _find_peaks(self):

        """

        Fits p and k parameters, then computes contour lengths for each (F, d) pair and saves them as a new
        column in _data field. Saves found parameters at p_k_table/p_k_results.txt. If p and k of the particular
        trace are given or saved in the file before, skips that step. Finds properties of histogram peaks (heights,
        means, widths, number of peaks).

        Returns
        -------
        Peaks properties : DataFrame
            DataFrame with three columns: heights, means, widths. DataFrame is then saved as _histo_data field.

        """

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, "p_k_table/")

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        with open(results_dir + "p_k_results.txt") as myfile:

            if ("trace" + str(self._case)) in myfile.read():
                csv_data = pd.read_table(results_dir + "p_k_results.txt", delim_whitespace=True, header=0)
                self.p = csv_data.loc[csv_data["trace"] == "trace" + str(self._case)]["p"].values[0]
                self.k = csv_data.loc[csv_data["trace"] == "trace" + str(self._case)]["k"].values[0]
            else:
                myfile.close()
                if not (hasattr(self, "p") or hasattr(self, "k")):
                    fitted = self._fit(self._find_last_range())
                    self.p = fitted[0]
                    self.k = fitted[1]
                    self.parameters_to_txt()

        L = find_contour_lengths(self._data, self.p, self.k)
        self._data["L"] = L
        histo_data = decompose_histogram(self._data["L"])

        return histo_data

    def plot_contour_length_histo(self, position):

        """
        Pre-plotting contour length histo.

        Parameters
        ----------
        position : .axes.Axes
            Chosen axis.

        """

        l_space = np.linspace(0, self._data['L'].max() + 20, 1001)
        position.hist(self._data['L'], bins=4 * int(100), range=[0, 100], density=True, alpha=0.5)
        for index, row in self._histo_data[['means', 'widths', 'heights']].iterrows():
            mean, width, height = tuple(row.to_numpy())
            residues = 1 + int(mean / self.bond_length)
            label = "L= " + str(round(mean, 3)) + ' (' + str(residues) + ' AA)'
            y_gauss = single_gaussian(l_space, height, mean, width)
            position.plot(l_space, y_gauss, linestyle='--', linewidth=0.5, label=label,
                          color=get_color(index))

        position.set_title('Contour length histogram')
        position.set_xlabel('Extension [nm]')
        position.set_ylabel('Counts')
        position.legend(fontsize='small')
        position.set_xlim(0, 100)

    def plot_fd(self, position):

        """
        Pre-plotting F(d).

        Parameters
        ----------
        position : .axes.Axes
            Chosen axis.

        """
        raise NotImplemented("To be implemented in another class")

    def plot(self):

        """

        Creates two subplots: contour length histo and F(d). Saves figure in catalogue '/Images'. If the
        catalogue doesn't exist, creates one. Name of the image is the same as name of input data.

        """

        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 12}
        matplotlib.rc('font', **font)

        fig, ax = plt.subplots(1, 2, dpi=200, figsize=(10, 5))
        self.plot_contour_length_histo(position=ax[0])
        self.plot_fd(position=ax[1])

        plt.tight_layout()

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Images/')
        sample_file_name = 'trace' + str(self._case)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(results_dir + sample_file_name)
        print('saved as', os.path.join(results_dir, sample_file_name))
        plt.close()

    def _to_minimize(self, x, last_range):
        """
        Creating a function to be minimized.

        """
        raise NotImplemented()

    def _fit(self, last_range):
        """
        Fitting p and k parameters to the data.

        """
        raise NotImplemented()

    def _find_last_range(self):
        """
        Finding the range to fit parameters.

        """
        raise NotImplemented()

    def state_boundaries(self):
        """

        Finding boundaries of each state. A boundary is defined when WLC fit for each state reaches the value 12.
        Method changes histo_data field - it adds two new columns: 'begs' and 'ends'. Returns None.

        """
        raise NotImplemented()

    def calculate_work(self):

        """

        Calculating work done in each state by applying the definition: W = <F> * dx. This method changes the
        _histo_data field by adding one more column: 'work'.


        """

        self._histo_data['work'] = work(self._data, self._histo_data['begs'], self._histo_data['ends'])
        self._histo_data['work-s'] = simpson(self._smooth_data, self._histo_data['begs'], self._histo_data['ends'])

        if hasattr(self, "_data_inverse"):
            self._histo_data['work_i'] = work(self._data_inverse, self._histo_data['begs'], self._histo_data['ends'])

    def rupture_forces(self):

        """

        Calculating rupture forces that lead the protein to the next state. This method changes the
        _histo_data field by adding one more column: 'rupture'.


        """

        rupture_list = []
        for ind, row in self._histo_data.iterrows():
            rupture_list.append(self._smooth_data.loc[(row['begs'] < self._smooth_data['d']) &
                                                      (self._smooth_data['d'] < row['ends'])]['F'].max())

        self._histo_data['rupture'] = rupture_list

    def results_to_latex(self, data):

        """

        Saves trace data as .txt file in the form of latex table. Filename is "trace" + case number.

        Parameters
        ----------
        data : DataFrame
            Input DataFrame.

        Returns
        -------
        Peaks properties : DataFrame
            DataFrame with three columns: heights, means, widths. DataFrame is then saved as _histo_data field.

        """

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, "Trajectory_tables/")
        sample_file_name = "trace" + str(self._case)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        text_file = open(results_dir + sample_file_name + ".txt", "w")
        text_file.write(data.to_latex(index=False))
        text_file.close()

    def results_to_txt(self):

        """
        Saving contour length histo data to txt files in catalogue Trajectory_tables_txt. If catalogue doesn't
        exist, creates one.

        """

        self.state_boundaries()

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, "Trajectory_tables_txt/")
        sample_file_name = "trace" + str(self._case)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        self.histo_data.to_csv(results_dir + sample_file_name + ".txt", header=True,
                               index=False,
                               sep=" ", mode="w")

    def parameters_to_txt(self):

        """
        Method saving parameters to txt file. Saves table in catalogue 'p_k_table'.
        """

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, "p_k_table/")
        sample_file_name = "p_k_results"
        p_k_df = pd.DataFrame({"trace": ["trace" + str(self._case)], "p": [self.p], "k": [self.k]})
        p_k_df.set_index("trace")

        p_k_df.to_csv(results_dir + sample_file_name + ".txt", header=os.stat(results_dir + sample_file_name +
                                                                              ".txt").st_size == 0, index=False,
                      sep=" ", mode="a")

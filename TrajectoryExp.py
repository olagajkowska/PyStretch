from Tools import *
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path


class TrajectoryExp:
    bond_length = 0.365
    residues = 240 # -14
    initial_guess_exp_d = {'p_prot': 5.88, 'p_dna': 0.13, 'k_prot': 0, 'k_dna': 0.005, 'l_dna': 335}
    initial_guess_exp = [0.55, 0]
    bounds = ((0.1, 3), (0, 0))

    # __slots__ = ['__data', '__smooth_data', 'p', 'k', 'histo_data', '__path', '__case']

    def __init__(self, filename, case, inverse=False, **kwargs):

        self.__inverse = inverse
        self.__path = os.path.join(os.path.dirname(__file__), 'data_exp/', filename)

        if filename == 'total.txt':
            self.__data = copy.deepcopy(load_data(self.__path))
        else:
            self.__data = copy.deepcopy(read_excel(self.__path, None, [5 * (case - 1), 5 * case - 4]))

        if self.__inverse:
            self.__data_inverse = copy.deepcopy(read_excel(self.__path, None, [5 * (case - 1) + 2, 5 * case - 2]))

        self.__case = case

        self.__smooth_data = copy.deepcopy(smooth_data(self.__data))

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'p_k_table_exp/')

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        name = 'p_k_results_exp.txt'
        name_tables = 'Trajectory_tables_exp_txt/'

        with open(results_dir + name) as myfile:

            if 'trace' + str(self.__case) in myfile.read():
                csv_data = pd.read_table(results_dir + name, delim_whitespace=True, header=0)
                self.p = csv_data.loc[csv_data['trace'] == str(self.__case)]['p'].values[0]
                self.k = csv_data.loc[csv_data['trace'] == str(self.__case)]['k'].values[0]
            else:
                myfile.close()
                if 'p' in kwargs:
                    self.p = kwargs['p']
                if 'k' in kwargs:
                    self.k = kwargs['k']
                else:
                    fitted = fit(self.__data, find_last_range(self.__data, self.__smooth_data), self.bond_length,
                                 self.residues, self.initial_guess_exp, self.bounds)
                    # print(fitted)
                    self.p = fitted[0]
                    self.k = fitted[1]
                    self.parameters_to_txt()

        L = find_contour_lengths(self.__data, self.p, self.k)
        self.__data['L'] = L

        self.histo_data = decompose_histogram(self.__data['L'], significance=0.022)
        self.state_boundaries()

        if self.__path != os.path.join(os.path.dirname(__file__), 'data_exp/', 'total.txt'):
            my_file = Path(os.path.join(os.path.dirname(__file__), name_tables, 'trace' + str(
                self.__case)))
            if my_file.is_file():
                self.histo_data = pd.read_csv(os.path.join(os.path.dirname(__file__), name_tables,
                                                           'trace' + str(self.__case)), sep=" ", header=0)
            else:
                self.histo_data = decompose_histogram(self.__data['L'])
                self.results_to_txt()
                self.results_to_latex(self.histo_data)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, column):
        self.__data[str(column)] = column

    @property
    def smooth_data(self):
        return self.__smooth_data

    @smooth_data.setter
    def smooth_data(self, column):
        self.__smooth_data[str(column)] = column

    @property
    def path(self):
        return self.__path

    def plot_contour_length_histo(self, position):

        """Pre-plotting contour length histo.

        :param position: position of the plot
        :type position: numpy.ndarray
        :return: None

        """

        l_space = np.linspace(0, self.__data['L'].max() + 20, 1001)
        position.hist(self.__data['L'], bins=4 * int(100), range=[0, 100], density=True, alpha=0.5)
        for index, row in self.histo_data[['means', 'widths', 'heights']].iterrows():
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

        """Pre-plotting F(d).

                :param position: position of the plot
                :type position: numpy.ndarray
                :return: None

                """
        # if self.__path != os.path.join(os.path.dirname(__file__), 'data/', 'total.txt'):
        position.plot(self.__data.sort_values(by='d')['d'], self.__data.sort_values(by='d')['F'])
        if self.__inverse:
            position.plot(self.__data_inverse['d'], self.__data_inverse['F'], color=mcolors.CSS4_COLORS[
                'lightsteelblue'])
        position.plot(self.__smooth_data['d'], self.__smooth_data['F'], color=mcolors.CSS4_COLORS['pink'])
        index = 0
        for mean in self.histo_data['means']:
            residues = 1 + int(mean / self.bond_length)
            d_space = np.linspace(1, self.__data['d'].max())

            label = "L= " + str(round(mean, 3)) + ' (' + str(residues) + ' AA)'
            y_fit = wlc(d_space, mean, self.p, self.k)

            position.plot(d_space, y_fit, ls='--', linewidth=1, label=label, color=get_color(index))
            index += 1

        position.set_ylim(0, 45)
        position.set_xlim(min(self.__data['d']), max(self.__data['d']))
        position.set_title('Trace fits')
        position.set_xlabel('Extension [nm]')
        position.set_ylabel('Force [pN]')
        position.legend(fontsize='small')

    def plot(self):

        """ Creates two subplots: contour length histo and F(d). Saves figure in catalogue '/Images'. If the
        catalogue doesn't exist, creates one. Name of the image is the same as name of input data (contains its last
        4 characters).

        """

        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 12}
        matplotlib.rc('font', **font)

        fig, ax = plt.subplots(1, 2, dpi=200, figsize=(10, 5))
        self.plot_contour_length_histo(position=ax[0])
        self.plot_fd(position=ax[1])

        for i in range(len(self.histo_data)):
            ax[1].axvline(x=self.histo_data['begs'].iloc[i], color='gray')
            if i == len(self.histo_data) - 1:
                ax[1].axvline(x=self.histo_data['ends'].iloc[i], color='gray')
                break

        plt.tight_layout()

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Images_exp/')

        sample_file_name = 'trace' + str(self.__case)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(results_dir + sample_file_name)
        plt.close()

    def state_boundaries(self):

        """Finding boundaries of each state. A boundary is defined when WLC fit for each state reaches the value 12.
        Method changes histo_data field - it adds two new columns: 'begs' and 'ends'. Returns None.

        """

        begs = [round(self.__smooth_data['d'].min(), 3)]
        ends = []
        for mean in self.histo_data['means']:
            cut = invert_wlc(25, self.p, self.k) * mean
            bound = self.__smooth_data.iloc[(self.__smooth_data['d'] - cut).abs().argsort()[:1]]['d'].to_list()[0]
            data_near = self.__smooth_data.loc[self.__smooth_data['F']
                                               == self.__smooth_data.loc[abs(self.__smooth_data['d'] - bound)
                                                                         < 1]['F'].max()]['d'].min()
            # # dealing with 72 boundary
            # if float(70) < mean < float(74):
            #
            #     try:
            #         maximas = argrelextrema(
            #             self.__smooth_data.loc[abs(self.__smooth_data['d'] - bound) < 5]['F'].to_numpy(),
            #             np.greater)
            #         data_near = self.__smooth_data.loc[abs(self.__smooth_data['d'] - bound) < 5]['d'].to_numpy()[
            #             maximas[0].min()]
            #     except:
            #         data_near = self.__smooth_data.loc[self.__smooth_data['F']
            #                                            == self.__smooth_data.loc[abs(self.__smooth_data['d'] - bound)
            #                                                                      < 2]['F'].max()]['d'].min()
            #
            # dealing with the wide one
            if float(43) < mean < float(45):
                data_near = self.__smooth_data.loc[self.__smooth_data['F']
                                                   == self.__smooth_data.loc[abs(self.__smooth_data['d'] - bound)
                                                                             < 3]['F'].max()]['d'].min()

            data_near = round(data_near, 3)
            if mean == self.histo_data['means'].iloc[-1]:
                ends.append(self.__smooth_data['d'].iloc[-1])
                break
            begs.append(data_near)
            ends.append(data_near)

        # boundaries = [[begs[i], ends[i]] for i in range(len(begs))]
        self.histo_data['begs'] = begs
        self.histo_data['ends'] = ends

    def calculate_work(self):

        """Calculating work done in each state by applying the definition: W = <F> * dx. This method changes the
        histo-data field by adding one more column: 'work'.


        """

        self.histo_data['work'] = work(self.__data, self.histo_data['begs'], self.histo_data['ends'])
        self.histo_data['work-s'] = simpson(self.__smooth_data, self.histo_data['begs'], self.histo_data['ends'])

    def rupture_forces(self):

        rupture_list = []
        for ind, row in self.histo_data.iterrows():
            rupture_list.append(self.__smooth_data.loc[(row['begs'] < self.__smooth_data['d']) &
                                                       (self.__smooth_data['d'] < row['ends'])]['F'].max())

        self.histo_data['rupture'] = rupture_list

    def results_to_latex(self, data):

        """Method exporting DataFrame to latex. Saves latex table in catalogue 'Trajectory_tables'.

        :param data: Data to export into latex table.
        :type data: DataFrame

        """

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Trajectory_tables_exp/')
        sample_file_name = 'trace' + str(self.__case)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        text_file = open(results_dir + sample_file_name + ".txt", "w")
        text_file.write(data.to_latex(index=False))
        text_file.close()

    def results_to_txt(self):

        """ Saving contour length histo data to txt files in catalogue Trajectory_tables_txt. If catalogue doesn't
        exist, creates one.

        :return: None
        """

        self.state_boundaries()
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Trajectory_tables_exp_txt/')

        sample_file_name = 'trace' + str(self.__case)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        self.histo_data.to_csv(results_dir + sample_file_name + ".txt", header=True,
                               index=False,
                               sep=' ', mode='w')

    def parameters_to_txt(self):

        """Method exporting saving parameters to txt file. Saves table in catalogue 'p_k_table'.

                """

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'p_k_table_exp/')
        sample_file_name = 'p_k_results_exp'

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        p_k_df = pd.DataFrame({'trace': [str(self.__case)], 'p': [self.p], 'k': [self.k]})
        p_k_df.set_index('trace')

        p_k_df.to_csv(results_dir + sample_file_name + ".txt", header=os.stat(results_dir + sample_file_name +
                                                                              ".txt").st_size == 0, index=False,
                      sep=' ', mode='a')


if __name__ == '__main__':

    for i in range(1, 10):
        trace = TrajectoryExp('data_test.xls', i, inverse=True)
        trace.plot()




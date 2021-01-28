from Tools import *
from Trajectory import Trajectory
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import ast
from scipy.stats import norm


class Experiment:
    __slots__ = ['__traces', 'p_mean', 'k_mean', '__data_total', 'total_histo_data', '__total_trace', 'states',
                 'states_rup', 'dhs', '__rupture_table']

    def __init__(self, traces):

        if traces == 1:
            raise ValueError('More than 1 trajectory required')
        self.__traces = []
        self.__rupture_table = pd.DataFrame()
        self.dhs = pd.DataFrame()
        p = 0
        k = 0

        for trace in range(1, traces + 1):
            print('Analyzing file ' + 'aa' + str(int(trace)).zfill(2) + '.afm')

            self.__traces.append(Trajectory('aa' + str(int(trace)).zfill(2) + '.afm'))
            p += self.__traces[trace - 1].p
            k += self.__traces[trace - 1].k
            self.__traces[trace - 1].calculate_work()
            self.__traces[trace - 1].rupture_forces()
            self.__traces[trace - 1].plot()
            self.__traces[trace - 1].results_to_latex(self.__traces[trace - 1].histo_data)

        max_states = len(self.__traces[0].histo_data)
        index_max = 0

        for index, trace in enumerate(self.__traces):
            if len(trace.histo_data) > max_states:
                max_states = len(trace.histo_data)
                index_max = index

        self.__data_total = self.total_data().sort_values(by='d')
        self.p_mean = p / traces
        self.k_mean = k / traces

        self.__data_total.to_csv("data/total.txt", header=True, index=False, sep=' ', mode='w')
        total_trace = Trajectory("total.txt", p=self.p_mean, k=self.k_mean)

        my_file = Path(os.path.join(os.path.dirname(__file__), 'Trajectory_tables_txt/', total_trace.path[-8:-4]))
        if my_file.is_file():
            total_trace.histo_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Trajectory_tables_txt/',
                                                              total_trace.__path[-8:-4]), sep=" ", header=0)
        else:
            total_trace.histo_data = decompose_histogram(total_trace.data['L'])  # , compare=self.__traces[index_max])
            print(self.__traces[index_max].histo_data, index_max)
            total_trace.results_to_txt()
            total_trace.state_boundaries()
        # total_trace.calculate_work()
        self.__total_trace = total_trace
        total_trace.results_to_latex(total_trace.histo_data)
        total_trace.plot()

        # my_file = Path(os.path.join(os.path.dirname(__file__), "Work_histograms/works" + str(len(self.__traces)) +
        #                             ".txt"))
        #
        # if my_file.is_file():
        #     file = open("Work_histograms/works" + str(len(self.__traces)) + ".txt", "r")
        #     contents = file.read()
        #     self.states = ast.literal_eval(contents)
        #     file.close()
        # else:
        # for file in os.listdir(os.path.join(os.path.dirname(__file__), 'Work_histograms/')):
        # if file.startswith("work"):
        #     print(file)
        #     contents = open(os.path.join(os.path.dirname(__file__), 'Work_histograms/', file))
        #     contents_ = contents.read()
        #     first_dict = ast.literal_eval(contents_)
        #     contents.close()
        #     start = int(file[-6:-4])
        #     print(start)
        #     self.states = first_dict.update(self.trajectory_loop(start=start))
        #     new_file = open(file, 'w')
        #     new_file.write(str(self.states))
        #     new_file.close()
        # else:

        w, r = self.trajectory_loop()
        self.states = w
        self.states_rup = r
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Work_histograms/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        results_dir_rup = os.path.join(script_dir, 'Force_histograms/')
        if not os.path.isdir(results_dir_rup):
            os.makedirs(results_dir_rup)
        text_file = open("Work_histograms/works" + str(len(self.__traces)) + ".txt", "w")
        text_file.write(str(self.states))
        text_file.close()
        text_file = open("Force_histograms/forces" + str(len(self.__traces)) + ".txt", "w")
        text_file.write(str(self.states_rup))
        text_file.close()

    @property
    def data_total(self):
        return self.__data_total

    @data_total.setter
    def data_total(self, data):
        self.__data_total = data

    def total_data(self):

        """Concats data from all trajectories into one.


        :return: total data
        :rtype: DataFrame
        """

        data_total = pd.concat([i.data for i in self.__traces], ignore_index=True)
        return data_total

    def trajectory_loop(self, tolerance=3, start=0):

        """Classifies states of one trace to states identified in total histogram.

        :param start: Starting trace
        :type start: int
        :param tolerance: max difference between states boundaries
        :type tolerance: int/float
        :return: dictionary (keys: states, values: works)
        :rtype: dict
        """

        dct1 = {}
        dct2 = {}

        for i in range(len(self.__total_trace.histo_data)):
            dct1['state' + str(i)] = []
            dct2['state' + str(i)] = []

        for trace in self.__traces[start:-1]:
            for index, row in self.__total_trace.histo_data.iterrows():
                if index == len(trace.histo_data):
                    break
                # print(row['ends'], trace.histo_data['ends'][index])
                if abs(row['ends'] - trace.histo_data['ends'][index]) < tolerance:
                    dct1['state' + str(index)].append(trace.histo_data['work'][index])
                    dct2['state' + str(index)].append(trace.histo_data['rupture'][index])
                else:
                    for index2, row2 in self.__total_trace.histo_data[index + 1:].iterrows():
                        if abs(row2['begs'] - trace.histo_data['begs'][index]) < tolerance:
                            if abs(row2['ends'] - trace.histo_data['ends'][index]) < tolerance:
                                dct1['state' + str(index2)].append(trace.histo_data['work'][index])
                                dct2['state' + str(index2)].append(trace.histo_data['rupture'][index])
                                break
                    # dct['state' + str(index)].append(row['work'])

        return dct1, dct2

    @staticmethod
    def plot_histogram(values, label, bins, range):

        """Static method. Plotting the histogram and fitting the curve.

        :param values: List to plot
        :type values: list
        :return: None
        """

        w_space = np.linspace(0, max(values) + 20, 1000)
        plt.figure(figsize=(5, 5))
        # Fit a normal distribution to the data:
        mu, std = norm.fit(values)
        # Plot the histogram.
        plt.hist(values, bins=bins, density=True, alpha=0.3, color='r', range=range)
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, '--r', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)

        plt.xlabel(label)
        plt.ylabel('Counts')
        plt.legend(fontsize='small')

    def plot_work_histograms(self):

        """ Plotting works histograms and saving them to Images/Work_histograms/

        :return: None
        """

        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 12}
        matplotlib.rc('font', **font)

        script_dir = os.path.dirname(__file__)
        results_dir_work = os.path.join(script_dir, 'Work_histograms/')

        for state in self.states:
            self.plot_histogram(values=self.states[state], label='Work', bins=70, range=[0, 180])
            plt.tight_layout()
            sample_file_name = state
            plt.savefig(results_dir_work + sample_file_name)
            plt.close()

        results_dir_rup = os.path.join(script_dir, 'Force_histograms/')
        plt.figure(figsize=(5, 5))
        i = 0
        param_list = []
        for state in self.states_rup:
            # params = decompose_histogram(self.states_rup[state])
            # Fit a normal distribution to the data:
            mu, std = norm.fit(self.states_rup[state])
            # Plot the histogram.
            # Plot the PDF.
            f_space = np.linspace(0, 20, 1000)
            p = norm.pdf(f_space, mu, std)
            plt.plot(f_space, p, ls='--', linewidth=0.5, color=get_color(i))
            plt.hist(self.states_rup[state], bins=130, density=True, alpha=0.3, color=get_color(i), range=[0, 20])
            params = pd.DataFrame({'heights': [max(p)], 'means': [mu], 'widths': [std]})
            # y_plot = [single_gaussian(f_s, params['heights'], params['means'], params['widths']) for f_s in f_space]
            # plt.plot(f_space, y_plot, linestyle='--', linewidth=0.5,
            #          color=get_color(i))
            i += 1
            # self.plot_histogram(values=self.states_rup[state], label='Force', bins=30, range=[0, 20])
            param_list.append(params)

        self.__rupture_table = pd.concat([par for par in param_list], ignore_index=True)
        print(self.__rupture_table)
        plt.xlabel("Rupture force")
        plt.ylabel("Counts")
        plt.title("Rupture forces histogram")
        plt.tight_layout()
        sample_file_name = 'Force_histo'
        plt.savefig(results_dir_rup + sample_file_name)
        plt.close()

    def dudko_hummer_szabo(self):

        fig = plt.figure(figsize=(5, 5), dpi=100)
        results = {'x': [], 't0': [], 'g': []}
        for ind, row in self.__rupture_table[['heights', 'means', 'widths']][:-1].iterrows():
            f_space = np.linspace(min(list(self.states_rup.values())[ind]), row['means'], 1000)
            height, mean, width = tuple(row.to_numpy())

            # print(norm.pdf(f_space, mean, width))
            dhs_data = pd.DataFrame({'forces': f_space,
                                     'force_load': loading_force(),
                                     'probability': norm.pdf(f_space, mean, width),
                                     'nominator': integrate_gauss(f_space, mean, width)})

            dhs_data['denominator'] = dhs_data['probability'] * dhs_data['force_load']
            # dhs_data = dhs_data[dhs_data['denominator'] > 0.1]
            dhs_data['lifetime'] = dhs_data['nominator'] / dhs_data['denominator']
            print(dhs_data)
            coefficients = {}
            init_lifetime = dhs_data['lifetime'].head(1).values[0]
            init_x = 1
            # v = 1
            p0 = (init_x, init_lifetime)
            popt, pcov = curve_fit(dhs_feat_bell, dhs_data['forces'], np.log(dhs_data['lifetime']), p0=p0)
            print(popt)
            coefficients['bell'] = {'x': popt[0], 't0': popt[1], 'g': np.NaN}  # 'covariance': pcov}

            # v = 1/2
            p0 = (coefficients['bell']['x'], coefficients['bell']['t0'], coefficients['bell']['x'] * dhs_data[
                'forces'].max())
            try:
                popt, pcov = curve_fit(dhs_feat_linear_cubic, dhs_data['forces'], dhs_data['lifetime'], p0=p0)
                result = {'x': popt[0], 't0': popt[1], 'g': popt[2]}  # , 'covariance': pcov}
                print('cusp', popt)

            except RuntimeError:
                result = None

            try:
                # plt.plot(f_space, dhs_feat_cusp(f_space, coefficients['cusp']['x'], coefficients['cusp']['t0'],
                #                                 coefficients['cusp']['g']), color='black')
                results['x'].append(coefficients['cusp']['x'])
                results['t0'].append(coefficients['cusp']['t0'])
                results['g'].append(coefficients['cusp']['g'])
            except:
                coefficients['cusp'] = result
                results['x'].append(coefficients['bell']['x'])
                results['t0'].append(coefficients['bell']['t0'])
                results['g'].append(coefficients['bell']['g'])
            label = 'F = ' + str(round(row['means'], 3)) + ' pN'

            plt.plot(dhs_data['forces'], np.log(dhs_data['lifetime']), color=get_color(ind), label=label)
            plt.plot(f_space, dhs_feat_bell(f_space, coefficients['bell']['x'], coefficients['bell']['t0']),
                     color='black', ls='--')
            # plt.plot(f_space, dhs_feat_cusp(f_space, coefficients['bell']['x'], coefficients['bell']['t0'], 5),
            #          color='black', ls='--')
            # plt.axvline(x=mean)

        plt.title('Dudko-Hummer-Szabo lifetimes')
        plt.xlabel('Rupture force')
        plt.ylabel('log(state lifetime)')
        # plt.yscale('log')
        plt.legend(fontsize='small')
        plt.show()
        # results = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in results.items() ]))
        self.dhs = pd.DataFrame.from_dict(results)
        print(self.dhs)


if __name__ == '__main__':
    exp = Experiment(50)
    print(exp.p_mean, exp.k_mean)
    exp.plot_work_histograms()
    exp.dudko_hummer_szabo()

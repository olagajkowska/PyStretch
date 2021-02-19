from Trajectory import Trajectory
from Tools import *


class Experiment(Trajectory):

    def __init__(self, filename, case, **kwargs):

        bond_length = 0.365
        residues = 240  # -14
        initial_guess = [0.55, 0]
        bounds = ((0.1, 7), (0, 0))

        self.l_dna = 33.5
        self.k_dna = 0.005
        self.p_dna = 0.13
        # 'p_prot': 5.88, 'p_dna': 0.13, 'k_prot': 0, 'k_dna': 0.005, 'l_dna': 335}

        super().__init__(filename, case, bond_length, residues, initial_guess, bounds, **kwargs)

    def _find_last_range(self):
        last = (25, 36)
        return last

    def _to_minimize(self, x, last_range):
        print(x)
        fit_data = self._data[self._data['d'].between(last_range[0], last_range[1])]
        # length = self.bond_length * (self.residues - 1)
        length = self.bond_length * (117 - 1)
        d_dna = get_d_dna(self.p_dna, self.l_dna, self.k_dna, fit_data['F'].to_numpy())
        fit_data = fit_data.reset_index(drop=True)
        fit_data['d'] = fit_data['d'] - d_dna
        fit_data = fit_data[fit_data['d'] > 0]
        fit_f = wlc(fit_data['d'], length, x[0], x[1])
        return np.linalg.norm(fit_f - fit_data['F'].to_numpy())

    def _fit(self, last_range):
        opt = minimize(self._to_minimize, x0=self.initial_guess, args=list(last_range), method='TNC',
                       bounds=self.bounds)
        return opt['x']

    def state_boundaries(self):

        """Finding boundaries of each state. A boundary is defined when WLC fit for each state reaches the value 12.
        Method changes histo_data field - it adds two new columns: 'begs' and 'ends'. Returns None.

        """

        begs = [round(self._smooth_data['d'].min(), 3)]
        ends = []
        for mean in self.histo_data['means']:
            cut = invert_wlc(25, self.p, self.k) * mean
            bound = self._smooth_data.iloc[(self._smooth_data['d'] - cut).abs().argsort()[:1]]['d'].to_list()[0]
            data_near = self._smooth_data.loc[self._smooth_data['F']
                                              == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                       < 1]['F'].max()]['d'].min()

            if float(43) < mean < float(45):
                data_near = self._smooth_data.loc[self._smooth_data['F']
                                                  == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                           < 3]['F'].max()]['d'].min()

            data_near = round(data_near, 3)
            if mean == self._histo_data['means'].iloc[-1]:
                ends.append(self._smooth_data['d'].iloc[-1])
                break
            begs.append(data_near)
            ends.append(data_near)

        # boundaries = [[begs[i], ends[i]] for i in range(len(begs))]
        self.histo_data['begs'] = begs
        self.histo_data['ends'] = ends

    def plot_fd(self, position):
        new_data = self._data
        d_dna = get_d_dna(self.p_dna, self.l_dna, self.k_dna, new_data['F'].to_numpy())
        new_data['d'] = new_data['d'] - d_dna
        sm_new = smooth_data(new_data)

        # position.plot(self._data.sort_values(by='d')['d'], self._data.sort_values(by='d')['F'])
        position.plot(new_data.sort_values(by='d')['d'], new_data.sort_values(by='d')['F'])
        position.plot(sm_new['d'], sm_new['F'], color=mcolors.CSS4_COLORS['pink'])
        if hasattr(self, "_data_inverse"):
            position.plot(self._data_inverse.sort_values(by='d')["d"], self._data_inverse.sort_values(by='d')['F'],
                          color=mcolors.CSS4_COLORS['lightsteelblue'])
        # position.plot(self._smooth_data['d'], self._smooth_data['F'], color=mcolors.CSS4_COLORS['pink'])
        index = 0
        for mean in self.histo_data['means']:
            residues = 1 + int(mean / self.bond_length)
            d_space = np.linspace(1, self._data['d'].max())

            label = "L= " + str(round(mean, 3)) + ' (' + str(residues) + ' AA)'
            y_fit = wlc(d_space, mean, self.p, self.k)

            position.plot(d_space, y_fit, ls='--', linewidth=1, label=label, color=get_color(index))
            index += 1

        position.set_ylim(0, 45)
        position.set_xlim(min(self._data['d']), max(self._data['d']))
        position.set_title('Trace fits')
        position.set_xlabel('Extension [nm]')
        position.set_ylabel('Force [pN]')
        position.legend(fontsize='small')

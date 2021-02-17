from Trajectory import Trajectory
from Tools import *


class Theory(Trajectory):

    def __init__(self, filename, case, **kwargs):

        bond_length = 0.38
        residues = 240
        initial_guess = [0.7, 0.005]
        bounds = ((0.3, 0.8), (0.0009, 0.009))

        super().__init__(filename, case, bond_length, residues, initial_guess, bounds, **kwargs)

    def state_boundaries(self):
        begs = [round(self._smooth_data['d'].min(), 3)]
        ends = []

        for mean in self.histo_data['means']:
            cut = invert_wlc(12, self.p, self.k) * mean
            bound = self._smooth_data.iloc[(self._smooth_data['d'] - cut).abs().argsort()[:1]]['d'].to_list()[0]
            data_near = self._smooth_data.loc[self._smooth_data['F']
                                              == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                       < 2]['F'].max()]['d'].min()
            # dealing with 72 boundary
            if float(70) < mean < float(74):

                try:
                    maximas = argrelextrema(
                        self._smooth_data.loc[abs(self._smooth_data['d'] - bound) < 5]['F'].to_numpy(),
                        np.greater)
                    data_near = self._smooth_data.loc[abs(self._smooth_data['d'] - bound) < 5]['d'].to_numpy()[
                        maximas[0].min()]
                except:
                    data_near = self._smooth_data.loc[self._smooth_data['F']
                                                      == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                               < 2]['F'].max()]['d'].min()

            # dealing with the wide one
            if float(30) < mean < float(35):
                data_near = self._smooth_data.loc[self._smooth_data['F']
                                                  == self._smooth_data.loc[abs(self._smooth_data['d'] - bound)
                                                                           < 5]['F'].max()]['d'].min()

            data_near = round(data_near, 3)
            if mean == self._histo_data['means'].iloc[-1]:
                ends.append(data_near)
                break
            begs.append(data_near)
            ends.append(data_near)

        # boundaries = [[begs[i], ends[i]] for i in range(len(begs))]
        self._histo_data['begs'] = begs
        self._histo_data['ends'] = ends

    def plot_fd(self, position):

        if hasattr(self, "_data_inverse"):
            position.plot(self._data_inverse["d"], self._data_inverse['F'], color=mcolors.CSS4_COLORS[
                'lightsteelblue'])
        position.plot(self._data.sort_values(by='d')['d'], self._data.sort_values(by='d')['F'])
        position.plot(self._smooth_data['d'], self._smooth_data['F'], color=mcolors.CSS4_COLORS['pink'])
        index = 0
        for mean in self.histo_data['means']:
            residues = 1 + int(mean / self.bond_length)
            d_space = np.linspace(1, self._data['d'].max())

            label = "L= " + str(round(mean, 3)) + ' (' + str(residues) + ' AA)'
            y_fit = wlc(d_space, mean, self.p, self.k)

            position.plot(d_space, y_fit, ls='--', linewidth=1, label=label, color=get_color(index))
            index += 1

        position.set_ylim(0, 12)
        position.set_xlim(min(self._data['d']), max(self._data['d']))
        position.set_title('Trace fits')
        position.set_xlabel('Extension [nm]')
        position.set_ylabel('Force [pN]')
        position.legend(fontsize='small')

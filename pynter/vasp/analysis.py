import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pymatgen.core.periodic_table import Element
from pymatgen.analysis.eos import EOS

matplotlib.rcParams.update({'font.size': 15})


class DatasetAnalysis:

    def __init__(self, jobs):
        """
        Class to analyse multiple Jobs

        Parameters
        ----------
        jobs : 
            List of VaspJob objects
        """
        self.jobs = jobs

    def birch_murnaghan(self, title=None):
        """
        Fit E,V data with Birch-Murnaghan EOS

        Parameters
        ----------
        title : (str)
            Plot title

        Returns
        -------
        plt :
            Matplotlib object.
        eos_fit : 
            Pymatgen EosBase object.
        """
        V, E = [], []
        for j in self.jobs:
            V.append(j.initial_structure.lattice.volume)
            E.append(j.final_energy)
        eos = EOS(eos_name='birch_murnaghan')
        eos_fit = eos.fit(V, E)
        eos_fit.plot(width=10, height=10, text='', markersize=15, label='Birch-Murnaghan fit')
        plt.legend(loc=2, prop={'size': 20})
        if title:
            plt.title(title, size=25)
        plt.tight_layout()

        return plt, eos_fit

    def plot_convergence_encut(self, title=None, delta=1e-03, get_value=False):
        """
        Plot total energy convergence with respect to ENCUT

        Parameters
        ----------
        title : (str), optional
            Plot title. The default is None.
        delta : (float), optional
            Value of delta E for energy convergence. The default is 1e-03.
        get_value : (bool)
            Return also converged cutoff value.

        Returns
        -------
        plt : 
            Matplotlib object. Returns a tuple with converged cutoff value if get_value is True.
        """
        plt.figure(figsize=(10, 8))
        cutoff_max = 0
        energies = {}
        for j in self.jobs:
            cutoff = j.incar['ENCUT']
            if cutoff > cutoff_max:
                cutoff_max = cutoff
            energies[cutoff] = j.final_energy / len(j.initial_structure)

        E0 = energies[cutoff_max]
        cutoffs = [c for c in energies.keys()]
        deltas = [E0 - e for e in energies.values()]

        convergence = False
        for i in range(0, len(cutoffs)):
            if i != 0:
                if i != len(cutoffs) - 1:
                    if abs(deltas[i] - deltas[i - 1]) < delta and abs(deltas[i] - deltas[i + 1]) < delta:
                        convergence = True
                else:
                    if abs(deltas[i] - deltas[i - 1]) < delta:
                        convergence = True
                if convergence:
                    conv_cutoff = cutoffs[i]
                    conv_delta = deltas[i]
                    break

        if not convergence:
            print(f'Convergence not reached for delta E = {delta}')

        plt.plot(cutoffs, deltas, 'o--', markersize=10)
        if convergence:
            plt.scatter(conv_cutoff, conv_delta, color=[], edgecolor='r', linewidths=3, s=450)
        plt.xlabel('PW cutoff (eV)')
        plt.ylabel('Energy difference (eV/atom)')
        plt.grid()

        if get_value:
            return plt, conv_cutoff
        else:
            return plt

    def plot_convergence_kpoint(self, title=None, delta=1e-03, get_value=False):
        """
        Plot total energy convergence with respect to K-Point Mesh.

        Parameters
        ----------
        title : (str), optional
            Plot title. The default is None.
        delta : (float), optional
            Value of delta E for energy convergence. The default is 1e-03.
        get_value : (bool)
            Return also converged cutoff value.
        Returns
        -------
        plt :
            Matplotlib object. Returns a tuple with converged cutoff value if get_value is True.
        """
        plt.figure(figsize=(10, 8))
        kpoint_sum_max = 0
        energies = {}
        for j in self.jobs:
            kpoint = j.kpoints.kpts[0]
            kpoint_label = 'x'.join([str(k) for k in kpoint])
            kpoint_sum = len(j.computed_entry.data['actual_kpoints'])  # count total number of kpoints
            if kpoint_sum > kpoint_sum_max:
                kpoint_sum_max = kpoint_sum
                kpoint_max = kpoint_label
            energies[kpoint_label] = j.final_energy / len(j.initial_structure)

        E0 = energies[kpoint_max]
        kpoints = [c for c in energies.keys()]
        deltas = [-E0 + e for e in energies.values()]

        convergence = False
        for i in range(0, len(kpoints)):
            if i != 0:
                if i != len(kpoints) - 1:
                    if abs(deltas[i] - deltas[i - 1]) < delta and abs(deltas[i] - deltas[i + 1]) < delta:
                        convergence = True
                else:
                    if abs(deltas[i] - deltas[i - 1]) < delta:
                        convergence = True
                if convergence:
                    conv_kpoint = kpoints[i]
                    conv_delta = deltas[i]
                    break

        if not convergence:
            print(f'Convergence not reached for delta E = {delta}')

        plt.plot(kpoints, deltas, 'o--', markersize=10)
        if convergence:
            plt.scatter(conv_kpoint, conv_delta, color=[], edgecolor='r', linewidths=3, s=450)
        plt.xlabel('K-Point Mesh')
        plt.ylabel('Energy difference (eV/atom)')
        plt.grid()

    def plot_fractional_charge(self, reference='electrons', name='', new_figure=True, legend_out=False):
        """
        Plot fractional charge study
        
        Parameters
        ----------
        reference: (str)
            Reference for fractional charge sign. 'electrons' or 'holes'. Default is 'electrons'.
        name: (str)
            Name to put in legend
        new_figure: (bool)
            If true initialize new matplotlib figure
        legend_out: (bool)
            If True place legend outside of the plot

        Returns
        -------
        plt : 
            Matplotlib object
        """
        if new_figure:
            plt.figure(figsize=(8, 6))
        jobs = self.jobs
        energy_dict = {}

        for j in jobs:
            if reference == 'electrons':
                n = -1 * np.around(j.charge, decimals=1)  # express in terms of occupation
            if reference == 'holes':
                n = 1 - np.around(j.charge, decimals=1)  # shift to 0 charge of +1
            energy_dict[n] = j.final_energy
        energy_dict = {k: v for k, v in sorted(energy_dict.items(), key=lambda item: item[0])}  # order by charge

        def linear(x, E0, E1):
            return (E1 - E0) * x + E0

        e_rescaled = {}
        e0 = float(energy_dict[0])
        e1 = float(energy_dict[1])
        for n in energy_dict:
            e = float(energy_dict[n])
            e_resc = e - linear(n, e0, e1)
            e_rescaled[n] = e_resc

        charges = list(e_rescaled.keys())
        energies = list(e_rescaled.values())

        ax = plt.gca()
        if not ax.lines:
            plt.hlines(0, xmin=0, xmax=1, linestyles='dashed', label='Exact', color='k')
        plt.plot(charges, energies, 'o--', label=name)
        if new_figure:
            width = max([abs(e) for e in energies])
            plt.ylim(-3 * width, +3 * width)
        plt.xlabel('Fractional charge')
        plt.ylabel('Energy (eV)')
        if legend_out:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            plt.legend()
        plt.grid()

        return plt

    def plot_U_dependency(self, element_orbital: tuple[str, str], lattice_parameters: tuple = ('a',),
                          ref_egap: float = None):
        """
        Plot dependency of lattice parameter and energy gap on U parameter assigned
        to a specific element

        Parameters
        ----------
        element_orbital : Tuple(str, str)
            Tuple of 2 strings for element and orbital to which U parameter is applied.
        lattice_parameters : (str), optional
            Lattice parameter whose dependency is evaluated. The default is 'a'.
        ref_egap: (float, optional)
            Reference energy band gap value to add %delta axis
        Returns
        -------
        plt :
            Matplotlib object
        """
        el = Element(element_orbital[0])
        U_list = []
        lattice_parameters = set(lattice_parameters)
        lattice_param_values = {p: [] for p in lattice_parameters}
        energy_gap = []
        for j in self.jobs:
            U_list.append(j.hubbards[el])
            lattice = j.final_structure.lattice
            for l_param in lattice_parameters:
                lattice_param_values[l_param].append(getattr(lattice, l_param))
            energy_gap.append(j.energy_gap)

        df = pd.DataFrame()
        df['U'] = U_list
        for l_param in lattice_param_values:
            df[l_param] = lattice_param_values[l_param]
        df['energy_gap'] = energy_gap
        df = df.sort_values('U')

        fig, (ax_lat, ax_egap) = plt.subplots(1, 2, sharex=True, figsize=(12, 8))

        x_label = f'U on {el.name}-{element_orbital[1]}'
        ax_lat.set_xlabel(x_label)
        ax_lat.set_ylabel('($\AA$)')
        for l_param in lattice_param_values:
            ax_lat.plot(df['U'].to_list(), df[l_param].to_list(), 'o--')
        ax_lat.legend(lattice_param_values.keys())
        ax_lat.grid()

        ax_egap.set_xlabel(x_label)
        ax_egap.set_ylabel('Energy gap (eV)')
        ax_egap.plot(df['U'].to_list(), df['energy_gap'].to_list(), 'o--')
        ax_egap.grid()

        if ref_egap is not None:
            ax_egap2 = ax_egap.secondary_yaxis('right',
                                               functions=(lambda band_gap: (band_gap - ref_egap) / ref_egap * 100,
                                                          lambda percent: percent * ref_egap + ref_egap / 100)
                                               )
            ax_egap2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
            ax_egap2.set_ylabel('ΔEnergy gap (%)')

        fig.tight_layout()

        return plt

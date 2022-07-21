#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:24:29 2021

@author: villa
"""

import matplotlib
import matplotlib.pyplot as plt
from pymatgen.util.plotting import pretty_plot
import pandas as pd
from pynter.defects.entries import get_formatted_legend, format_legend_with_charge

class ConcPlotter:

    def __init__(self,concentrations,format_names=True):
        """
        Class to visualize concentrations dictionaries with pd.DataFrame and 
        pd.Series (fortotal concentrations).

        Parameters
        ----------
        concentrations : (list or dict)
            Concentrations can be in different formats: list of dict for "all" and 
            "stable_charges" and dict for "total".
        format_names : (bool), optional
            Format names with latex symbols. The default is True.
        """
        self.conc = concentrations.as_dict()
        self.conc_total = concentrations.total
        self.format = format_names
        self.df = pd.DataFrame(self.conc)
        self.series = pd.Series(self.conc_total,name='Total Concentrations')
        if self.format:
            self.format_names()
    

    def __print__(self):
        return self.df.__print__()
    
    def __repr__(self):
        return self.df.__repr__()

    def copy(self):
        return self.df.copy()


    def format_names(self):
        """
        Format names with latex symbols.
        """
        format_legend = get_formatted_legend
        df = self.df
        df.name = df.name.map(format_legend)
        df.charge = df.charge.astype(str)
        df.name = df[['name', 'charge']].agg(','.join, axis=1)
        self.series.index = self.series.index.map(format_legend)
        self.df = df     
        return 
    
    
    def plot_bar(self,conc_range=(1e13,1e40),ylim=None,total=True,**kwargs):
        """
        Bar plot of concentrations with pd.DataFrame.plot

        Parameters
        ----------
        conc_range : (tuple), optional
            Range of concentrations to include in df. The default is (1e13,1e40).
        ylim : (tuple), optional
            Limit of y-axis in plot. If None conc_range is used. The default is None.
        total : (bool), optional
            plot total concentrations. The default is True.
        **kwargs : (dict)
            Kwargs to pass to df.plot().

        Returns
        -------
        plt : 
            Matplotlib object.
        """
        if conc_range:
            if not ylim:
                ylim = conc_range
            series = self.limit_conc_range(conc_range,reset_df=False)[1]
            df = self.limit_conc_range(conc_range,reset_df=False)[0]
        else:
            series = self.series
            df = self.df
        if not total:
            df.plot(x='name',y='conc',kind='bar',ylim=ylim,logy=True,grid=True,xlabel='Name , Charge',
                    ylabel='Concentrations(cm$^{-3}$)',legend=None,**kwargs)
        else:
            series.plot(kind='bar',logy=True,ylim=ylim,ylabel='Concentrations(cm$^{-3}$)',grid=True,**kwargs)

        return plt


    def limit_conc_range(self,conc_range=(1e10,1e40),reset_df=False):
        """
        Limit df to a concentration range

        Parameters
        ----------
        conc_range : (tuple), optional
            Range of concentrations to include in df. The default is (1e10,1e40).
        reset_df : (bool), optional
            Reset df attribute or return a new df. The default is False.

        Returns
        -------
        df, series : 
            DataFrame object, Series object.
        """
        if reset_df:
            self.df = self.df[self.df.conc.between(conc_range[0],conc_range[1])]
            self.series = self.series.between(conc_range[0],conc_range[1])
            return
        else:
            df = self.df.copy()
            series = self.series.copy()
            df = df[df.conc.between(conc_range[0],conc_range[1])]
            series = series[series.between(conc_range[0],conc_range[1])]
            return df , series
    
    
    
class PressurePlotter:
    """
    Class to plot oxygen partial pressure dependencies
    """
    def __init__(self,xlim=(1e-20,1e08)):
        
        self.xlim = xlim

    
    def plot_concentrations(self,thermodata,defect_indexes=None,output='total',size=(12,8),xlim=None,ylim=None):
        """
        Plot defect and carrier concentrations in a range of oxygen partial pressure.

        Parameters
        ----------
        thermodata: (ThermoData)
            ThermoData object that contains the thermodynamic data:
                partial_pressures : (list)
                    list with partial pressure values.
                defect_concentrations : (list or dict)
                    Defect concentrations in the same format as the output of DefectsAnalysis. 
                carrier_concentrations : (list)
                    List of tuples with intrinsic carriers concentrations (holes,electrons).
        defect_indexes : (list), optional
            List of indexes of the entry that need to be included in the plot. If None 
            all defect entries are included. The default is None.
        pressure_range : (tuple), optional
            Exponential range in which to evaluate the partial pressure. The default is from 1e-20 to 1e10.
        output : (str), optional
            Type of output for defect concentrations:
                "all": The output is the concentration of every defect entry.
                "stable": The output is the concentration of the stable charge for every defect at each fermi level point.
                "total": The output is the sum of the concentration in every charge for each specie.
                The default is 'total'.
        size : (tuple), optional
            Size of the matplotlib figure. The default is (12,8).
        xlim : (tuple), optional
            Range of x-axis. The default is (1e-20,1e08).
        ylim : (tuple), optional
            Range of y-axis. The default is None.

        Returns
        -------
        plt : 
            Matplotlib object.
        """
        td = thermodata.data
        p,dc,cc = td['partial_pressures'],td['defect_concentrations'],td['carrier_concentrations']
        matplotlib.rcParams.update({'font.size': 22})
        if output == 'all' or output == 'stable':
            plt = self._plot_conc(p,dc,cc,defect_indexes,output,size)
        elif output == 'total':
            plt = self._plot_conc_total(p,dc,cc,size)
            
        plt.xscale('log')
        plt.yscale('log')
        xlim = xlim if xlim else self.xlim
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel('Oxygen partial pressure (atm)')
        plt.ylabel('Concentrations(cm$^{-3})$')
        plt.legend()
        plt.grid()
        
        return plt
        

    def plot_conductivity(self,thermodata,new_figure=True,label=None,size=(12,8),xlim=None,ylim=None):
        """
        Plot conductivity as a function of the oxygen partial pressure.

        Parameters
        ----------
        thermodata: (ThermoData)
            ThermoData object that contains the thermodynamic data:
                partial_pressures : (list)
                    list with partial pressure values.
                conductivities : (dict or list)
                    If is a dict multiples lines will be plotted, with labels as keys and conductivity list
                    as values. If is a list only one line is plotted with label taken from the "label" argument.
        new_figure : (bool), optional
            Initialize a new matplotlib figure. The default is True.
        label : (str), optional
            Label for the data. The default is None.
        size : (tuple), optional
            Size of the matplotlib figure. The default is (12,8).
        xlim : (tuple), optional
            Range of x-axis. The default is (1e-20,1e08).
        ylim : (tuple), optional
            Range of y-axis. The default is None.

        Returns
        -------
        plt : 
            Matplotlib object.
        """
        td = thermodata.data
        partial_pressures,conductivities = td['partial_pressures'], td['conductivities']
        if not label:
            label = '$\sigma$'
        matplotlib.rcParams.update({'font.size': 22})
        if new_figure:
            plt.figure(figsize=(size))
        if isinstance(conductivities,dict):
            for name,sigma in conductivities.items():
                p = partial_pressures
                plt.plot(p,sigma,linewidth=4,marker='s',label=name)
        else:
            p,sigma = partial_pressures, conductivities
            plt.plot(p,sigma,linewidth=4,marker='s',label=label)
        plt.xscale('log')
    #    plt.yscale('log')
        xlim = xlim if xlim else self.xlim
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel('Oxygen partial pressure (atm)')
        plt.ylabel('Conductivity (S/m)')
        plt.legend()
        if new_figure:
            plt.grid()
        
        return plt
   
    
    def plot_fermi_level(self,partial_pressures,fermi_levels,band_gap,new_figure=True,label=None,size=(12,8),xlim=None,ylim=None):
        """
        Plot Fermi level as a function of the oxygen partial pressure.

        Parameters
        ----------
        partial_pressures : (list)
            list with partial pressure values.
        fermi_levels : (dict or list)
            If is a dict multiples lines will be plotted, with labels as keys and fermi level list
            as values. If is a list only one line is plotted with label taken from the "label" argument.
        band_gap : (float)
            Band gap of the bulk material.
        new_figure : (bool), optional
            Initialize a new matplotlib figure. The default is True.
        label : (str), optional
            Label for the data. The default is None.
        size : (tuple), optional
            Size of the matplotlib figure. The default is (12,8).
        xlim : (tuple), optional
            Range of x-axis. The default is (1e-20,1e08).
        ylim : (tuple), optional
            Range of y-axis. The default is None.

        Returns
        -------
        plt : 
            Matplotlib object.
        """
        ylim = ylim if ylim else (-0.5, band_gap+0.5)
        if not label:
            label = '$\mu_{e}$'
        matplotlib.rcParams.update({'font.size': 22})
        if new_figure:
            plt.figure(figsize=(size))
        if isinstance(fermi_levels,dict):
            for name,mue in fermi_levels.items():
                p = partial_pressures
                plt.plot(p,mue,linewidth=4,marker='s',label=name)
        else:
            p,mue = partial_pressures, fermi_levels
            plt.plot(p,mue,linewidth=4,marker='s',label=label)
        plt.xscale('log')
        xlim = xlim if xlim else self.xlim
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.hlines(0, xlim[0], xlim[1],color='k')
        plt.text(xlim[1]+(xlim[1]-xlim[0]),-0.05,'VB')
        plt.text(xlim[1]+(xlim[1]-xlim[0]),band_gap-0.05,'CB')
        plt.hlines(band_gap, xlim[0], xlim[1],color='k')
        plt.axhspan(ylim[0], 0, alpha=0.2,color='k')
        plt.axhspan(band_gap, ylim[1], alpha=0.2,color='k')
        plt.xlabel('Oxygen partial pressure (atm)')
        plt.ylabel('Electron chemical potential (eV)')
        plt.legend()
        if new_figure:
            plt.grid()
        
        return plt
    
    
    def _plot_conc(self,partial_pressures,defect_concentrations,carrier_concentrations,defect_indexes,
                   output,size):
        
        plt.figure(figsize=size)
        filter_defects = True if defect_indexes else False
        p = partial_pressures
        dc = defect_concentrations
        if output == 'stable':
            dc = dc.stable
        h = [cr[0] for cr in carrier_concentrations] 
        n = [cr[1] for cr in carrier_concentrations]
        previous_charge = None
        for i in range(0,len(dc[0])):
            if filter_defects is False or (defect_indexes is not None and i in defect_indexes):
                conc = [c[i].conc for c in dc]
                charges = [c[i].charge for c in dc]
                label_txt = get_formatted_legend(dc[0][i].name)
                if output == 'all':
                    label_txt = format_legend_with_charge(label_txt,dc[0][i].charge)
                elif output == 'stable':
                    for q in charges:
                        if q != previous_charge:
                            previous_charge = q
                            label_charge = '+' + str(q) if q > 0 else str(q)
                            index = charges.index(q)
                            plt.text(p[index],conc[index],label_charge,clip_on=True)
                plt.plot(p,conc,label=label_txt,linewidth=4)
        plt.plot(p,h,label='$n_{h}$',linestyle='--',color='r',linewidth=4)
        plt.plot(p,n,label='$n_{e}$',linestyle='--',color='b',linewidth=4)
        return plt
    
    
    def _plot_conc_total(self,partial_pressures,defect_concentrations,carrier_concentrations,size):
        
        dc = defect_concentrations
        plt.figure(figsize=size)
        p = partial_pressures
        h = [cr[0] for cr in carrier_concentrations] 
        n = [cr[1] for cr in carrier_concentrations] 
        for name in dc[0].total:
            conc = [c.total[name] for c in dc]
            label_txt = get_formatted_legend(name)
            plt.plot(p,conc,label=label_txt,linewidth=4)
        plt.plot(p,h,label='$n_{h}$',linestyle='--',color='r',linewidth=4)
        plt.plot(p,n,label='$n_{e}$',linestyle='--',color='b',linewidth=4)
        return plt
    
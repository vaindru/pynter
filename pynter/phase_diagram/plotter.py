#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:15:47 2021

@author: lorenzo
"""

import json
import os.path as op
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram, GrandPotentialPhaseDiagram, PDPlotter
from pynter.tools.format import format_composition
from pynter.phase_diagram.analysis import Reservoirs, PDHandler, ChempotAnalysis

             

class PDPlotterAdder:
    
    def __init__(self,phase_diagram=None,size=1):
        """
        Class with tools to add features to default PD plots generated by Pymatgen.

        Parameters
        ----------
        phase_diagram : (PhaseDiagram)
            Pymatgen PhaseDiagram object
        size : (float)
            Multiplier for the size of the objects added in the plot
        """
        self.pd = phase_diagram if phase_diagram else None
        self.chempots_analysis = ChempotAnalysis(phase_diagram) if phase_diagram else None
        self.size = size
        
    
    def add_points(self,points,size=1,label_size=1,color=[],edgecolor='k',label_color='k',linewidths=3,**kwargs):
        """
        Add points to plot.

        Parameters
        ----------
        points : (dict)
            Dictionary with points labels as keys and tuples or list with coordinates as values.
        size : (float)
            Float multiplier for points size. Default is 1, which would yield a default size of 450*self.size
        label_size : (float)
            Float multiplier for labels size. Default is 1, which would yield a default size of 30*self.size
        color : Color of filling of points
        edgecolor : Color of point edge
        label_color: Color of labels
        linewidths : line width of point edge
        kwargs: kwargs to pass to matplotlib

        Returns
        -------
        plt : Matplotlib object
        """
        for p in points:
            plt.scatter(points[p][0],points[p][1], color=color, edgecolor=edgecolor, linewidths=linewidths, s=450*self.size*size,**kwargs)
            plt.text(points[p][0]+(0.1/self.size*label_size),points[p][1],p,size=30*self.size*label_size,color=label_color)
        return plt
    
    
    def add_constant_chempot_line(self, comp, variable_element, fixed_chempots,**kwargs):
        """
        Add line of constant chemical potential (at a given composition) to the plot. Only works for 3 component PD.

        Parameters
        ----------
        comp : (Pymatgen Composition object)
            Composition of the phase.
        variable_element : (Pymatgen Element object)
            Element chosen as indipendent variable.
        fixed_chempots : (dict)
            Dictionary with fixed chemical potentials (values relative to reference phase). the format is {Element:chempot}
        **kwargs : 
            kwargs passed to Matplotlib plot function.
        Returns
        -------
        plt : Matplotlib object
        """
        axes = plt.gca()
        xlim , ylim = axes.get_xlim() , axes.get_ylim()
        plt.xlim(xlim)
        plt.ylim(ylim)
        mu = np.arange(xlim[0]-1,xlim[1]+1,0.01)
        plt.plot(mu,self.constant_chempot_line(mu,comp,variable_element,fixed_chempots),
                 linewidth= 4.5*self.size , **kwargs)
        return plt
    

    def add_heatmap(self,comp,elements,cbar_label='$\Delta\mu_{O}$',cbar_values=True,**kwargs):
        """
        Add heatmap that shows the value of the last chemical potential based on the values of the other two "free" 
        chemical potentials and the composition of interest. Currently works only for 3 component PDs.

        Parameters
        ----------
        comp : (Pymatgen Composition object)
            Composition of interest to compute the chemical potential.
        elements : (list)
            List of strings with elements with free chemical potentials. These will be converted in Element objects
        cbar_label : (string), optional
            String with label of the colormap. The default is ''.
        cbar_values : (tuple or bool), optional
            Show max e min chempot values on colorbar. If tuple the values are used, if not the 
            minimum chempot and 0 are used. The default is True.
        **kwargs : (dict)
            kwargs for "pcolormesh" function.

        Returns
        -------
        Matplotlib object
        """
        
        el1,el2 = elements  
        
        def f(mu1,mu2):            
            return self.chempots_analysis.calculate_single_chempot(comp,{Element(el1):mu1,Element(el2):mu2})
        
        axes = plt.gca()
        xlim , ylim = axes.get_xlim() , axes.get_ylim()
        npoints = 100
        x = np.arange(xlim[0],xlim[1]+0.1,abs(xlim[1]+0.1-xlim[0])/npoints)
        y = np.arange(ylim[0],ylim[1]+0.1,abs(ylim[1]+0.1-ylim[0])/npoints)   
        
        X,Y = np.meshgrid(x,y)
        Z = f(X,Y)

        plt.pcolormesh(X,Y,Z,vmax=0,shading='auto',**kwargs)

        cbar = plt.colorbar()
       # cbar.ax.tick_params(labelsize='xx-large')
        if cbar_values:
            if isinstance(cbar_values,tuple):
                cbar_min,cbar_max = cbar_values[0], cbar_values[1]
            else:
                cbar_min = np.around(Z.min(),decimals=1)        # colorbar min value - avoid going out of range
                cbar_max = 0                                    # colorbar max value
            plt.text(0.81,1.6,str(cbar_max),size=15)         # easier to show cbar labels as text
            plt.text(0.73,-14.2,str(cbar_min),size=15)
        cbar.set_ticks([]) # comment if you want ticks
        cbar.ax.set_yticklabels('') # comment if you want tick labels
        cbar.ax.set_ylabel(cbar_label,fontsize='xx-large')

        return plt
        

    def add_reservoirs(self,reservoirs,elements,size=1,label_size=1,color=[],edgecolor='k',label_color='k',linewidths=3,**kwargs):
        """
        

        Parameters
        ----------
        reservoirs : (Reservoirs)
            Reservoirs object.
        elements : (list)
            List with strings of the elements to be used as free variables.
        size : (float)
            Float multiplier for points size. Default is 1, which would yield a default size of 450*self.size
        label_size : (float)
            Float multiplier for labels size. Default is 1, which would yield a default size of 30*self.size
        color : Color of filling of points
        edgecolor : Color of point edge
        label_color: Color of labels
        linewidths : line width of point edge
        kwargs: kwargs to pass to matplotlib

        Returns
        -------
        plt : Matplotlib object
        """
         
        points = {}
        for r,mu in reservoirs.items():
            points[r] = [mu[Element(el)] for el in elements]
        
        return self.add_points(points,size,label_size,color,edgecolor,label_color,linewidths,**kwargs)
        
    
    def constant_chempot_line(self, mu, comp, variable_element, fixed_chempots):
        """
        Function that expresses line of constant chemical potential of a given composition. Only works for 3-component PD.

        Parameters
        ----------
        mu : (float)
            Indipendent variable of chemical potential
        comp : (Pymatgen Composition object)
            Composition of the phase.
        variable_element : (Pymatgen Element object)
            Element chosen as indipendent variable.
        fixed_chempots : (dict)
            Dictionary with fixed chemical potentials (values relative to reference phase). the format is {Element:chempot}
        """

        fixed_chempots[variable_element] = mu
        return self.chempots_analysis.calculate_single_chempot(comp,fixed_chempots)

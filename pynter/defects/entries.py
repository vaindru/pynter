#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:33:12 2020

@author: villa
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from monty.json import MSONable
import json
import numpy as np
import importlib
from pymatgen.core.units import kb
from pymatgen.core.structure import Structure
from pynter.defects.structure import defect_finder
from monty.json import MontyDecoder, MontyEncoder
from pynter.defects.elasticity import Stresses
from pynter.defects.defects import format_legend_with_charge_kv, format_legend_with_charge_number
#from pynter.defects.defects import DefectName,



class DefectEntry(MSONable,metaclass=ABCMeta):
    
    def __init__(self,defect,bulk_structure,energy_diff,corrections,data=None,label=None):
        """
        Contains the data for a defect calculation.
        
        Args:
            defect: Defect object (Vacancy, Interstitial, Substitution, Polaron or DefectComplex)
            bulk_structure: Pymatgen Structure without any defects
            energy_diff (float): difference btw energy of defect structure and energy of pure structure
            corrections (dict): Dict of corrections for defect formation energy. All values will be summed and
                                added to the defect formation energy.     
            data : (dict), optional
                Store additional data in dict format.
            label : (str), optional
                Additional label to add to defect specie. Does not influence non equilibrium calculations.
        """
        self._defect = defect
        self._bulk_structure = bulk_structure
        self._energy_diff = energy_diff
        self._corrections = corrections if corrections else {}
        self._data = data if data else {}
        self._defect.set_label(label)
   
    @property
    def defect(self):
        return self._defect  

    @property
    def bulk_structure(self):
        return self._bulk_structure
    
    @property
    def energy_diff(self):
        return self._energy_diff
    
    @property
    def corrections(self):
        return self._corrections
    
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,data):
        self._data = data
        return 

    @property
    def label(self):
        return self.defect.label
    
    @label.setter
    def label(self,label):
        self.defect.set_label(label)
        return
    
    @property
    def name(self):
        return self.defect.name
    
    @property
    def symbol(self):
        return self.defect.name.symbol

    @property
    def symbol_full(self):
        return format_legend_with_charge_number(self.symbol,self.charge)

    @property
    def symbol_kroger(self):
        return format_legend_with_charge_kv(self.symbol,self.charge)

    @property
    def defect_type(self):
        return self.defect.defect_type   
      
    @property
    def defect_species(self):
        return self.defect.defect_species

    @property
    def charge(self):
        return self.defect.charge
    
    @property
    def multiplicity(self):
        return self.defect.multiplicity
    
    @multiplicity.setter
    def multiplicity(self,multiplicity):
        self.defect.set_multiplicity(multiplicity)
        return

    @property
    def delta_atoms(self):
        """
        Dictionary with Element as keys and particle difference between defect structure
        and bulk structure as values.
        """
        return self.defect.delta_atoms
        

    def __repr__(self):
        return "DefectEntry: Name=%s, Charge=%i" %(self.name,self.charge)

    def __str__(self):
        output = [
            "DefectEntry",
            "Defect: %s" %(self.defect.__str__()),
            "Bulk System: %s" %self.bulk_structure.composition,
            "Energy: %.4f" %self.energy_diff,
            "Corrections: %.4f" %sum([v for v in self.corrections.values()]),
            "Charge: %i" %self.charge,
            "Multiplicity: %i" %self.multiplicity,
            "Data: %s" %self.data.keys(),
            "Name: %s" %self.name,
            "\n"
            ]
        return "\n".join(output)
    
    
    @staticmethod
    def from_jobs(job_defect, job_bulk, corrections, defect_structure=None,multiplicity=1,data=None,label=None,tol=1e-03):
        """
        Generate DefectEntry object from VaspJob objects.

        Parameters
        ----------
        job_defect : (VaspJob)
            Defect calculation.
        job_bulk : (VaspJob)
            Bulk calculation.
        corrections : (dict)
            Dict of corrections for defect formation energy. All values will be summed and
            added to the defect formation energy.
        defect_structure : (Structure)
            Structure of the defect. If None the intial structure of job_defect is taken. The default is None. 
        multiplicity : (int), optional
            Multiplicity of defect within the supercell. The default is 1.
        data : (dict), optional
            Store additional data in dict format.
        label : (str), optional
            Additional label to add to defect specie. Does not influence non equilibrium calculations.
        tol : (float)
            Tolerance for defect_finder function. The default is 1e-03.

        Returns
        -------
        DefectEntry
        """ 
        defect_structure = defect_structure if defect_structure else job_defect.initial_structure
        bulk_structure = job_bulk.final_structure
        energy_diff = job_defect.final_energy - job_bulk.final_energy
        charge = job_defect.charge
        
        return DefectEntry.from_structures(defect_structure, bulk_structure, energy_diff, corrections,
                                                  charge,multiplicity,data,label,tol=tol)


    @staticmethod
    def from_structures(defect_structure,bulk_structure,energy_diff,corrections,charge=0,multiplicity=None,data=None,label=None,tol=1e-03):
        """
        Generate DefectEntry object from Structure objects.

        Parameters
        ----------
        defect_structure : (Structure)
            Defect structure.
        bulk_structure : (Structure)
            Bulk structure.
        energy_diff (float): 
            Difference btw energy of defect structure and energy of pure structure
        corrections (dict): 
            Dict of corrections for defect formation energy. All values will be summed and
            added to the defect formation energy.  
        charge : (int), optional
            Charge of the defect system. The default is 0.
        multiplicity : (int), optional
            multiplicity of defect within the supercell. The default is None.
            If not provided is calculated by Pymatgen analysing the symmetry of the structure.
        data : (dict), optional
            Store additional data in dict format.
        label : (str), optional
            Additional label to add to defect specie. Does not influence non equilibrium calculations.
        tol : (float)
            Tolerance for defect_finder function. The default is 1e-03.

        Returns
        -------
        DefectEntry
        """
        defect = defect_finder(defect_structure, bulk_structure,tol=tol)
        defect.set_charge(charge)
        defect.set_multiplicity(multiplicity)
        
        return DefectEntry(defect, bulk_structure, energy_diff, corrections,data,label)



    def formation_energy(self,vbm,chemical_potentials,fermi_level=0):
        """
        Compute the formation energy for a defect taking into account a given chemical potential and fermi_level
        Args:
            vbm(float): Valence band maximum of pure structure
            chemical_potentials (dict): Dictionary of elemental chemical potential values.
                Keys are Element objects within the defect structure's composition.
                Values are float numbers equal to the atomic chemical potential for that element.
            fermi_level (float):  Value corresponding to the electron chemical potential.
            """
            
        formation_energy = (self.energy_diff + self.charge*(vbm+fermi_level) + 
                       sum([ self.corrections[correction_type]  for correction_type in self.corrections ]) 
                        ) 
        
        if chemical_potentials:
            chempot_correction = -1 * sum([self.delta_atoms[el]*chemical_potentials[el] for el in self.delta_atoms])
        else:
            chempot_correction = 0
            
        formation_energy = formation_energy + chempot_correction
        
        return formation_energy


    def defect_concentration(self, vbm, chemical_potentials, temperature=300, fermi_level=0.0, 
                             per_unit_volume=True,occupation_function='MB'):
        """
        Compute the defect concentration for a temperature and Fermi level.
        Args:
            temperature:
                the temperature in K
            fermi_level:
                the fermi level in eV (with respect to the VBM)
        Returns:
            defects concentration in cm^-3
        """
        n = self.multiplicity * 1e24 / self.bulk_structure.volume if per_unit_volume else self.multiplicity 
        eform = self.formation_energy(vbm, chemical_potentials, fermi_level=fermi_level)
        
        if occupation_function=='FD':
            conc = n * fermi_dirac(eform,temperature)
        elif occupation_function=='MB':
            conc = n * maxwell_boltzmann(eform,temperature)
        else:
            raise ValueError('Invalid occupation function. Options are: "FD" for Fermi-Dirac and "MB" for Maxwell-Boltzmann.')
        return conc
    
    
    def relaxation_volume(self,stress_bulk,bulk_modulus,add_corrections=True): #still to decide weather to keep this method
        """
        Calculate relaxation volume from stresses. Stresses data needs to be in numpy.array format and present 
        in the "data" dictionary with realtive "stress" key. Duplicate of function that can be found in Stresses
        class in elasticity module, added here for convenience.

        Parameters
        ----------
        stress_bulk : (np.array)
            Stresses of bulk calculation.
        bulk_modulus : (float)
            Bulk modulus in GPa.
        add_corrections : (bool)
            Add correction terms from "elastic_corrections" dict (if key is present in dict).

        Returns
        -------
        rel_volume : (float)
            Relaxation volume in A°^3.
        """
        es = Stresses(stress_bulk)
        return es.get_relaxation_volume(self, bulk_modulus)



def fermi_dirac(E,T):
    """
    Returns the defect occupation as a function of the formation energy,
    using the Fermi-Dirac distribution with chemical potential equal to 0. 
    Args:
        E (float): energy in eV
        T (float): the temperature in kelvin
    """
    return 1. / (1. + np.exp(E/(kb*T)) )


def maxwell_boltzmann(E,T):
    """
    Returns the defect occupation as a function of the formation energy,
    using the exponential dependence of the Maxwell-Boltzmann distribution. 
    This is the more common approach, which is the approximation of the FD-like 
    distribution for N_sites >> N_defects.
    Args:
        E (float): energy in eV
        T (float): the temperature in kelvin
    """
    return np.exp(-1.0*E /(kb*T)) 

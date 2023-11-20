# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:02:30 2022

@author: georg.lipps
"""

##### imports

from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.stats import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import mean_absolute_error


import logomaker

import ipyparallel as ipp

from time import time

import io
import base64
from PIL import Image
from IPython.display import HTML, display


##### utility methods

### sequenc handling

def split_sequence_in_nucleotides (ds_in):
    """transform sequences (uniform length)
    pd.Series containg strings with sequences
    returns pd.DataFrame single nucleotides of the sequence per column"""
    df_out=pd.DataFrame()
    length_of_probes=len(ds_in.iloc[0])
    for i in range(length_of_probes):
        df_out[str(i)+'_nuc']=ds_in.apply(lambda seq: seq[i:i+1])
    return df_out

def hotencode_sequence (ds_in, nuc_type=None):
    """hotencode sequence
    pd.DataSeries containing sequences as string, only upper letters
    returns: 2dim np.array with #sequnces * #length_sequences*4
    df """
    df_nuc=split_sequence_in_nucleotides(ds_in)
    if nuc_type=='DNA':
        bases=['A','C','G', 'T']  #make sure that the same encoding is used in all columns
    elif nuc_type=='RNA':
        bases=['A','C','G', 'U']  #make sure that the same encoding is used in all columns
    else: raise Exception("E: Nucleotide Type not given, aborting!") 
    enc = OneHotEncoder(dtype=bool, categories=df_nuc.shape[1]*[bases], handle_unknown='ignore', sparse_output=False) #for padded sequences the gap - gives an all zero hotencodement
    enc.fit(df_nuc)
    array_out=enc.transform(df_nuc)
    return array_out

def generate_sub(X, motif_length):
    """X: hotencoded sequence as np.array(#sequences, #length_of_sequence*4)
    returns: np.array with all consecutive subsequences of self.motif_length (#number of sequences, #number of subsequences, #motif_length*4)"""
    num_seq=X.shape[0]
    len_seq=X.shape[1]//4
    sub=np.zeros((num_seq,len_seq-motif_length+1,motif_length*4), dtype=bool)  
    for i in range(len_seq-motif_length+1):
        sub[:,i,:]=X[:,4*i:4*(i+motif_length)]
    return sub

### binding energy calculation

def calculate_G(X, energies, motif_length):
    """caluculates G for each subsequence of each probe X (hotencoded sequence) with predetermined energies (energy matrix)
    Only the forward strand is used for calculation.
    return np.array of all free energies G"""
    motif_length=energies.shape[0]//4
    sub=generate_sub(X, motif_length) #hotencoded subsequences
    G=sub*energies
    Gsub=np.apply_along_axis(sum, 2, G) #energies for each position of a subsequence are added up
    Gsub=Gsub.flatten() # reshape in 1 dim np.array
    return Gsub


### logos & information

def energies2logo(energies, nuc_type=None, graphics=True):
    """energies: 1 dim. np.array of energy matrix, lenght=4*motif length
    display information logo
    returns df with energies
    """
    # reformat linear array in axis0: nucleotides axis 1: position
    if nuc_type=='DNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'T'])
    elif nuc_type=='RNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'U'])
    else: raise Exception("E: Nucleotide Type not given, aborting!")
    #print(dfenergies)
    #transform energies in to probablilities according to Boltzman distribution
    dffreq=dfenergies.applymap(lambda energies: np.exp(-energies/8.31/298))
    dffreq=(dffreq.transpose()/dffreq.sum(axis=1)).transpose()  #relative frequencies
    #transform in information df and display log
    df_info=logomaker.transform_matrix(dffreq,from_type='probability',to_type='information')
    if graphics:
        logo=logomaker.Logo(df_info, font_name='sans', figsize=(len(dfenergies), 3), color_scheme='classic')
        logo.ax.set_ylabel("I (bit)", labelpad=10)
        logo.ax.set_ylim([0,2])
        logo.ax.set_yticks([0,0.5,1,1.5,2])
        logo.ax.set_xticks(range(len(dfenergies)))
        logo
        plt.show()
    return dfenergies

def energies2energylogo(energies, nuc_type=None, graphics=True):
    """energies: 1 dim. np.array of energy matrix, lenght=4*motif length
    display energy logo
    returns df with energies
    """
    # reformat linear array in axis0: nucleotides axis 1: positions
    if nuc_type=='DNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'T'])
    elif nuc_type=='RNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'U'])
    else: raise Exception("E: Nucleotide Type not given, aborting!")
    #print(dfenergies)
    dfenergies=dfenergies/1000 #rescale to kJ/mol
    if graphics:
        logo=logomaker.Logo(-dfenergies, shade_below=.5, fade_below=.5, flip_below=False, font_name='sans', figsize=[len(dfenergies),3], color_scheme='classic')
        logo.ax.set_ylabel("$-\Delta \Delta G$ (kJ/mol)", labelpad=2)
        logo.ax.set_xticks(range(len(dfenergies)))
        logo.ax.set_ylim([-30,30])
        logo
        plt.show()
    return dfenergies*1000

def energies2information(energies):
    """energies: 1 dim. np.array of energy matrix, lenght=4*motif length
    returns information content of motif [bit]
    """
    dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'T'])
    dffreq=dfenergies.applymap(lambda energies: np.exp(-energies/8.31/298))
    dffreq=(dffreq.transpose()/dffreq.sum(axis=1)).transpose()
    df_info=logomaker.transform_matrix(dffreq,from_type='probability',to_type='information')
    return df_info.sum().sum()

def energies2matrixinfo(energies):
    """energies: 1 dim. np.array of energy matrix, lenght=4*motif length
    returns np.array with information
    """
    # reformat linear array in axis0: nucleotides axis 1: position
    dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'T'])
    #transform energies in to probablilities according to Boltzman distribution
    dffreq=dfenergies.applymap(lambda energies: np.exp(-energies/8.31/298))
    dffreq=(dffreq.transpose()/dffreq.sum(axis=1)).transpose()  #relative frequencies
    #transform in information df and display log
    df_info=logomaker.transform_matrix(dffreq,from_type='probability',to_type='information')
    return df_info.to_numpy().flatten() #return nformation vector

def positions2energylogo(energies, position_labels=None, nuc_type=None):
    """energies: 1 dim. np.array of energy matrix, lenght=4*motif length
    used to display energy logo of the positions around the core motif 
    returns df with energies
    """
    # reformat linear array in axis0: nucleotides axis 1: positions
    if nuc_type=='DNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'T'])
    elif nuc_type=='RNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'U'])
    else: raise Exception("E: Nucleotide Type not given, aborting!")
    if position_labels==None:
        position_labels=range(len(energies)//4)
    dfenergies=dfenergies/1000 #rescale to kJ/mol
    logo=logomaker.Logo(-dfenergies, shade_below=.5, fade_below=.5, flip_below=False, font_name='sans', figsize=[len(dfenergies),3], color_scheme='classic')
    logo.ax.set_ylabel("$-\Delta \Delta G$ (kJ/mol)", labelpad=2)
    logo.ax.set_xticks(ticks=range(len(energies)//4))
    logo.ax.set_xticklabels(position_labels)
    logo.ax.set_ylim([-15,15])
    logo
    plt.show()
    return dfenergies*1000

### visualisation of logos within dataframes

def energies2fig(energies, nuc_type=None):
    """energies: 1 dim. np.array of energy matrix, lenght=4*motif length
    returns information loga as matplotlib figure
    """
    # reformat linear array in axis0: nucleotides axis 1: position
    if nuc_type=='DNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'T'])
    elif nuc_type=='RNA':
        dfenergies=pd.DataFrame(energies.reshape(-1,4), columns=['A', 'C', 'G', 'U'])
    else: raise Exception("E: Nucleotide Type not given, aborting!")
    #transform energies in to probablilities according to Boltzman distribution
    dffreq=dfenergies.applymap(lambda energies: np.exp(-energies/8.31/298))
    dffreq=(dffreq.transpose()/dffreq.sum(axis=1)).transpose()  #relative frequencies
    #transform in information df and display log
    df_info=logomaker.transform_matrix(dffreq,from_type='probability',to_type='information')
    logo=logomaker.Logo(df_info, font_name='sans', figsize=(len(dfenergies)*3, 3*3), color_scheme='classic')
    logo.ax.set_ylabel("I (bit)", labelpad=10)
    logo.ax.set_ylim([0,2])
    logo.ax.set_yticks([0,0.5,1,1.5,2])
    logo.ax.set_xticks(range(len(dfenergies)))
    fig=plt.gcf()
    plt.close()
    return fig

def fig2img(fig):
    """Convert a Matplotlib figure to an RGB PIL Image and returns it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def display_df(df_in, nuc_type=None):
    """df_in pd.DataFrame with a column energies, model
    The energy matrix in the column energies is used to generate an informatio logo which is stored as RGB PIL in image in column logo.
    The whole dataframe is then reformatted into HTML as displayed.
    returns: HTML picture of df_in with logo
    """
    def image_formatter(im): #formatter for df_in to html
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}" style="height: 50px">'
    
    def image_base64(im): #store PIL RGB as jpeg and codes in base64 for html display
        with io.BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()
        
    df_in['logo']=df_in['energies'].apply(lambda e: energies2fig(e, nuc_type=nuc_type)).apply(fig2img)
    return display(HTML(df_in.to_html(formatters={'logo':image_formatter, 'energies': lambda e: f'{int(e[0])},..', 'model': lambda m: 'suppressed'}, escape=False)))

def display_logos(df_in, nuc_type=None):
    """df_in pd.DataFrame with conataining only columns with energies that is 1 dim np.arrays
    The energy matrix in the column energies is used to generate an informatio logo which is stored as RGB PIL in image in column logo.
    The whole dataframe is then reformatted into HTML as displayed.
    returns: HTML picture of df_in as logos
    """
    def image_formatter(im): #formatter for df_in to html
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}" style="height: 50px">'  #style="height:836px; width:592px"
    
    def image_base64(im): #store PIL RGB as jpeg and codes in base64 for html display
        with io.BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()
        
    df_out=df_in.applymap(lambda e: energies2fig(e, nuc_type=nuc_type)).applymap(fig2img)
    return display(HTML(df_out.to_html(formatters= [image_formatter]*len(df_in.columns), escape=False)))

### anaylsis of motif

def explore_energies(model, X, y, energies):
    """Calculates the occupancy of each probe in X (hotencoded sequence) with predetermined energies (energy matrix)
    Most important statistics on occupancy are reported and graphics are produced for visualization.
    This function uses the methods of findmotif but the internal_model is never fitted."""
    
    # adapt model to user definded energies
    model.energies_=energies
    model.motif_length=len(energies)//4
    sub=generate_sub(X, model.motif_length)
    binding,_=model.calculate_binding(sub,model.finalG0_,model.energies_)
    r=linregress(y, binding).rvalue
    df_pred=pd.DataFrame({'y': y, 'binding':binding})
    
    print('number of samples: %i' %(X.shape[0]))
    print('pearson  r: %3.4f' %(r))
    print('binding:    %3.4f .. %3.4f (ratio: %3.1f)' %(binding.min(),binding.max(), binding.max()/binding.min())) 
    print('\n',abs(model.energies_.reshape(model.motif_length,4)).sum(axis=1).astype(int))
    print('energy matrix - mean/abs: %i    max/abs: %i' %(np.mean(abs(model.energies_)), np.max(abs(model.energies_))))
    
    sns.lmplot(data=df_pred, x='y', y='binding')
    sns.jointplot(data=df_pred, x='y', y='binding', kind='hist', cmap='vlag')
    return model

def roc(df_in, threshold, top=None, graphics=True):
    """ df_in: pd.DataFrame with columns y, y_pred
    threshold: for classification of a positive probe
    top: how many of the top-scoring probes are shown in the heatplot
    display two plots and 
    returns AUROC"""

    df_in['positive probe']=df_in['y'].apply(lambda y: True if y>threshold else False)
    fpr, tpr, thresholds = roc_curve(df_in['positive probe'], df_in['y_pred'])
    
    if graphics:
        #plot ROC curve
        df_roc=pd.DataFrame({'true pos. rate': tpr, 'false pos. rate': fpr, 'thresholds': thresholds})
        df_roc.plot(kind='line', x='false pos. rate', y='true pos. rate')
        plt.show()
    
        # plot heatmap of top probes of y_pred sorted
        if type(top)==type(None):
            all_probes=len(df_in)
            pos_probes=len(df_in[df_in['positive probe']])
            max_probes=5_000
            min_probes=500
            top=min(max(min(4*pos_probes, max_probes), min_probes), all_probes) #top is between min_probes or all_probes and max_probes and at best 3*pos_probes
        df_in=df_in.sort_values('y_pred', ascending=False)
        fig, ax = plt.subplots(figsize=(1,3))
        sns.heatmap(df_in['positive probe'].to_numpy().reshape((-1,1))[:top], cmap='vlag', ax=ax, cbar=False, yticklabels=top//2-1, xticklabels=False)
        ax.set_yticklabels([1, top//2, top]) # labels not perfectly placed for small numbers of top
        #ax.set_xticks([])
        plt.show()
    return roc_auc_score(df_in['positive probe'], df_in['y_pred'])


### energy matrix calculation

def acg2acgt(energies_acg):
    """energies_acg: energy matrix as np.array(#motif_length*3) representing the energies of the bases a,c and g
    returns: energy matris as np.array(#motif_length*4) representing the energies of all bases, the energies of one position add up to 0"""
    acg=energies_acg.reshape(-1,3)     #reshaped matrix for bases a,c,g 
    t=-acg.sum(axis=1).reshape(-1,1)   #calculate energies for base t/u (sum of all bases = 0)
    energies=np.concatenate((acg,t), axis=1).flatten()
    return energies

### manipulate energy matrices

def reverse_complement(energies):
    """reverses energies array with hotencoded structure first pos:acgt 2nd position :acgt ...
    reversed energies array has structure last pos: tgca 2nd last positions tgca
    Thus the reversed energies are in affect the reverse complement for the calculation"""
    return energies[::-1]

def symmetrize_energies(energies):
    """imposes 2 fold symmetry on energy matrix, can be used for palindromic motifs
    return symmetrized energies"""
    return (energies + reverse_complement(energies))/2


def modify_energies(energies, end5=0, end3=0):
    """add zero energy positions end5/end3>0 or delete positions end5/end3<0 from the energy matrix"""
    if int(end5)>=1:
        energies=np.concatenate((np.array([0]*4*int(end5)), energies))
    elif int(end5) <= -1:
        energies=energies[4*-int(end5):]
    else:
        pass
  
    if int(end3)>=1:
        energies=np.concatenate((energies,np.array([0]*4*int(end3))))
    elif int(end3) <= -1:
        energies=energies[:4*int(end3)]
    else:
        pass
    return energies


### class to store the models and the most important properties of the models in a dataframe

class stage():
    """intiates a stage.df to keep track of the models achieved during the workflow
    """
    def __init__(self, protein=None):
        self.df = pd.DataFrame()
        self.protein = protein
        return

    def append(self,stage_name, model, new_entries={}):
        """adds one row to stage.df with the standard information of a single model
        additional information can be added through the new_entries dictionary
        returns: stage.df
        """
        stage_df = pd.DataFrame({**{
            'stage':stage_name,
            'protein': self.protein,
            '# probes': model.len_X,
            'stage': stage_name,
            'motif length': model.motif_length,
            'r': model.rvalue,
            'AUROC': model.auroc,
            'G0': model.finalG0_,
            'G0 fitted': model.fit_G0,
            'ratio': model.ratio,
            'max binding': model.max_binding_fit,
            'min binding': model.min_binding_fit,
            'energies': [model.energies_],
            'model': model}, **new_entries})
        self.df=pd.concat([self.df, stage_df], ignore_index=True)
        return

##### scikit-lean Regressor

class findmotif(BaseEstimator, RegressorMixin):
    def __init__(self,
                 G0=None, fit_G0=False, max_bind_target=1, G0_offset=0,               
                 protein_conc=1, motif_length=2, both_strands=None,
                 time_dissociation=0,  
                 global_optimization=False, start=None,
                 weight_ratio=0,
                 weight_information=0,
                 ftol=None, xtol=None,
                 penalty_max=0.4, threshold_base = 17500, penalty_base=3E-9, kon=1e5, ratio_half_max=50):

        
        self.G0=G0                              #free energies of binding to an unspecific sequence, if set to None a sensible G0 is used
        self.fit_G0=fit_G0                      #whether G0 should be also fitted
        self.max_bind_target=max_bind_target    #maximal protein/probe occupancy aimed at when Go is fitted
        self.G0_offset=G0_offset                #option to correct the estimated G0
        self.penalty_max=penalty_max            #penalty weight for deviation of max_binding from max_bind_target (required when G0 is fitted)
        
        self.protein_conc = protein_conc    #protein concentration used in binding experiment in uM
        self.both_strands=both_strands      #True: reverse complement strand also considered (for double-stranded nucleic acids)
        self.motif_length = motif_length    #length of subsequenz/motif
        
        self.threshold_base = threshold_base        #base-specific energies above threshold base J/mol are penalized
        self.penalty_base=penalty_base              #penalty weight for large base specific energies / position
         
        self.global_optimization=global_optimization    #local or global optimization
        self.start=start                                #(optional) initial guess for local optimization
        
        self.time_dissociation=time_dissociation    #time span [s] after binding experiment under conditions of dissociation
        self.kon=kon                                #kon [M-1 s-1] association rate between protein and nucleic acid, 1E5 reasonable association rate
        
        self.ratio_half_max=ratio_half_max          #ratio_half_max=log(max/min) ===> r*(1+weight_ratio/2)
        self.weight_ratio=weight_ratio              #r-value multiplied with (1+weight_ratio*ratio_half_max/(ratio_half_max+log(max/min))
        self.weight_information=weight_information  #weigth for taking into consideration the information content of the motif
        
        self.auroc=None                             #variables set by analysis_motif
        self.mae=None
        
        self.ftol=ftol                              #termination criteria for local optimization
        self.xtol=xtol                              

    
   
    def calculate_binding(self,sub,G0,energies):
        """G0: free energy of binding to unspecific sequence
        energies: energy matrix as np.array(4*length of subsequence), postive or negative, if negative improve binding
        returns occupancy of each sequence as np.array(#number of sequences)
        """
        #this is the calculation for the forward probe sequence
        G=sub*energies #calculate energies for each base of all subsequences
        Gsub=np.apply_along_axis(sum, 2, G)+G0 #sum-up over all positions of a subsequence for each subsequence and add G0
        
        f_eq=lambda G: self.protein_conc/(self.protein_conc+(1/np.exp(-G/8.31/298)*1E6))
        f_diss=lambda G: np.exp(-self.time_dissociation*1/np.exp(-G/8.31/298)*self.kon)

        #  f_eq: (1/np.exp(-dG/R/T)*1E6) equals to Kd [uM]
        #  first part of expression represents fractional saturation at given protein concentration before dissociation
        #  f_diss: 1/(-G/8.31/298)*kon = koff
        #  np.exp(-dis_time*1/np.exp(-G/8.31/298)*kon) decrease due to dissociation
    
        bindsub_eq=f_eq(Gsub)  #calculate binding occupancy at a subsequence
        bindsub_diss=bindsub_eq*f_diss(Gsub)

       
        bind_eq=np.apply_along_axis(sum, 1, bindsub_eq) #cumulated occupancy over all subsubsequences (=over complete probe)
        bind_diss=np.apply_along_axis(sum, 1, bindsub_diss)
        
        if self.both_strands:
            #calculate also for reverse strand if both strands must be analyzed (double-stranded DNA)
            energies_rc=energies[::-1] 
            #reverses energies array with hotencoded structure first pos:acgt 2nd position :acgt ...
            #reversed energies array has structure last pos: tgca 2nd last positions tgca
            #Thus the reversed energies are in affect the reverse complement for the calculation
            G_rc=sub*energies_rc
            Gsub_rc=np.apply_along_axis(sum, 2, G_rc)+G0
            bindsub_eq_rc=f_eq(Gsub_rc)
            bindsub_diss_rc=bindsub_eq_rc*f_diss(Gsub_rc)            
            bindsub_eq+=bindsub_eq_rc #sum up occupancies on both strands
            bindsub_diss+=bindsub_diss_rc
            np.clip(bindsub_eq,0,1,out=bindsub_eq) # limit to maximal occupancy of 1 per subsite
            np.clip(bindsub_diss,0,1,out=bindsub_diss)
            bind_eq=np.apply_along_axis(sum, 1, bindsub_eq) #cumulated occupancy over all subsubsequences (=over complete probe)
            bind_diss=np.apply_along_axis(sum, 1, bindsub_diss)
        elif self.both_strands==False:
            pass
        else: 
            raise Exception("E: Boolean parameter both_strands not defined!")    
        return bind_diss, bind_eq
    
    def calculateGforoccupancy(self, G, occupancy):
        """helper function to fit G so that occupancy is reached
        """
        #return (self.protein_conc/(self.protein_conc+(1/np.exp(-G/8.31/298)*1E6)) * np.exp(-self.time_dissociation*1/np.exp(-G/8.31/298)*self.kon)-occupancy) 
        return (self.protein_conc/(self.protein_conc+(1/np.exp(-G/8.31/298)*1E6))-occupancy)     
    
    def fit(self, X, y):
        """X: hotencoded sequences as np.array(#sequences, #length_of_sequence*4)
        y: meassured binding np.array(#sequences)
        calculates energy matrix describing motif, predicted occupancies per sequence and regression parameters                              
        returns self
        """
        
        X, y = check_X_y(X, y)
        
        if type(self.start)!=type(None):
            if len(self.start)!=4*self.motif_length:
                raise ValueError('E: Start energies does not match motif_length. For each nucleotide position four energies are required.')
        
        if X.shape[1]%4!=0:
            raise ValueError('E: The hotencoded input sequence is not a mutiple of 4 - Aborting!')
        
        if X.shape[1]/4<self.motif_length:
            raise ValueError('E: The hotencoded input sequence is shorter than the motif length - Aborting!')
        
        self.len_X=len(X)
        
        sub=generate_sub(X, self.motif_length) #hotencoded sequence is split up in subsequences of length motif_length

        #set bounds for optimizations
        maxGperbase=1.5*self.threshold_base    # J/mol maximal base-specific energy; values above threshold_base are already penalized
        bounds=((-maxGperbase*1.01,maxGperbase*1.01), (-maxGperbase,maxGperbase),
                (-maxGperbase*0.99,maxGperbase*0.99))*self.motif_length  # bounds are required for global optimization
        
        #set initial (random) guess for optimization if not set by user
        if type(self.start)==type(None):
            start=(np.random.rand(self.motif_length*3)-0.5)*self.threshold_base/3 # set random start for localization  !! /2?
        else:
            start=np.split(self.start.reshape(-1,4), [0,3], axis=1)[1].flatten() #reformat full start energy matrix into energy matrix with acg only
            
        #set reasonable G0 if not set by user
        occupancy=0.1
        G_low_occupancy=optimize.root_scalar(self.calculateGforoccupancy, args=(occupancy), x0=-10_000, method='brentq', bracket=[-100_000,+100_000]).root
        lower_end_Gdistribution=(2*10000)*np.sqrt((self.motif_length)/12)*2.3 #first part standard deviation of a normal distribution Irwin Hall distribution  
                                                                                     #with motif_length number of uniform distributions between -10000 and +10000
                                                                                     # *2.3 to adjust for lowest 1%
        estimatedG0=G_low_occupancy+lower_end_Gdistribution+self.G0_offset
        G0=estimatedG0 if self.G0==None else self.G0 #G0 ist set here but will be included in fitting if self.fit_G0 is True
          
        # define wrapper functions for optimization with and without fitting G0
        def target(energies_ACG):  
            energies=acg2acgt(energies_ACG)
            binding_diss,_=self.calculate_binding(sub,G0,energies)
            r=linregress(y, binding_diss).rvalue  #regression (shall be positive)
            penalty_base=np.sum(np.maximum(abs(energies)- self.threshold_base, 0)**2)*self.motif_length*self.penalty_base   #penalty if energy of a base is too high 
            max_min=(max(binding_diss)/max(min(binding_diss),1e-10))                                                                  #or use log?
            factor_ratio= 1+self.weight_ratio*max_min/(self.ratio_half_max + max_min)-self.weight_ratio/2 # favor higher ratios
            exemption_information=self.weight_information*np.log(energies2information(energies)/self.motif_length) #favor more information content
            return -r*factor_ratio + penalty_base - exemption_information
        
        
        def targetG0(G0energies_ACG):
            G0=G0energies_ACG[0]  # split-up parameters
            energies=acg2acgt(G0energies_ACG[1:])
            binding_diss,binding_eq=self.calculate_binding(sub, G0, energies)
            r=linregress(y, binding_diss).rvalue  #regression (shall be positive)
            penalty_base=np.sum(np.maximum(abs(energies)- self.threshold_base, 0)**2)*self.motif_length*self.penalty_base   #penalty if energy of a base is too high 
            penalty_maxbinding=np.log10(max(binding_eq)/self.max_bind_target)**2*self.penalty_max                                            
            max_min=(max(binding_diss)/max(min(binding_diss),1e-10))                                                 
            factor_ratio= 1+self.weight_ratio*max_min/(self.ratio_half_max + max_min)-self.weight_ratio/2 # favor higher ratios
            exemption_information=self.weight_information*np.log(energies2information(energies)/self.motif_length) #favor more information content
            return -r*factor_ratio+penalty_base+penalty_maxbinding  - exemption_information
       
        # start optimization global or local with or without G0
        options={key:value for (key,value) in {'disp': False, 'ftol':self.ftol, 'xtol':self.xtol }.items() if value != None } #options directory for local minimization
       
        if self.fit_G0:      #only fit G0 and energy matrix
            if self.global_optimization:
                res = optimize.dual_annealing(targetG0, ((G0-25000,G0+25000), *bounds))  #bounds for G0 is centered around midG0
            else:
                startG0=np.concatenate((np.array([G0]),start))
                res = minimize(targetG0, startG0, method='Powell',tol=None, callback=None, options=options, bounds=((G0-25000,G0+25000), *bounds))
        else:                #only fit energy matrix
            if self.global_optimization:
                res = optimize.dual_annealing(target, bounds)  
            else:
                res = minimize(target, start, method='Powell',tol=None, callback=None, options=options, bounds=bounds)
                                             
        # finalize with results of optimization
        self.raw_score=res.fun   #r-value corrected with penalties
        self.success=res.success
        self.message=res.message
        
        if self.fit_G0:
            self.finalG0_=res.x[0]
            self.energies_=acg2acgt(res.x[1:])
        else:
            self.energies_=acg2acgt(res.x)
            self.finalG0_=G0
            
        self.binding_fit, self.binding_fit_eq=self.calculate_binding(sub,self.finalG0_, self.energies_)
        self.max_binding_fit=max(self.binding_fit)
        self.min_binding_fit=min(self.binding_fit)
        np.seterr(divide = 'ignore')
        self.ratio=self.max_binding_fit/self.min_binding_fit
        np.seterr(divide = 'warn')
        try:
            regression=linregress(self.binding_fit,y)
            self.slope_=regression.slope
            self.intercept_=regression.intercept
            self.rvalue=regression.rvalue
        except:   # exception if self.binding_fit is constant
            self.slope_=0
            self.intercept_=np.mean(y)
            self.rvalue=0
        return self


    def analyse_motif(self, X, y, threshold, nuc_type=None, graphics=True):
        """carries out several analysis on the fitted model and display the corresponding graphics,
        calculates auroc
        returns self """
        
        # plot logos and energy matrix
        print('I: energy matrix and logos:\n')
        df_energy=energies2logo(self.energies_, nuc_type, graphics=graphics)
        energies2energylogo(self.energies_, nuc_type, graphics=graphics)
        print('\n %s\n'%df_energy.astype(int))
        df_positions=df_energy.apply(abs).sum(axis=1)
        print('I: summed absolute energies of each position:\n%s\n'%df_positions.astype(int))
        print('I: averaged summed energy over all positions: %i'%int(df_positions.mean()))
        
        # plot histogramm of free energies G over all probes and subsequences
        # calculate free binding energies G over all subsequences forward
        Gsub=calculate_G(X, self.energies_, self.motif_length)
        Gsub+=self.finalG0_
        #calculate free binding energies G over all subsequences reverse if double-stranded
        if self.both_strands:
            energies_rc=self.energies_[::-1]
            Gsub_rc=calculate_G(X, energies_rc,self.motif_length)
            Gsub=np.concatenate((Gsub,Gsub_rc))
            
        if graphics:
            mean, variance = norm.fit(Gsub)
            print('I: Mean and Standard Deviation for the Free Energy G to all subsequences of all probes: %i +/- %i' %(mean, variance))
            # prepare dataframes 
            df_Gsub=pd.DataFrame({'Gsub':Gsub})
            df_norm=pd.DataFrame({'G': np.linspace(min(Gsub)-10000, max(Gsub)+10000, 100)})
            rv_norm=norm(mean, variance)
            df_norm['frequency']=df_norm['G'].apply(rv_norm.pdf)
            df_occupancy = pd.DataFrame({'G': np.arange(-80000, 0, 100)})
            f=lambda G: self.protein_conc/(self.protein_conc+(1/np.exp(-G/8.31/298)*1E6)) * np.exp(-self.time_dissociation*1/np.exp(-G/8.31/298)*self.kon)
            df_occupancy['occupancy']=df_occupancy['G'].apply(f)
            # plot histogramm of all free energies and overlay with normal distribution  and overlay with occupancy of subsite as function of free energy G
            print('I: Plot of the Occupancy of a subsite as the function of the Free Energy G \n   overlaid with the distribution of the Free Energy of all subsites.')
            print('I: There shall be only a small overlap of both curves. i.e. only the most negative Free Energies\n    lead to a measurable occupancy.')
            fig, ax=plt.subplots()
            ax2=ax.twinx()
            df_occupancy.plot(ax=ax, x='G', y='occupancy', ylabel='Occupancy', legend=False, ylim=[0,1])
            ax.vlines(x=self.finalG0_, ymin=0, ymax=1, colors='r', linestyles='-.', lw=0.5, label='current G0')
            df_Gsub.plot(ax=ax2, kind='hist',density=True, alpha=0.5, bins=30, color='orange', legend=False)
            df_norm.plot(ax=ax2, x='G', y='frequency', color='orange', legend=False)
            plt.legend()
            plt.show()
            print('I: Calculated occupancy over all subsite of a single probe:')

        # prepare dataframes for plots
        df=pd.DataFrame({'y':y,'y_pred': self.predict(X), 'bound/probe': self.binding_predict})   
        if graphics:         # plot observed signal vs. predicted signal and occupancy respectively
            sns.lmplot(data=df, x='y', y='y_pred', scatter_kws={"s": 5}).refline(x=threshold, label='classfication threshold', color='r', lw=0.5)
            plt.legend(loc='upper left')
            sns.jointplot(data=df, x='y', y='bound/probe', kind='hist', cmap='vlag', joint_kws=dict(bins=100), marginal_kws=dict(bins=100)).refline(x=threshold, color='r', label='classfication threshold', lw=0.5)
            plt.legend(loc='upper left')
            #plt.show()
        print('   binding:  %1.5f .. %1.5f (ratio: %3.1f)' %(self.binding_predict.min(), self.binding_predict.max(), self.ratio))   #### FIXME        
        self.mae=mean_absolute_error(y, df['y_pred'])
        print('I: number of probes: %i' %(len(X)))
        print('I: Pearson Correlation  r: %3.4f' %(linregress(df['y_pred'],y).rvalue))
        print('I: mean absolute error: %3.4f' %(self.mae))
        
        ## plot receiver operating characteristic for classification diagnostics
        ## plot highest top predicted probes and color positive probes in red
        self.auroc=roc(df, threshold, graphics=graphics)
        print('I: Classification performance AUROC: %1.4f'%self.auroc)    
        return self

  
    def refit_mae(self, X, y):
        """X: hotencoded sequences as np.array(#sequences, #length_of_sequence*4)
        y: meassured binding np.array(#sequences)
        takes the model and refines it in order to minimize mae,
        calculates energy matrix describing motif, predicted occupancies per sequence and regression parameters                               
        returns self
        """
        
        X, y = check_X_y(X, y)
        sub=generate_sub(X, self.motif_length) #hotencoded sequence is split up in subsequences of length motif_length

     
        G0= self.finalG0_ 
        # define wrapper functions for optimization with and without fitting G0
        
        def target_mae(intercept_slope_energies_ACG):
            intercept=intercept_slope_energies_ACG[0]
            slope=intercept_slope_energies_ACG[1]
            energies=acg2acgt(intercept_slope_energies_ACG[2:])
            binding_diss, _=self.calculate_binding(sub,G0,energies)
            y_pred=slope*binding_diss+intercept
            mae=mean_absolute_error(y, y_pred)
            penalty=np.sum(np.maximum(abs(energies)- self.threshold_base, 0)**2)/(self.penalty_base*self.motif_length) #penalty (is positive shall be small)
            #penalty=penalty-self.weight_ratio*min(2*np.log(max(binding)/max(min(binding),1e-20)),8)/8 #ratios above e^4=54 reduce penalty by weight_ratio FIXME could be also a factor to reduce r
            return mae+penalty
        
       
        # further local optimization with already fitted model
        start_intercept=self.intercept_
        start_slope=self.slope_
        start_energies=np.split(self.energies_.reshape(-1,4), [0,3], axis=1)[1].flatten() #reformat full start energy matrix into energy matrix with acg only
        start=np.concatenate((np.array([start_intercept,start_slope]),start_energies))
        res = minimize(target_mae, start, method='Powell',tol=None, callback=None, options={'disp': True})
                                           
        # finalize with results of optimization
        self.intercept_=res.x[0]
        self.slope_=res.x[1]
        self.energies_=acg2acgt(res.x[2:])
        self.binding_fit, self.binding_fit_eq=self.calculate_binding(sub,self.finalG0_, self.energies_)
        self.max_binding_fit=max(self.binding_fit)
        self.min_binding_fit=min(self.binding_fit)
        y_pred=self.slope_*self.binding_fit+self.intercept_
        self.mae=mean_absolute_error(y, y_pred)
        self.rvalue=linregress(self.binding_fit,y).rvalue
        return self

  
   
    def predict(self, X):   
        check_is_fitted(self)
        X = check_array(X)
        sub=generate_sub(X, self.motif_length)
        self.binding_predict,_=self.calculate_binding(sub,self.finalG0_,self.energies_)
        y_pred=self.slope_*self.binding_predict+self.intercept_
        return y_pred
    
    def investigate_extension_parallel(self, X, y, nuc_type, end5=3, end3=3):
        """takes the energy matrix of the fitted solution of the core motif and extends the energy matrix 
        by additional bordering positions at the 5' and/or 3' end.
        Only a single extended position is optimized while the energymatrix of the remaining positions is kept constant.
        end5, end3: number of positions analyzed beyond the core motif (self.energies_)
        returns df with index -end5, +end3 with the r-value achieved with the singly fitted position and
        the optimzed energy matrix of this position, r over baseline % over baseline and +2% (boolean)
        """
        # define jobs for parallel execution
        def job5(start_energies, position):
            """"add position 5' to start_energies and optimizes energy of this position.
            returns position, achieved r and energy matrix of 5' position"""
            import motif as mf
            import numpy as np
            from scipy.optimize import minimize
            from scipy.stats import pearsonr

            def target_single_5(energies_ACG_first_position, energies_core, sub): 
                energies_first_position=acg2acgt(energies_ACG_first_position)
                energies=np.concatenate((energies_first_position, energies_core))
                binding_diss,_=self.calculate_binding(sub,self.finalG0_,energies)
                r=pearsonr(binding_diss,y)[0] 
                return -r
            

            sub=mf.generate_sub(X, len(start_energies)//4+1) #hotencoded sequence is split up in subsequences of motif length 
            res = minimize(target_single_5, np.array([0,0,0]), args=(start_energies, sub), method='Powell', options={'disp': False})
            return {'position': position,'r': -res.fun, 'energies': acg2acgt(res.x)}
  
        
        def job3(start_energies, position):
            """"add position 5' to start_energies and optimizes energy of this position.
            returns position, achieved r and energy matrix of 5' position"""           
            import motif as mf
            import numpy as np
            from scipy.optimize import minimize
            from scipy.stats import pearsonr
            
            def target_single_3(energies_ACG_last_position, energies_core,sub): 
                energies_last_position=acg2acgt(energies_ACG_last_position)
                energies=np.concatenate((energies_core, energies_last_position))
                binding_diss, _=self.calculate_binding(sub,self.finalG0_,energies)
                r=pearsonr(binding_diss,y)[0] 
                return -r

            sub=mf.generate_sub(X, len(start_energies)//4+1) #hotencoded sequence is split up in subsequences of motif length 
            res = minimize(target_single_3, np.array([0,0,0]), args=(start_energies, sub), method='Powell', options={'disp': False})
            return {'position': position,'r': -res.fun, 'energies': acg2acgt(res.x)}

        # prepare arguments for parallel execution
        list_start5=[]
        list_pos5=[]
        for add5 in range(int(end5)-1, -1 ,-1): #iterate over 5' positions starting from most distant
            list_start5.append(np.concatenate((np.array([0]*4*add5), self.energies_))) #add between (end5-1) .. 0 empty positions
            list_pos5.append(-(add5+1)) #positions -add5 .. -1

        list_start3=[]
        list_pos3=[]
        for add3 in range(int(end3)): #iterate over 5' positions starting from most distant
            list_start3.append(np.concatenate((self.energies_, np.array([0]*4*add3)))) #add between 0..(end3-1) empty positions
            list_pos3.append(add3+1) #positions -add5 .. -1       
                
        # start execution
        start = time()
        with ipp.Cluster(log_level=40, n=min(end3+end5, 10)) as rc:
            rc[:].use_dill()
            view = rc.load_balanced_view()
            if len(list_start5):
                asyncresult5 = view.map_async(job5, list_start5, list_pos5)
            if len(list_start3):
                asyncresult3 = view.map_async(job3, list_start3, list_pos3)
            if len(list_start5):
                asyncresult5.wait_interactive()
                result5 = asyncresult5.get()
            else: result5=[]
            if len(list_start3): 
                asyncresult3.wait_interactive()
                result3 = asyncresult3.get()
            else: result3=[]
        print("I: Optimization took %.2f hours." % ((time() - start)/3600))

       
        df_out=pd.concat([pd.DataFrame(result5),
                          pd.DataFrame({'position': 0,'r': self.rvalue, 'energies': [np.zeros(4)]}),
                          pd.DataFrame(result3)]).set_index('position')
        df_out['r over baseline']=df_out['r']-df_out.at[0,'r']
        df_out['+2%']=df_out['r over baseline']>=df_out.at[0,'r']*0.02
        fig, ax=plt.subplots()
        df_out['r over baseline'].plot(ax=ax,kind='bar').axhline(y=df_out.at[0,'r']*0.02, label='+2% increase over baseline', ls='-.',lw=0.5) # plot improvement together with line indication +2%
        ax.axhline(0, color='black', lw=0.5)
        plt.legend()
        plt.show()
        # display energies of singly optimized positions
        positions2energylogo(np.concatenate(df_out['energies'].tolist()), position_labels=df_out.index.tolist(), nuc_type=nuc_type)
        return df_out 

    def explore_positions(self, X, y):
        """Calculates the reduction of the r value of each position of the fitted self 
        when the energies of this position are set to [0,0,0,0]
        This function uses the methods of findmotif but the model is never fitted.
        A graphical overview is presented and a dataframe with pos, energies, r-value and r-value-background is returned"""
        
        length=self.motif_length
        sub=generate_sub(X, self.motif_length)
        
        df_out=pd.DataFrame({'pos': range(length)})
        df_out['energies']=df_out['pos'].apply(lambda pos:np.array(([1,1,1,1]*pos+[0,0,0,0]*1+[1,1,1,1]*(length-pos-1))*self.energies_))
        df_out['r']=df_out['energies'].apply(lambda e: linregress(y, self.calculate_binding(sub,self.finalG0_,e)[0]).rvalue)
        df_out['r under baseline']=df_out['r']-self.rvalue
        df_out['-2%']=df_out['r']<=self.rvalue*0.98
        fig, ax=plt.subplots()
        df_out['r under baseline'].plot(ax=ax,kind='bar').axhline(y=-self.rvalue*0.02, label='-2% decrease against baseline', ls='-.',lw=0.5) # plot improvement together with line indication +2%
        ax.axhline(0, color='black', lw=0.5)
        plt.legend()
        plt.show()
        return df_out

    
    def investigate_G0(self, X, y):
        """takes the energy matrix and finalG0_ of the fitted model and changes G0 in a range of -30000 ... +30000 J/mol
        plot max binding, r and ratio as function of G0 with region of sensible max occumpancy highlighted (blue lines) along with current G0 (red) and G0 of optimal r (green)
        returns df with columns G0, max binding, min binding, r, ratio
        """
        # calculate occupancies over a range of G0 values
        sub=generate_sub(X, self.motif_length) 
        df_out = pd.DataFrame({'G0': np.arange(self.finalG0_-30000, self.finalG0_+30000, 1000)})  
        df_out['binding']=df_out['G0'].apply(lambda G0: self.calculate_binding(sub, G0, self.energies_ )[0])
        df_out['min binding']=df_out['binding'].apply(min)
        df_out['max binding']=df_out['binding'].apply(max)
        df_out['r']=df_out['binding'].apply(lambda binding: pearsonr(binding,y)[0])
        np.seterr(divide = 'ignore')
        df_out['ratio']=df_out['max binding']/ df_out['min binding']
        np.seterr(divide = 'warn')
        
        # current and maximal r
        index_currentG0=(df_out['G0']-self.finalG0_).abs().argsort()[0]
        currentr=df_out.at[index_currentG0, 'r']
        print(f'I: Current G0 = {self.finalG0_:.0f} J/mol (see red broken line in figure below) with r = {currentr:.3f}.')
        index_r_max=df_out['r'].idxmax()
        G0_maxr=df_out.at[index_r_max, 'G0']
        maxr=df_out.at[index_r_max, 'r']
        print(f"I: Maximal r is {maxr:.3f} at G0={G0_maxr:.0f} J/mol (see green broken line below).")
        
        # maximal occupancy of 2 and 0.2
        index_binding_2=(df_out['max binding']-2).abs().argsort()[0]
        index_binding_0_2=(df_out['max binding']-0.2).abs().argsort()[0]
        G0_max_binding_high=df_out.at[index_binding_2, 'G0'] 
        G0_max_binding_low=df_out.at[index_binding_0_2, 'G0'] 
        print(f"I: Maximal occupancy of 2 is reached at G0={G0_max_binding_high:.0f} J/mol (see blue broken line below).")
        print(f"I: Maximal occupancy of 0.2 is reached at G0={G0_max_binding_low:.0f} J/mol (see blue broken line below).")

        #plot max_binding as function of G0 with the given self
        fig, ax=plt.subplots()
        df_out.plot(ax=ax,x='G0', y='max binding')
        ax.hlines(y=0.2, xmin=self.finalG0_-20000, xmax=G0_max_binding_low, colors='r', linestyles='-', lw=0.5)
        ax.hlines(y=2, xmin=self.finalG0_-20000, xmax=G0_max_binding_high, colors='r', linestyles='-', lw=0.5)
        ax.vlines(x=G0_max_binding_low, ymin=0.2, ymax=max(df_out['max binding']), colors='b', linestyles='-.', lw=0.5, label='0.2 and 2')
        ax.vlines(x=G0_max_binding_high, ymin=2, ymax=max(df_out['max binding']), colors='b', linestyles='-.', lw=0.5)
        ax.vlines(x=self.finalG0_, ymin=0, ymax=max(df_out['max binding']), colors='r', linestyles='-.', lw=0.5, label='current G0')
        ax.vlines(x=G0_maxr, ymin=0, ymax=max(df_out['max binding']), colors='g', linestyles='-.', lw=0.5, label='G0 with maximal r')
        plt.legend()
        plt.show()
          
        # plot quality of regression (r) and ratio of bindinng range as function of G0 of the given self
        fig, ax=plt.subplots()
        df_out.plot(ax=ax,x='G0', y=['r','ratio'], secondary_y=['ratio'])
        ax.vlines(x=[G0_max_binding_low,G0_max_binding_high], ymin=0, ymax=max(df_out['r']), colors='b', linestyles='-.', lw=0.5, label="0.2 and 2")
        ax.vlines(x=self.finalG0_, ymin=0, ymax=max(df_out['r']), colors='r', linestyles='-.', lw=0.5, label='current G0')
        ax.vlines(x=G0_maxr, ymin=0, ymax=max(df_out['r']), colors='g', linestyles='-.', lw=0.5, label='G0 with maximal r')
        #plt.legend()
        plt.show()
        if self.finalG0_>=G0_max_binding_high and self.finalG0_<=G0_max_binding_low:
            print('I: G0 is in a range leading to maximal probe occupancy between 0.2 and 2. Good.')
        else:
            if self.finalG0_<G0_max_binding_high:
                print('E: Current G0 leads to a maximal probe occupancy exceeding 2. G0 must be increased to avoid oversaturation.')
            else:
                print('W: Current G0 leads to a maximal probe occupancy below 0.2. G0 can be manuylly set and be decreased.')

        if self.finalG0_>=0.95*G0_maxr and self.finalG0_<=1.05*G0_maxr: 
            print('I: Current G0 is close to the G0 leading to maximal r. Good.')
        else:
            if abs(currentr-maxr)/currentr<0.01:
                print('I: Maximal r is close to r achieved with current G0. Good.')
            else:
                print('W: A higher r is achieved with a different G0 value. G0 might be set manually for adjustment.')
              
        return df_out
    
"""
        def fitG0(G0):  
            binding=self.calculate_binding(sub,G0,self.energies_)
            r=pearsonr(binding,y)[0] #regression (shall be positive)
            return -r
        
        G0_maxr = minimize(fitG0, self.finalG0_, method='Powell', bounds=[(self.finalG0_-20000,self.finalG0_+20000)])
        
 
        #find G0 with max binding 0.2 and 2 (reasonable range of max binding for a successful binding experiment)
        
        def fitG0_binding(G0, target_max_binding):  
            max_binding=max(self.calculate_binding(sub,G0,self.energies_))
            return max_binding-target_max_binding
       
        
        print('I: Optimizing G0 for predetermined maximal binding ...')
        G0_max_binding_low=optimize.root_scalar(fitG0_binding, args=(0.2), x0=-10_000, method='brentq', bracket=[-100_000,+100_000]).root
        G0_max_binding_high=optimize.root_scalar(fitG0_binding, args=(2), x0=-10_000, method='brentq', bracket=[-100_000,+100_000]).root
        
        print(f'        max binding    G0 [J/mol]\n'     
              f'        0.2           {G0_max_binding_low:.0f}\n'
              f'        2             {G0_max_binding_high:.0f}\n'
              f'... see blue broken lines in figure\n')
"""
    

import os,sys,rdkit,math
import rdkit.Chem as Chem
from rdkit.Chem import Draw
import numpy as np
from env import PlanetEstimator,VinaEstimator
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl

class PreAnalyse():
    def __init__(self,vina_config=None):
        self.planet = PlanetEstimator()
        if vina_config is not None:
            self.vina = VinaEstimator(0,vina_config,os.path.join(os.getcwd(),'Vina/work_tmp'))
            if not self.vina.check_executable():
                print('vina is not set correctly.')
                sys.exit()
        else:
            self.vina = None

    def read_protein(self,protein_pdb,ligand_sdf=None,x=None,y=None,z=None):
        try:
            if ligand_sdf is not None:
                self.planet.set_pocket_from_ligand(protein_pdb,ligand_sdf)
            elif x is not None and y is not None and z is not None:
                self.planet.set_pocket_from_coordinate(protein_pdb,x,y,z)
        except RuntimeError:
            print('Error happens during setting protein pocket. This usually happens when the protein pdb file \
                    has not been correctly fixed. Please prepare the protein pdb file using other third-party softwares,\
                    especially take care of alpha-carboon of each residue.')
            sys.exit()
        if self.vina is not None:
            protein_pdbqt = self.vina.prepare_protein_pdbqt(protein_pdb)
            if protein_pdbqt is not None:
                self.vina.set_protein(protein_pdbqt)
            else:
                print('Error happends when transform protein .pdb to .pdbqt file, please check.')
                sys.exit()
    
    def read_input(self,input_file):
        ### input file has the format like "smi activity"
        try:
            smis,activities = [],[]
            with open(input_file,'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        smis.append(line.split()[0])
                        activities.append(float(line.split()[1]) )
            activities = np.array(activities)
            return smis,activities
        except:
            print('Wrong input format. input file has the format like "smi activity" in each line\n \
            activity is -log Kd/Ki or IC50. for example: O=C(O)c1ccccc1 3.5 ')
            sys.exit()

    def estimate(self,smis,activities):
        out_smis,out_activities = [],[]
        planet_pred_activities = []
        vina_pred_activities = [] if self.vina is not None else None
        for (smi,activity) in zip(smis,activities):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print('error happens when reading from {}'.format(smi))
            else:
                out_smis.append(smi)
                out_activities.append(activity)
                planet_pred_activities.append(self.planet.estimate(mol))
                if self.vina is not None:
                    vina_pred_activities.append(self.vina.estimate(mol))

        return out_smis,out_activities,planet_pred_activities,vina_pred_activities

    def statistics(self,out_smis,out_activities,planet_pred_activities,vina_pred_activities):
        out_activities = np.array(out_activities)
        planet_pred_activities = np.array(planet_pred_activities)
        if vina_pred_activities is not None:
            vina_pred_activities = np.array(vina_pred_activities)
        for (pred_activities,estimator) in zip([planet_pred_activities,vina_pred_activities],['PLANET','vina']):
            if pred_activities is None:
                continue
            else:
                P_correlation,_ = stats.pearsonr(pred_activities,out_activities)
                S_correlation,_ =  stats.spearmanr(pred_activities,out_activities)  
                self.plot(estimator,out_smis,out_activities,pred_activities)
                if estimator == 'PLANET':
                    MAE = np.mean(np.abs(pred_activities-out_activities))
                    RMSE = np.sqrt(np.sum(np.square(pred_activities-out_activities)) / len(pred_activities))
                    print('Metrics for {}: MAE:{:3f}\tRMSE:{:3f}\tPearson:{:3f}\tSpearman:{:3f}'.format(estimator,MAE,RMSE,P_correlation,S_correlation))
                else:
                    print('Metrics for {}: Pearson:{:3f}\tSpearman:{:3f}'.format(estimator,P_correlation,S_correlation))

    def plot(self,estimator:str,out_smis,out_activities,pred_activities):
        fig,ax = plt.subplots(figsize=(8,8))
        ax.scatter(out_activities,pred_activities,s=10,color='blue')
        activity_range = math.ceil(max(out_activities)-min(out_activities))
        pred_range = math.ceil(max(pred_activities)-min(pred_activities))
        final_range = max([activity_range,pred_range]) + 2
        x_lim = [min(out_activities)-1,min(out_activities)+final_range]
        y_lim = [min(pred_activities)-1,min(pred_activities)+final_range]
    
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        sort_index = np.argsort(out_activities)
        draw_index = list(set(sort_index[::5] + [sort_index[-1]]))

        for i in draw_index:
            location = [out_activities[i] + 1 ,pred_activities[i] + 1]
            img = Draw.MolToImage(Chem.MolFromSmiles(out_smis[i]),size=(225,100))
            imagebox = mpl.offsetbox.AnnotationBbox(mpl.offsetbox.OffsetImage(img),location,frameon=False)
            ax.add_artist(imagebox)
            ax.annotate('',xy=(out_activities[i] ,pred_activities[i]),xytext=tuple(location),arrowprops=dict(arrowstyle='->'))
        if estimator == 'PLANET':
            ax.plot((0,1),(0,1),transform=ax.transAxes,ls='--',c='k',alpha=0.1)
        elif estimator == 'vina':
            ax.plot((0,1),(1,0),transform=ax.transAxes,ls='--',c='k',alpha=0.1)
        fig.savefig(os.path.join(os.getcwd(),'{}_results.png'.format(estimator)),dpi=300)

class PostAnalyse():
    def __init__(self,**kwargs):
        self.design_log = kwargs['logger_path']

    def scan(self):
        with open(self.design_log,'r') as f:
            line = f.readline()
            elapsed_time,smi,reward_valid,reward_qed,reward_scscore,reward_activity,reward,qed_score,scscore,activity = line.strip().split()

if __name__ == '__main__':
    from rdkit import RDLogger
    import argparse
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',required=True,choices=['pre','post'],type=str)
    parser.add_argument('-y','--yaml',required=True)
    parser.add_argument('-i','--input_file',required=True)
    #parser.add_argument('')
    args = parser.parse_args()
    if args.mode == 'pre':
        analyse = PreAnalyse(args.vina_config,args.plot)
        analyse.read_protein(args.pdb,args.ligand)
        smis,activities = analyse.read_input(args.input_file)
        out_smis,out_activities,planet_pred_activities,vina_pred_activities = analyse.estimate(smis,activities)
        analyse.statistics(out_activities,planet_pred_activities,vina_pred_activities)
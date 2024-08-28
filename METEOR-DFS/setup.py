import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk
from rdkit import Chem
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

window = tk.Tk()
screen_width,screen_height = window .winfo_screenwidth(),window .winfo_screenheight()
window.geometry('{}x{}'.format(str(screen_width//2),str(screen_height//2)))

frame_start_mol = tk.Frame(window)
frame_start_mol.grid(row=0,column=0,rowspan=3)

start_mol_label = tk.Label(frame_start_mol,text='Start mol SMILES: ')
start_mol_label.grid(row=0)

start_mol_smi = tk.StringVar()
def mol_check():
    '''
    check smi validaity of input
    '''
    try:
        mol = Chem.MolFromSmiles(start_mol_smi.get())
    except:
        mol = None
    finally:
        if mol is None:
            messagebox.showerror(title='Error', message='Not a valid SMILES string, please check!')
            return False
        else:
            return True
    
MolCheck = frame_start_mol.register(mol_check)
start_mol_entry = tk.Entry(frame_start_mol,textvariable=start_mol_smi,validate ="focusout",validatecommand=MolCheck,width='30')
start_mol_entry.grid(row=0,column=1)


allow_label = tk.Label(frame_start_mol,text='Grow from: ')
allow_label.grid(row=1)

allow_index = tk.StringVar()

def index_format_check():
    '''
    format check 
    '''
    idx_str = allow_index.get()
    if not all([i.isnumeric() for i in idx_str.split()]):
        messagebox.showerror(title='Error', message='Index format error! Example \n "Grow from: 1 2 3"')
        return False
    else:
        return True
    
IndexCheck = frame_start_mol.register(index_format_check)
allow_index_entry = tk.Entry(frame_start_mol,textvariable=allow_index,validate ="focusout",validatecommand=IndexCheck,width='30')
allow_index_entry.grid(row=1,column=1)

# mol figure
photo = tk.PhotoImage()  #.configure change photo
mol_photo = tk.Label(frame_start_mol,image=photo,background='white',width=300,height=225)
mol_photo.grid(row=2,column=0,columnspan=3,rowspan=3,sticky='we',padx='5px',pady='5px')


def update_mol_photo():
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)    
    mol = Chem.MolFromSmiles(start_mol_smi.get()) if mol_check() else None
    if index_format_check() and mol is not None:
        allow_index = list(map(int,allow_index_entry.get().split())) 
        d2d = rdMolDraw2D.MolDraw2DCairo(300,225)
        d2d.drawOptions().addAtomIndices=True
        d2d.drawOptions().setHighlightColour((0.8,0.8,0.8))
        d2d.DrawMolecule(mol,highlightAtoms=allow_index)
        d2d.FinishDrawing()
        png_bit = d2d.GetDrawingText()
        with open('mol_image.png', 'wb') as png_file:
            png_file.write(png_bit)
        img = ImageTk.PhotoImage(file='mol_image.png')
        mol_photo.configure(image=img)
        mol_photo.image = img
        os.remove('mol_image.png')
    else:
        pass
    
def check_index():
    if index_format_check():
        mol_atom_count = Chem.MolFromSmiles(start_mol_entry.get()).GetNumAtoms() 
        if max(list(map(int,allow_index_entry.get().split()))) < mol_atom_count:
            messagebox.showinfo(title = 'OK',message='Check pass.')
            return True
        else:
            if mol_atom_count == 0:
                messagebox.showerror(title='Error', message='Fill the "Start mol SMILES" first.')
            else:
                messagebox.showerror(title='Error', message='Index should not be larger than {}'.format(str(mol_atom_count)))
            return False
    else:
        return False    

update_buttom = tk.Button(frame_start_mol,text='update',command=update_mol_photo)
update_buttom.grid(row=0,rowspan=1,column=2,padx='5px',pady='3px',sticky='we')
check_buttom = tk.Button(frame_start_mol,text='check',command=check_index)
check_buttom.grid(row=1,rowspan=1,column=2,padx='5px',pady='3px',sticky='we')

###activity estimation
frame_activity = tk.Frame(window)
frame_activity.grid(row=0,column=1,sticky='ne')
activity_setting_label = tk.Label(frame_activity,text='Activity Prediction Settings ')
activity_setting_label.grid(row=0,column=0,columnspan=6)

activity_estimator_label = tk.Label(frame_activity,text='Activity Estimator: ')
activity_estimator_label.grid(row=1,column=0,columnspan=2,sticky='e')
activity_estimator = tk.IntVar(frame_activity,value=0)

vina_config_label = tk.Label(frame_activity,text='Vina config path: ')
vina_config_path = tk.StringVar()
vina_config_entry = tk.Entry(frame_activity,textvariable=vina_config_path ,width='40')

def select_estimator():
    if activity_estimator.get() == 1:
        vina_config_label.grid(row=5,column=0,columnspan=2,sticky='e')
        vina_config_entry.grid(row=5,column=2,columnspan=4,sticky='w')
    else:
        vina_config_label.grid_forget()
        vina_config_entry.grid_forget()

tk.Radiobutton(frame_activity, text="PLANET", variable=activity_estimator, value=0,command=select_estimator).grid(row=1,column=2,columnspan=2)
tk.Radiobutton(frame_activity, text="Vina", variable=activity_estimator, value=1,command=select_estimator).grid(row=1,column=4,columnspan=2)

protein_path_label = tk.Label(frame_activity,text='Protein PDB path: ')
protein_path_label.grid(row=2,column=0,columnspan=2,sticky='e')
protein_path = tk.StringVar()
protein_path_entry = tk.Entry(frame_activity,textvariable=protein_path ,width='40')
protein_path_entry.grid(row=2,column=2,columnspan=4,sticky='w')

ligand_path_label = tk.Label(frame_activity,text='Ligand SDF path: ')
ligand_path_label.grid(row=3,column=0,columnspan=2,sticky='e')
ligand_path = tk.StringVar()
ligand_path_entry = tk.Entry(frame_activity,textvariable=ligand_path ,width='40')
ligand_path_entry.grid(row=3,column=2,columnspan=4,sticky='w')

x_label = tk.Label(frame_activity,text='Cneter X: ').grid(row=4,column=0)
x_entry = tk.Entry(frame_activity,textvariable=tk.DoubleVar(),width='8').grid(row=4,column=1,sticky='e')
y_label = tk.Label(frame_activity,text='Cneter Y: ').grid(row=4,column=2)
y_entry = tk.Entry(frame_activity,textvariable=tk.DoubleVar(),width='8').grid(row=4,column=3,sticky='e')
z_label = tk.Label(frame_activity,text='Cneter Z: ').grid(row=4,column=4)
z_entry = tk.Entry(frame_activity,textvariable=tk.DoubleVar(),width='8').grid(row=4,column=5,sticky='e')

###reward settings
frame_reward = tk.Frame(window)
frame_reward.grid(row=1,column=1,sticky='n')
reward_label = tk.Label(frame_reward,text='Reward Settings')
reward_label.grid(row=0,column=0,columnspan=6,sticky='n')

qed_label = tk.Label(frame_reward,text='QED ratio: ')
qed_label.grid(row=1,column=0,sticky='e')
qed_ratio = tk.DoubleVar(frame_reward,1.0)
qed_entry = tk.Entry(frame_reward,textvariable=qed_ratio,width='5')
qed_entry.grid(row=1,column=1,sticky='e')

sa_label = tk.Label(frame_reward,text='SA ratio: ')
sa_label.grid(row=2,column=0,sticky='e')
sa_ratio = tk.DoubleVar(frame_reward,value=0.5)
sa_entry = tk.Entry(frame_reward,textvariable=sa_ratio,width='5')
sa_entry.grid(row=2,column=1,sticky='e')

sa_lower_label = tk.Label(frame_reward,text='SA lower: ')
sa_lower_label.grid(row=2,column=2,padx='3px',sticky='e')
sa_lower = tk.DoubleVar(frame_reward,value=12.0)
sa_lower_entry = tk.Entry(frame_reward,textvariable=sa_lower,width='5')
sa_lower_entry.grid(row=2,column=3)

sa_higher_label = tk.Label(frame_reward,text='SA higher: ')
sa_higher_label.grid(row=2,column=4,sticky='e',padx='3px')
sa_higher = tk.DoubleVar(frame_reward,value=30.0)
sa_higher_entry = tk.Entry(frame_reward,textvariable=sa_higher,width='5')
sa_higher_entry.grid(row=2,column=5,sticky='w')

activity_label = tk.Label(frame_reward,text='Act. ratio: ')
activity_label.grid(row=3,column=0,sticky='e')
activity_ratio = tk.DoubleVar(frame_reward,2.5)
activity_entry = tk.Entry(frame_reward,textvariable=activity_ratio,width='5')
activity_entry.grid(row=3,column=1,sticky='e')

activity_lower_label = tk.Label(frame_reward,text='Act. lower: ')
activity_lower_label.grid(row=3,column=2,padx='3px',sticky='e')
activity_lower = tk.DoubleVar(frame_reward,value=3.0)
activity_lower_entry = tk.Entry(frame_reward,textvariable=activity_lower,width='5')
activity_lower_entry.grid(row=3,column=3)

activity_higher_label = tk.Label(frame_reward,text='Act. higher: ')
activity_higher_label.grid(row=3,column=4,sticky='e',padx='3px')
activity_higher = tk.DoubleVar(frame_reward,value=8.0)
activity_higher_entry = tk.Entry(frame_reward,textvariable=activity_higher,width='5')
activity_higher_entry.grid(row=3,column=5,sticky='w')

memory_min_label = tk.Label(frame_reward,text='Memory min: ')
memory_min_label.grid(row=4,column=0,sticky='e')
memory_min = tk.IntVar(frame_reward,value=20)
memory_min_entry = tk.Entry(frame_reward,textvariable=memory_min,width='5')
memory_min_entry.grid(row=4,column=1)

memory_max_label = tk.Label(frame_reward,text='Memory max: ')
memory_max_label.grid(row=4,column=2,padx='3px',sticky='e')
memory_max = tk.IntVar(frame_reward,value=100)
memory_max_entry = tk.Entry(frame_reward,textvariable=memory_max,width='5')
memory_max_entry.grid(row=4,column=3)

### Generative Model and RL settings
frame_model = tk.Frame(window)
frame_model.grid(row=2,column=1,sticky='n')
model_label = tk.Label(frame_model,text='Generative Model and Reinforcement Learning Settings')
model_label.grid(row=0,column=0,columnspan=6,sticky='n')

expert_model_label = tk.Label(frame_model,text='initial parameters: ')
expert_model_label.grid(row=1,column=0,columnspan=2,sticky='w')
expert_model_param = tk.StringVar(frame_model,value='./expert_training/expert_iter-1000000')
expert_model_entry = tk.Entry(frame_model,textvariable=expert_model_param,width='40')
expert_model_entry.grid(row=1,column=2,columnspan=4,sticky='e')

feature_dims_label = tk.Label(frame_model,text='feature dims: ')
feature_dims_label.grid(row=2,column=0,columnspan=2,sticky='w')
feature_dims = tk.IntVar(frame_model,value=128)
feature_dims_entry = tk.Entry(frame_model,textvariable=feature_dims,width='6')
feature_dims_entry.grid(row=2,column=2,sticky='w')

update_label = tk.Label(frame_model,text='update iters: ')
update_label.grid(row=2,column=4,sticky='w')
update = tk.IntVar(frame_model,value=5)
update_entry = tk.Entry(frame_model,textvariable=update,width='6')
update_entry.grid(row=2,column=5,sticky='e')

epsilon_label = tk.Label(frame_model,text='clip epsilon: ')
epsilon_label.grid(row=3,column=0,sticky='w')
epsilon = tk.DoubleVar(frame_model,value=0.1)
epsilon_entry = tk.Entry(frame_model,textvariable=epsilon,width='6')
epsilon_entry.grid(row=3,column=1,sticky='w')

gamma_label = tk.Label(frame_model,text='gamma: ')
gamma_label.grid(row=3,column=2,sticky='w')
gamma = tk.DoubleVar(frame_model,value=0.99)
gamma_entry = tk.Entry(frame_model,textvariable=gamma,width='6')
gamma_entry.grid(row=3,column=3,sticky='w')

entropy_coeff_label = tk.Label(frame_model,text='entropy coeff: ')
entropy_coeff_label.grid(row=3,column=4,sticky='w')
entropy_coeff = tk.DoubleVar(frame_model,value=0.0001)
entropy_coeff_entry = tk.Entry(frame_model,textvariable=entropy_coeff,width='6')
entropy_coeff_entry.grid(row=3,column=5,sticky='e')

batch_size_label = tk.Label(frame_model,text='batch size: ')
batch_size_label.grid(row=4,column=0,sticky='w')
batch_size = tk.IntVar(frame_model,value=256)
batch_size_entry = tk.Entry(frame_model,textvariable=batch_size,width='6')
batch_size_entry.grid(row=4,column=1,sticky='w')

train_epoch_label = tk.Label(frame_model,text='train epoch: ')
train_epoch_label.grid(row=4,column=2,sticky='w')
train_epoch = tk.IntVar(frame_model,value=1)
train_epoch_entry = tk.Entry(frame_model,textvariable=train_epoch,width='6')
train_epoch_entry.grid(row=4,column=3,sticky='w')

lr_label = tk.Label(frame_model,text='learning rate: ')
lr_label.grid(row=4,column=4,sticky='w')
lr = tk.DoubleVar(frame_model,value=0.0001)
lr_entry = tk.Entry(frame_model,textvariable=lr,width='6')
lr_entry.grid(row=4,column=5,sticky='e')

acting_round_label = tk.Label(frame_model,text='acting round: ')
acting_round_label.grid(row=5,column=0,sticky='w')
acting_round = tk.IntVar(frame_model,value=60)
acting_round_entry = tk.Entry(frame_model,textvariable=acting_round,width='6')
acting_round_entry.grid(row=5,column=1,sticky='w')

acting_batch_label = tk.Label(frame_model,text='acting batch: ')
acting_batch_label.grid(row=5,column=2,sticky='w')
acting_batch = tk.IntVar(frame_model,value=32)
acting_batch_entry = tk.Entry(frame_model,textvariable=acting_batch,width='6')
acting_batch_entry.grid(row=5,column=3,sticky='w')

time_label = tk.Label(frame_model,text='time (day): ')
time_label.grid(row=5,column=4,sticky='w')
time = tk.DoubleVar(frame_model,value=5.0)
time_entry = tk.Entry(frame_model,textvariable=time,width='6')
time_entry.grid(row=5,column=5,sticky='e')

###analysis and output control settings
frame_output = tk.Frame(window)
frame_output.grid(row=3,column=0,columnspan=2)
output_settings_label = tk.Label(frame_output,text='Analysis and Output Control Settings')
output_settings_label.grid(row=0,column=0,columnspan=6)

qed_cutoff_label = tk.Label(frame_output)

output_path_label = tk.Label(frame_model,text='output path: ')
output_path_label.grid(row=6,sticky='w')
output_path = tk.StringVar()
output_path_entry = tk.Entry(frame_model,textvariable=output_path,width='40')
output_path_entry.grid(row=6,column=1,columnspan=4,sticky='we')


def save_to_yaml():
    parameters = {
        'protein_pdb':protein_path_entry.get(),
        'ligand_sdf':ligand_path_entry.get(),
        'center_x':x_entry.get(),
        'center_y':y_entry.get(),
        'center_z':z_entry.get(),
        
        ### parameters for Molgen and MolCritic
        'initial_param':expert_model_entry.get(), 
        'feature_dims':feature_dims_entry.get(),
        'update_iters':update_entry.get(),

        ### qed 
        'qed_ratio':qed_entry.get(),

        ### scscore estimator
        'sa_upper_bound':sa_higher_entry.get(),
        'sa_lower_bound':sa_lower_entry.get(),
        'sa_ratio':sa_ratio.get(),
        
        ### activity estimator 
        'activity_upper_bound':activity_higher_entry.get(),
        'activity_lower_bound':activity_lower_entry.get(),
        'activity_ratio':activity_entry.get(),
        
        ### logger
        'logger_path':'./design_example/design.log',
        'training_log':'./design_example/training.log',

        ### memory
        'memory_min_count':memory_min_entry.get(),
        'memory_max_count':memory_max_entry.get(),
        
        ### MolEnv parameters
        'start_mol': start_mol_entry.get(),
        'allow_idx':allow_index_entry.get(),
        
        ### PPO2 training
        'batch_size':batch_size_entry.get(),
        'gamma':gamma_entry.get(),  #reward discount factor
        'clip_epsilon': epsilon_entry.get(), 
        'entropy_coeff': entropy_coeff_entry.get(),
        'training_epoch': train_epoch_entry.get(), 
        'acting_round':acting_round_entry.get(),
        'acting_batch_size':acting_batch_entry.get(),
        'learning_rate':lr_entry.get(),        
        
        'work_dir':'./design_example/',
        'tmp_states_dir':'./design_example/tmp_states_dir/',

        
        ### control
        'time_limit' : time_entry.get(), #in days
        
        ### post analysis
        'qed_cutoff':0.7, 
        'scscore_cutoff':3.5,
        'activity_cutoff':5.0,
        'out_sdf':'./design_example/desired.sdf',
        }

    if activity_estimator.get() == 0:
        parameters['use_PLANET'] = True
        parameters['use_Vina'] = False
    else:
        parameters['use_Vina'] = True
        parameters['vina_config'] = vina_config_entry.get()
        parameters['use_PLANET'] = False


save_buttom = tk.Button(frame_model,text='save',command=save_to_yaml)
save_buttom.grid(row=6,column=5,padx='2px',pady='2px',sticky='e')

window.mainloop()


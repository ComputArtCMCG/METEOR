import yaml,torch,time,os,shutil
import numpy as np
from networks import ActorCritic
from env import MoleculeEnv,MolEnvWarpper
from states import StatesDataset,StatesGenerator,RawTrajectoryProcessor,SAScoreEstimator,PlanetEstimator,ScaffoldMemory,PrepareActivityEstimate,ActivityStat
from states import ActivityStatWithDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class PPO():
    def __init__(self,rank,comm_size,**kwargs):
        self.kwargs = kwargs
        self.rank = rank
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count == 1:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:{}'.format(str(self.rank%device_count)))
        else:
            device_count = 0
            self.device = torch.device('cpu')
        self.actor_critic = ActorCritic(kwargs['feature_dims'],kwargs['update_iters'],\
                                        entropy_coeff=kwargs['entropy_coeff'],clip_epsilon=kwargs['clip_epsilon'])
        self.actor_critic.set_device(self.device)
        self.actor_critic.load_state_dict(torch.load(kwargs['initial_param'],map_location=self.device))       
        
        self.work_dir = kwargs['work_dir']
        self.tmp_states_dir = kwargs['tmp_states_dir']
        memory_dir = os.path.join(self.tmp_states_dir,'MEMORY')
        self.training_epoch = kwargs['training_epoch'] 
        self.acting_round = kwargs['acting_round']
        self.raw_traj_dir = os.path.join(self.tmp_states_dir,'RAW')
        self.raw_traj_file = os.path.join(self.raw_traj_dir,'{}_raw_traj.pkl'.format(self.rank))
        self.processed_traj_dir = os.path.join(self.tmp_states_dir,'PROCESSED')
        self.processed_traj_file = os.path.join(self.processed_traj_dir,'{}_processed_traj.pkl'.format(self.rank))
        self.activity_dir = os.path.join(self.tmp_states_dir,'ACTIVITY')
        activity_json = os.path.join(self.activity_dir,'{}_activity.json'.format(self.rank))
        self.feature_dir = os.path.join(self.tmp_states_dir,'PREP_ACT')
        self.feature_file = os.path.join(self.feature_dir,'{}_feature.pkl'.format(self.rank))
        self.other_tmp_dir = os.path.join(self.tmp_states_dir,'OTHER')
        memory_pkl = os.path.join(self.other_tmp_dir,'memory.pkl') # the file for dump scaffold memory
        if self.rank == 0:
            if os.path.exists(self.tmp_states_dir):
                shutil.rmtree(self.tmp_states_dir)    
            os.makedirs(self.tmp_states_dir)
            if not os.path.exists(memory_dir):
                os.makedirs(memory_dir)
            if not os.path.exists(self.raw_traj_dir):
                os.makedirs(self.raw_traj_dir)
            if not os.path.exists(self.processed_traj_dir):
                os.makedirs(self.processed_traj_dir)
            if not os.path.exists(self.activity_dir):
                os.makedirs(self.activity_dir)
            if not os.path.exists(self.feature_dir):
                os.makedirs(self.feature_dir)
            if not os.path.exists(self.other_tmp_dir):
                os.makedirs(self.other_tmp_dir)
            if os.path.exists(kwargs['design_log']):
                os.remove(kwargs['design_log'])
            if os.path.exists(kwargs['training_log']):
                os.remove(kwargs['training_log'])
        self.env_warpper = MolEnvWarpper([MoleculeEnv(start_mol='C',max_action=self.kwargs['max_action']) \
                                    for _ in range(self.kwargs['acting_batch_size'])],self.actor_critic,self.raw_traj_file)
        sascore_estimator = SAScoreEstimator()        
        #self.activity_prepare = PrepareActivityEstimate(self.raw_traj_file,self.feature_file,batch_size=16)
        reward_cutoff = 0.7 * (kwargs['qed_ratio'] + kwargs['sascore_ratio'] + kwargs['activity_ratio'])
        scaffold_memory = ScaffoldMemory(reward_cutoff,memory_dir)
        self.raw_trajectory_processor = RawTrajectoryProcessor(self.raw_traj_file,self.processed_traj_file,activity_json,memory_pkl,\
                                                                kwargs['design_log'],\
                                                                sascore_estimator,scaffold_memory,
                                                                qed_cutoff=(kwargs['qed_lower_bound'],kwargs['qed_upper_bound']),
                                                                sascore_cutoff=(kwargs['sascore_lower_bound'],kwargs['sascore_upper_bound']),
                                                                affinity_cutoff=(kwargs['activity_lower_bound'],kwargs['activity_upper_bound']),
                                                                reward_weights=(kwargs['qed_ratio'],kwargs['sascore_ratio'],kwargs['activity_ratio']),
                                                                property_meet_cutoff=(kwargs['qed_cutoff'],kwargs['sascore_cutoff'],kwargs['activity_cutoff']),
                                                                gamma=kwargs['gamma'])
                                                            
        self.batch_size = kwargs['batch_size'] # train batch size 
        self.training_log = kwargs['training_log']
        self.lr = kwargs['learning_rate']
        self.round,self.train_iter = 0,0
        self.train_meters = np.zeros([3])
        self.start_time = time.time()
        self.time_limit = kwargs['time_limit'] * 86400  # units in seconds

        if device_count > 0:
            if self.rank < device_count:
                # only the first several processors need to estimate activity with PLANET
                planet_estimator = PlanetEstimator()
                if kwargs['ligand_sdf'] is not None:
                    planet_estimator.set_pocket_from_ligand(kwargs['protein_pdb'],kwargs['ligand_sdf'])
                else:
                    planet_estimator.set_pocket_from_coordinate(kwargs['protein_pdb'],kwargs['center_x'],kwargs['center_y'],kwargs['center_z'])
                planet_estimator.to_cuda(self.device)
                all_raw_traj_files = [os.path.join(self.raw_traj_dir,'{}_raw_traj.pkl'.format(i)) for i in range(comm_size)]
                each_processor_count = comm_size // device_count
                raw_traj_files = all_raw_traj_files[self.rank*each_processor_count:(self.rank+1)*each_processor_count]
                #self.activity_stat = ActivityStat(self.activity_dir,feature_files,planet_estimator)
                self.activity_stat = ActivityStatWithDataset(self.activity_dir,raw_traj_files,batch_size=16,planet=planet_estimator)
            else:
                self.activity_stat = None

    def acting(self):
        self.round += 1
        if self.rank == 0:
            print('Round {} of acting is starting ...'.format(str(self.round)))
            self.raw_trajectory_processor.write_delimeter(self.round)
        if self.rank == 0:
            with torch.no_grad():
                for _ in tqdm(range(self.acting_round)):
                    self.env_warpper.acting()
        else:
            with torch.no_grad():
                for _ in range(self.acting_round):
                    self.env_warpper.acting()
        #self.activity_prepare.process_smis()
 
    def prepare_loader(self,adv_mean,adv_std):
        states_gen = StatesGenerator(self.processed_traj_dir,self.batch_size * 4)
        dataset = StatesDataset(states_gen,adv_mean,adv_std) 
        dataloader = DataLoader(dataset,num_workers=self.kwargs['njobs'],collate_fn=lambda x:x[0],prefetch_factor=1)
        return dataloader

    def training(self,dataloader):        
        print('Acting finished; now start learning ...')
        if torch.cuda.is_available():
            self.actor_critic.set_device(self.device)
            self.training_loop(dataloader)
        else:
            self.actor_critic.set_device(torch.device('cpu'))
            self.training_loop(dataloader)
        print('Parameters saved; learning finish')

    def training_loop(self,dataloader:DataLoader):
        if self.round == 1:
            self.actor_critic.requires_grad_(False)
            self.actor_critic.fatoms_critic_embedding.requires_grad_(True)
            self.actor_critic.qlayer.requires_grad_(True)
            self.actor_critic.critic_gcns.requires_grad_(True)
        else:
            self.actor_critic.requires_grad_(True)
        
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,self.actor_critic.parameters()),\
            lr=self.lr*(0.999**(self.round-1)),eps=1e-5)
        for _ in tqdm(range(self.training_epoch)):
            for (fatoms,adj_matrix,all_mask,current_mask,focus,prob_mask,action_batch,probs_batch,v_target_batch,advantage_batch) in dataloader:
                optimizer.zero_grad()
                self.actor_critic.zero_grad()
                fatoms = fatoms.to(self.actor_critic.device)
                adj_matrix = adj_matrix.to(self.actor_critic.device)
                all_mask = all_mask.to(self.actor_critic.device)
                current_mask = current_mask.to(self.actor_critic.device)
                current_mol_features = (fatoms,adj_matrix,all_mask,current_mask)
                focus = focus.to(self.actor_critic.device)
                prob_mask = prob_mask.to(self.actor_critic.device)
                action_batch = action_batch.to(self.actor_critic.device)
                probs_batch = probs_batch.to(self.actor_critic.device)
                v_target_batch = v_target_batch.to(self.actor_critic.device)
                advantage_batch = advantage_batch.to(self.actor_critic.device)
                #advantage_batch = (advantage_batch - torch.mean(advantage_batch)) / (torch.std(advantage_batch) + 1e-7)
                loss,critic_l,entropy,approxy_kl = self.actor_critic(current_mol_features,focus,prob_mask,action_batch,probs_batch,v_target_batch,advantage_batch,self.round)
                loss.backward()
                if self.round > 1:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                else:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 10.0)
                optimizer.step()
                self.write_training_log(critic_l,entropy,approxy_kl)
        torch.save(self.actor_critic.state_dict(), os.path.join(self.work_dir,'actorcritic_round{:03d}.param'.format(self.round)))

    def write_training_log(self,critic_l,entropy,approxy_kl):
        self.train_iter += 1
        self.train_meters += np.array([critic_l.item(),entropy.item(),approxy_kl.item()])
        if self.train_iter % 100 == 0 and self.train_iter > 0:
            self.train_meters /= 100
            with open(self.training_log,'a') as f:
                to_write = "[{}]\tCritic_L:{:.3f}\tEntropy:{:.3f}\tApproxy_kl:{:.3f}\n".\
                    format(self.train_iter,self.train_meters[0],self.train_meters[1],self.train_meters[2])
                f.write(to_write)
            self.train_meters = np.zeros(3)
    
    def synchronize(self):
        self.actor_critic.load_state_dict(torch.load(os.path.join(self.work_dir,'actorcritic_round{:03d}.param'.format(self.round)),map_location=torch.device('cpu')))
        self.raw_trajectory_processor.synchronize_memory()

    @staticmethod
    def write_yaml(yaml_file):
        """
        all parameters used is stored in yaml file.
        this method provide a standard yaml for molecule design.
        """
        parameters = {
            'protein_pdb':'./GBA_design/prepared_receptor.pdb',
            ### parameters for determining binding pocket if PLANET is used 
            'ligand_sdf':'./GBA_design/crystal_ligand.sdf',
            'center_x':None,
            'center_y':None,
            'center_z':None,
            
            ### parameters for Molgen and MolCritic
            'initial_param':'./expert_training_64_3/expert_iter-1000000', 
            'feature_dims':64,
            'update_iters':3,

            ### qed 
            'qed_ratio':1.0,
            'qed_upper_bound':0.8,
            'qed_lower_bound':0.2,

            ### scscore estimator
            'sascore_upper_bound':3.5,
            'sascore_lower_bound':2.0,
            'sascore_ratio':1.0,
            
            ### activity estimator
            'use_PLANET':True,
            'use_Vina':False,
            'vina_config':None,
            'vina_tmp':None,
            'activity_upper_bound':8.0,
            'activity_lower_bound':5.0,
            'activity_ratio':1.5,
            
            ### logger
            'design_log':'./GBA_design/design.log',
            'training_log':'./GBA_design/training.log',
            
            ### MolEnv parameters
            'start_mol':None,
            'max_action':50,
            
            ### PPO2 training
            'batch_size':128,
            'gamma':0.98,  #reward discount factor
            'work_dir':'./GBA_design/',
            'tmp_states_dir':'./GBA_design/tmp_states_dir/',
            'clip_epsilon': 0.1, 
            'entropy_coeff': 0.001,
            'training_epoch': 1, 
            'acting_round': 20,
            'acting_batch_size':128,
            'learning_rate':1e-4,
            
            ### control
            'time_limit' : 3.0, #in days
            'njobs' : 2,
            
            ### post analysis
            'qed_cutoff':0.6, 
            'sascore_cutoff':3.0,
            'activity_cutoff':7.0,
            'out_sdf':'./GBA_design/desired.sdf',
            }

        with open(yaml_file,'w') as f:
            yaml.dump(parameters,f)

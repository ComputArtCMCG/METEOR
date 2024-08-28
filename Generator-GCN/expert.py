import os,torch,argparse,sys
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from rdkit import Chem
from states import ExpertDataset
from networks import ActorCritic

class ExpertSolution():
    def __init__(self,work_folder=None,actor_critic=None):
        self.actor_critic = actor_critic
        self.work_folder = work_folder
        self.step = 0
        self.meters = np.zeros(1)
    
    def set_work_folder(self,work_folder):
        self.work_folder = work_folder
        os.makedirs(self.work_folder,exist_ok=True)

    def init_actor_critic(self,feature_dims,update_iters,lr):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu') #test
        self.actor_critic = ActorCritic(feature_dims,update_iters,self.device).to(self.device)
        for param in self.actor_critic.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),lr=lr)
    
    def training_loop(self,dataloader:DataLoader):
        for mol_features,expert_actions in dataloader:
            #try:
                self.optimizer.zero_grad()
                fatoms,adj_matrix,all_mask,current_mask = mol_features
                fatoms = fatoms.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                all_mask = all_mask.to(self.device)
                current_mask = current_mask.to(self.device)

                mol_features = (fatoms,adj_matrix,all_mask,current_mask)
                fatoms_input,super_input = self.actor_critic.actor_feature_update(*mol_features)
                loss = self.actor_critic.expert_training(fatoms_input,super_input,all_mask,expert_actions)
                loss.backward()
                self.logger_step(loss)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 10.0)
                self.optimizer.step()
                self.save_actor_critic()
                self.step += 1
            #except Exception as e:
                #print(e)
            #    continue

    def logger_step(self,loss):
        self.meters += np.array([loss.item()])
        if self.step > 0 and self.step % 250 == 0:
            self.meters /= 250
            print("[{}]\texpert_Loss:{:.3f}".format(self.step,self.meters[0]))
            sys.stdout.flush()
            self.meters *= 0.

    def save_actor_critic(self):
        if self.step > 0 and self.step % 10000 == 0:
            torch.save(self.actor_critic.state_dict(), self.work_folder + "/expert_iter-" + str(self.step))
        else:
            pass

    def expert_training(self,yaml_file):
        parameters = ExpertSolution.read_yaml(yaml_file)
        self.set_work_folder(parameters['work_folder'])
        self.init_actor_critic(parameters['feature_dims'],parameters['update_iters'],parameters['learning_rate'])
        for _ in range(parameters['epoch']):
            expert_dataset = ExpertDataset(parameters['expert_smiles'],parameters['batch_size'])
            dataloader = DataLoader(expert_dataset,batch_size=1,shuffle=False,num_workers=4,collate_fn=lambda x:x[0],pin_memory=True)
            self.training_loop(dataloader)

    @staticmethod
    def read_yaml(yaml_file):
        with open(yaml_file,'r') as f:
            parameters = yaml.load(f,Loader=yaml.FullLoader)
        return parameters
    
    @staticmethod
    def write_sample_yaml(yaml_file):
        parameters = {
            'work_folder':'./expert_training/',
            'expert_smiles': './dataset/zinc/zinc_new.smi',
            'batch_size':32,
            'feature_dims':128,
            'update_iters':5,
            'epoch':150,
            'learning_rate':1e-4,
        }
        with open(yaml_file,'w') as f:
            yaml.dump(parameters,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',required=True,default='train',choices=['train','yaml'],\
        help='train for expert training, yaml for generating default settings')
    parser.add_argument('-y','--yaml',required=True)
    args = parser.parse_args()

    if args.mode == 'yaml':
        ExpertSolution.write_sample_yaml(args.yaml)
    elif args.mode == 'train':
        solution = ExpertSolution()
        solution.expert_training(args.yaml)

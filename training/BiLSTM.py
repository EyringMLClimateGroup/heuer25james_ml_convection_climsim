import torch
from torch import nn, optim
from pytorch_lightning import LightningModule

class FFNN_LSTM_6_AVG(nn.Module):
    def __init__(self, feature_target_lengths, zeroout_index, nlev, argv1=0, argv2=0):
        super(FFNN_LSTM_6_AVG, self).__init__()
        self.nlev = nlev

        if argv1 == 5:
            # 13.6M model
            self.encode_dim = 300
            self.hidden_dim = 280
            self.iter_dim = 800
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,6,batch_first=True,dropout=0.01,bidirectional=True)
            # "learning_rate": 0.001,
            # "scheduler": "reduce_plat",
            # "weight_decay": 0.0002
        
        # if argv1 == 2:
        #     # 11.4M model
        #     self.encode_dim = 280
        #     self.hidden_dim = 330
        #     self.iter_dim = 570
        #     self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,4,batch_first=True,dropout=0.25,bidirectional=True)
        #     #---
        #     # "learning_rate": 0.00065,
        #     # "scheduler": "reduce_plat",
        #     # "weight_decay": 0.01

        # if argv1 == 1:
        if argv1 == 6:
            # 8M model
            self.encode_dim = 300
            self.hidden_dim = 280
            self.iter_dim = 800
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,3,batch_first=True,dropout=0.01,bidirectional=True)
        
        # # 1.2M model
        # self.encode_dim = 200
        # self.hidden_dim = 100
        # self.iter_dim = 300
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,3,batch_first=True,dropout=0.01,bidirectional=True)
        
        # # 0.5M model
        # self.encode_dim = 200
        # self.hidden_dim = 50
        # self.iter_dim = 190
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,5,batch_first=True,dropout=0.01,bidirectional=True)
        
        # if argv1 == 2:
        #     # 3.3M model
        #     self.encode_dim = 200
        #     self.hidden_dim = 100
        #     self.iter_dim = 1000
        #     self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,2,batch_first=True,dropout=0.01,bidirectional=True)
        
        # if argv1 == 2:
        #     # 3.3M model - 2
        #     self.encode_dim = 150
        #     self.hidden_dim = 150
        #     self.iter_dim = 600
        #     self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,4,batch_first=True,dropout=0.01,bidirectional=True)
        
        # # 5.5M model
        # self.encode_dim = 150
        # self.hidden_dim = 150
        # self.iter_dim = 600
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,4,batch_first=True,dropout=0.01,bidirectional=True)
        
#         # 6M model
#         self.encode_dim = 300
#         self.hidden_dim = 350
#         self.iter_dim = 400
#         self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,2,batch_first=True,dropout=0.01,bidirectional=True)
        
#         # 7.1M model
#         self.encode_dim = 200
#         self.hidden_dim = 350
#         self.iter_dim = 700
#         self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,2,batch_first=True,dropout=0.01,bidirectional=True)
        
#         # 7M model
#         self.encode_dim = 350
#         self.hidden_dim = 300
#         self.iter_dim = 400
#         self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,3,batch_first=True,dropout=0.01,bidirectional=True)

        #   # 0.24M model
        # self.encode_dim = 20
        # self.hidden_dim = 60
        # self.iter_dim = 160
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,2,batch_first=True,dropout=0.01,bidirectional=True)
        
        if argv1 == 7:
            # 0.88M model
            self.encode_dim = 80
            self.hidden_dim = 50
            self.iter_dim = 500
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,3,batch_first=True,dropout=0.07,bidirectional=True)
            # "batch_size": 256,
            # "learning_rate": 0.001,
            # "scheduler": "None",
            # "weight_decay": 0.0002
        
        if argv1 == 2:
            # 0.98M model
            self.encode_dim = 160
            self.hidden_dim = 90
            self.iter_dim = 190
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,4,batch_first=True,dropout=0.13,bidirectional=True)
            # "batch_size": 256,
            # "learning_rate": 0.001,
            # "scheduler": "reduce_plat",
            # "weight_decay": 0.01
        
        if argv1 == 3:
            # 1.6M model
            self.encode_dim = 380
            self.hidden_dim = 110
            self.iter_dim = 330
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,3,batch_first=True,dropout=0.02,bidirectional=True)
        #---
        # "learning_rate": 0.001,
        # "scheduler": "reduce_plat",
        # "weight_decay": 0.0002
        # "batch_size": 256,
        
        # 5.1 Model
        if argv1 == 4:
            self.encode_dim = 320
            self.hidden_dim = 130
            self.iter_dim = 890
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,6,batch_first=True,dropout=0.01,bidirectional=True)
            # "batch_size": 256,
            # "learning_rate": 0.001,
            # "scheduler": "reduce_plat",
            # "weight_decay": 0.01
        
        if argv1 == 1:
            # 0.54 Model
            self.encode_dim = 280
            self.hidden_dim = 60
            self.iter_dim = 120
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,4,batch_first=True,dropout=0.02,bidirectional=True)
        # "batch_size": 256,
        # "learning_rate": 0.001,
        # "scheduler": "None",
        # "weight_decay": 0.01
        
        # # 0.98 Model
        # self.encode_dim = 160
        # self.hidden_dim = 90
        # self.iter_dim = 190
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,4,batch_first=True,dropout=0.13,bidirectional=True)
        # # "learning_rate": 0.001,
        # # "scheduler": "reduce_plat",
        # # "weight_decay": 0.01
        
        # # 1.58 Model
        # self.encode_dim = 90
        # self.hidden_dim = 130
        # self.iter_dim = 340
        # self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,3,batch_first=True,dropout=0.05,bidirectional=True)
        # # "learning_rate": 0.00065,
        # # "scheduler": "reduce_plat",
        # # "weight_decay": 0.0002

        if argv1 == 0:
            # 0.17 M
            self.encode_dim = 20
            self.hidden_dim = 40
            self.iter_dim = 210
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,1,batch_first=True,dropout=0.02,bidirectional=True)
        # "learning_rate": 0.005,
        # "scheduler": "cosanh",
        # "weight_decay": 0.01
        # "batch_size": 256,
        
        if argv1 == 8:
            # 0.99 M
            self.encode_dim = 60
            self.hidden_dim = 80
            self.iter_dim = 260
            self.LSTM_1 = nn.LSTM(self.encode_dim,self.hidden_dim,5,batch_first=True,dropout=0.03,bidirectional=True)
        # "batch_size": 256,
        # "learning_rate": 0.001,
        # "scheduler": "reduce_plat",
        # "weight_decay": 0.0002

        self.n_column_features, self.n_scalar_features, self.n_column_targets, self.n_scalar_targets = \
            feature_target_lengths
        self.zeroout_index = zeroout_index
        self.input_size = self.n_column_features*self.nlev + self.n_scalar_features
        self.output_size = self.n_column_targets*self.nlev + self.n_scalar_targets
        
        self.Linear_1 = nn.Linear(self.n_column_features+self.n_scalar_features, self.encode_dim)
        self.Linear_2 = nn.Linear(6*self.hidden_dim+self.encode_dim, self.iter_dim)
        self.Linear_3 = nn.Linear(self.iter_dim, self.n_column_targets)
        self.Linear_4_0 = nn.Linear(self.iter_dim, self.iter_dim*2)

        self.Linear_4 = nn.Linear(self.iter_dim*2, self.n_scalar_targets)
        
        self.weight = nn.Parameter(torch.zeros(1,self.output_size))
        self.bias = nn.Parameter(torch.zeros(1,self.output_size))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.bias)
        # self.bias = nn.Linear(len(seq_y_list)*self.nlev+len(num_y_list),1)
        # self.weight = nn.Linear(len(seq_y_list)*self.nlev+len(num_y_list),1)
        
        self.avg_pool_1 = nn.AvgPool1d(kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
#         x_seq = x[:,0:NLEV*len(seq_fea_list)]
#         x_seq = x_seq.reshape((-1,len(seq_fea_list),NLEV))
#         x_seq = torch.transpose(x_seq, 1, 2)
        
#         x_num = x[:,self.nlev*len(seq_fea_list):x.shape[1]]
#         x_num_repeat = x_num.reshape((-1,1,len(num_fea_list)))
#         x_num_repeat = x_num_repeat.repeat((1,self.nlev,1))
        # x[]
        
        # x_seq = F.elu(self.Linear_1(torch.concat((x_seq,x_num_repeat),dim=-1)/5))
        # dims: b->batch, v->variable, h->height, e->encode
        # print(x.shape) # (b,h,v)
        x_seq = F.elu(self.Linear_1(x)) # (b,h,e)
        
        # self.LSTM_1.flatten_parameters()
        x_seq_1,_ = self.LSTM_1(x_seq/5)
        
        x_seq_1_mean = torch.mean(x_seq_1,dim=1,keepdim=True)
        x_seq_1_mean = x_seq_1_mean.repeat((1,self.nlev,1))

        x_seq_1_avg_pool = self.avg_pool_1(torch.transpose(x_seq_1, 1, 2))
        x_seq_1_avg_pool = torch.transpose(x_seq_1_avg_pool,1, 2)
        
        x_seq_1 = F.elu(self.Linear_2(torch.cat((x_seq_1,x_seq_1_mean,x_seq,x_seq_1_avg_pool),dim=-1)/5))
        
        x_seq_out = self.Linear_3(x_seq_1)
        x_seq_out = torch.transpose(x_seq_out, 1, 2)
        x_seq_out = x_seq_out.reshape((-1,self.nlev*self.n_column_targets)) # (b,seq_out)
        
        x_num_out = F.elu(self.Linear_4_0(torch.mean(x_seq_1,dim=1)))
        x_num_out = self.Linear_4(x_num_out) # (b,num_out)

        # print(x_seq_out.shape, x_num_out.shape)
        # output = self.weight.weight*(torch.concat((x_seq_out,x_num_out),dim=-1))/3+self.bias.weight/3
        output = self.weight*(torch.concat((x_seq_out,x_num_out),dim=-1))/3+self.bias/3
        # output = torch.concat((x_seq_out,x_num_out),dim=-1)

        if len(self.zeroout_index) > 0:
            output[:,self.zeroout_index] = output[:,self.zeroout_index]*0.0
        
        return output

class HighResLeapModelPhysicsConstraint(LightningModule):
    def __init__(self, feature_target_lengths, zeroout_index, norm_constants, in_indices, out_indices, nlev=60, ncorr_lvls=25, vertical_weighting='learned', use_confidence=False, cons_energy=False, cons_mass=False, cons_uv=False, pinn=False, pinn_weight=None, dP_scaled=True, noise_scheduler=None, argv1=0, argv2=0):
        super().__init__()
        self.nlev = nlev
        self.ncorr_lvls = ncorr_lvls
        # indices
        self.p_start_idx, self.p_tm1_start_idx = in_indices
        self.t_phy_start, self.qv_phy_start, self.qc_phy_start, self.qi_phy_start, self.u_phy_start, self.v_phy_start, \
            self.precc_start, self.precsc_start = out_indices
        self.t_phy_end, self.qv_phy_end, self.qc_phy_end, self.qi_phy_end, self.u_phy_end, self.v_phy_end = \
            self.t_phy_start + self.nlev, self.qv_phy_start + self.nlev, self.qc_phy_start + self.nlev, \
            self.qi_phy_start + self.nlev, self.u_phy_start + self.nlev, self.v_phy_start + self.nlev
        self.t_phy_start_c = self.t_phy_start + self.nlev - self.ncorr_lvls
        self.qv_phy_start_c = self.qv_phy_start + self.nlev - self.ncorr_lvls
        self.qc_phy_start_c = self.qc_phy_start + self.nlev - self.ncorr_lvls
        self.qi_phy_start_c = self.qi_phy_start + self.nlev - self.ncorr_lvls
        self.u_phy_start_c = self.u_phy_start + self.nlev - self.ncorr_lvls
        self.v_phy_start_c = self.v_phy_start + self.nlev - self.ncorr_lvls
        
        self.n_column_features, self.n_scalar_features, self.n_column_targets, self.n_scalar_targets = \
            feature_target_lengths
        self.input_size = self.n_column_features*self.nlev + self.n_scalar_features
        self.output_size = self.n_column_targets*self.nlev + self.n_scalar_targets
        self.zeroout_index = zeroout_index
        self.zeroout_mask = torch.zeros(self.n_column_targets*self.nlev+self.n_scalar_targets, dtype=bool)
        if len(zeroout_index) > 0:
            self.zeroout_mask[zeroout_index] = 1
        # self.lrs = []
        self.dP_scaled = dP_scaled
        if (not pinn and pinn_weight is not None) or (pinn and pinn_weight is None):
            raise Exception("If pinn mode is activated also provide a weight, if not do not")
        if pinn_weight is not None:
            if pinn_weight < 0 or pinn_weight > 1:
                raise Exception("pinn_weight must be btw. 0 and 1")
        self.pinn = pinn
        self.pinn_weight = pinn_weight
        # self.EPS = 1e-10
        
        # normalization constants
        self.p_sub = nn.Parameter(torch.tensor(norm_constants[0][:-1], dtype=torch.float32), requires_grad=False)
        self.p_div = nn.Parameter(torch.tensor(norm_constants[1][:-1], dtype=torch.float32), requires_grad=False)
        self.ps_sub = nn.Parameter(torch.tensor(norm_constants[0][-1], dtype=torch.float32), requires_grad=False)
        self.ps_div = nn.Parameter(torch.tensor(norm_constants[1][-1], dtype=torch.float32), requires_grad=False)
        # self.t_sub = nn.Parameter(torch.tensor(norm_constants[2], dtype=torch.float32), requires_grad=False)
        # self.t_div = nn.Parameter(torch.tensor(norm_constants[3], dtype=torch.float32), requires_grad=False)
        # self.qn_sub = nn.Parameter(torch.tensor(norm_constants[4], dtype=torch.float32), requires_grad=False)
        # self.qn_div = nn.Parameter(torch.tensor(norm_constants[5], dtype=torch.float32), requires_grad=False)
        # self.lbd_qn = nn.Parameter(torch.tensor(norm_constants[6], dtype=torch.float32), requires_grad=False)
        self.wind_scale = 10  # 10 m/s characteristic velocity scale
        self.output_scale = nn.Parameter(torch.tensor(norm_constants[2], dtype=torch.float32), requires_grad=False)
        self.tracer_weights = nn.Parameter(torch.ones(3))
        # self.tracer_weights = nn.Parameter(torch.tensor([1,0,0], dtype=torch.float32), requires_grad=False)
        self.vertical_weighting = vertical_weighting
        if self.vertical_weighting == 'learned':
            self.t_weights = nn.Parameter(torch.ones(self.ncorr_lvls))
            self.qv_weights = nn.Parameter(torch.ones(self.ncorr_lvls))
            self.qc_weights = nn.Parameter(torch.ones(self.ncorr_lvls))
            self.qi_weights = nn.Parameter(torch.ones(self.ncorr_lvls))
            self.u_weights = nn.Parameter(torch.ones(self.ncorr_lvls))
            self.v_weights = nn.Parameter(torch.ones(self.ncorr_lvls))
            torch.nn.init.uniform_(self.t_weights)
            torch.nn.init.uniform_(self.qv_weights)
            torch.nn.init.uniform_(self.qc_weights)
            torch.nn.init.uniform_(self.qi_weights)
            torch.nn.init.uniform_(self.u_weights)
            torch.nn.init.uniform_(self.v_weights)
        elif self.vertical_weighting == 'none':
            # --- dummy for tracing --- #
            self.t_weights = torch.tensor(0.)
            self.qv_weights = torch.tensor(0.)
            self.qc_weights = torch.tensor(0.)
            self.qi_weights = torch.tensor(0.)
            self.u_weights = torch.tensor(0.)
            self.v_weights = torch.tensor(0.)
            # --- --- #
        else:
            raise Exception(f'vertical_weighting mode: {self.vertical_weighting} not supported')
        # torch.nn.init.uniform_(self.tracer_weights)
        # physical constants
        self.g0 = 9.80665 # m s-2
        self.t0 = 10 # s
        self.rho0 = 1e3 # kg m-3
        self.cpd0 = 1004.64 # J K-1 kg-1
        self.l0 = self.g0*self.t0**2 # m
        self.e0 = self.l0**2/self.t0**2 # J kg-1 = m2 s-2
        self.T0 = self.e0/self.cpd0
        self.p0 = self.rho0*self.g0*self.l0
        self.alv0 = 2.5008e6/self.e0
        self.als0 = 2.8345e6/self.e0
        self.layer_heating = self.e0*self.rho0*self.l0/self.t0
        print(f'Physical Scales:\ng: {self.g0} m\nTime: {self.t0} s\nrho: {self.rho0} K\ncpd {self.cpd0} J/kg/K\n')
        print(f'Derived Physical Scales:\nLength: {self.l0} m\nEnergy: {self.e0} J/kg\nTemperature: {self.T0} K\n'+\
            f'Pressure {self.p0} Pa')
        print(f'Normalized Constants:\nalv: {self.alv0}\nals: {self.als0}')
        self.phys_output_scale = nn.Parameter(torch.ones_like(self.output_scale), requires_grad=False)
        self.phys_output_scale *= self.t0
        self.phys_output_scale[self.t_phy_start:self.t_phy_end] = self.phys_output_scale[self.t_phy_start:self.t_phy_end] / self.T0
        self.phys_output_scale[self.u_phy_start:self.v_phy_end] = self.phys_output_scale[self.u_phy_start:self.v_phy_end] * self.t0 / self.l0
        self.phys_output_scale[self.precc_start] = self.phys_output_scale[self.precc_start] / self.l0
        self.phys_output_scale[self.precsc_start] = self.phys_output_scale[self.precsc_start] / self.l0
        # self.alv = 2.5008e6 #J/kg
        # self.als = 2.8345e6 #J/kg
        # self.cpd = 1004.64 # J K-1 kg-1
        # self.grav = nn.Parameter(torch.tensor(9.80665/self.g0, dtype=torch.float32), requires_grad=False) # unnormed: m s-2
        # self.rho_h2o = nn.Parameter(torch.tensor(1e3/self.rho0, dtype=torch.float32), requires_grad=False) # unnormed: kg m-3
        # self.alv = nn.Parameter(torch.tensor(2.5008e6/self.e0, dtype=torch.float32), requires_grad=False) # unnormed: J/kg
        # self.als = nn.Parameter(torch.tensor(2.8345e6/self.e0, dtype=torch.float32), requires_grad=False) # unnormed: J/kg
        # self.cpd = nn.Parameter(torch.tensor(1004.64/self.cpd0, dtype=torch.float32), requires_grad=False) # unnormed: J K-1 kg-1
        # self.timestep = nn.Parameter(torch.tensor(1200., dtype=torch.float32), requires_grad=False) # s (climsim timestep)
        
        # self.sel_idx = list(range(34))
        # # dont use idx for LHFLX, SHFLX, TAU, albedos, tm_LHFLX, tm_SHFLX
        # self.sel_idx.pop(30)
        # self.sel_idx.pop(29)
        # self.sel_idx.pop(27)
        # self.sel_idx.pop(26)
        # self.sel_idx.pop(25)
        # self.sel_idx.pop(24)
        # self.sel_idx.pop(22)
        # self.sel_idx.pop(21)
        # self.sel_idx.pop(20)
        # self.sel_idx.pop(19)
        
        # which constraint to obey
        self.energy_constraint = cons_energy
        self.mass_constraint = cons_mass
        self.uv_constraint = cons_uv

        self.use_confidence = use_confidence
        # For confidence loss:
        if self.use_confidence:
            feature_target_lengths = \
                self.n_column_features-1, self.n_scalar_features-1, 2*self.n_column_targets, 2*self.n_scalar_targets
            # self.n_column_targets *= 2
            # self.n_scalar_targets *= 2

        self.network = FFNN_LSTM_6_AVG(feature_target_lengths, zeroout_index, nlev, argv1, argv2)
        # self.ema = ExponentialMovingAverage(self.network.parameters(), decay=0.99)
        self.noise_scheduler = noise_scheduler
        
        if self.use_confidence:
            self.criterion = nn.HuberLoss(delta=1., reduction='none')
            # self.mae_loss = nn.L1Loss(reduction='none')
        else:
            self.criterion = nn.HuberLoss(delta=1.)
    
    def calc_uv_residual(self, dP, out):
        u_phy = out[:,self.u_phy_start_c:self.u_phy_end]
        v_phy = out[:,self.v_phy_start_c:self.v_phy_end]
        # u_residual = torch.sum(u_phy * dP / self.grav, dim=-1)
        # v_residual = torch.sum(v_phy * dP / self.grav, dim=-1)
        u_residual = torch.sum(u_phy * dP, dim=-1)
        v_residual = torch.sum(v_phy * dP, dim=-1)

        return u_residual, v_residual


    def calc_mass_residual(self, dP, out):
        qv_phy = out[:,self.qv_phy_start_c:self.qv_phy_end]
        qc_phy = out[:,self.qc_phy_start_c:self.qc_phy_end]
        qi_phy = out[:,self.qi_phy_start_c:self.qi_phy_end]
        rain = out[:,self.precc_start]
        snow = out[:,self.precsc_start]
        # mass_residual = (torch.sum((qv_phy + qc_phy + qi_phy) * dP / self.grav, dim=-1) + rain*self.rho_h2o + snow*self.rho_h2o)
        mass_residual = (torch.sum((qv_phy + qc_phy + qi_phy) * dP, dim=-1) + rain + snow)

        return mass_residual


    def calc_energy_residual(self, dP, out):
        t_phy = out[:,self.t_phy_start_c:self.t_phy_end]
        qc_phy = out[:,self.qc_phy_start_c:self.qc_phy_end]
        qi_phy = out[:,self.qi_phy_start_c:self.qi_phy_end]
        rain = out[:,self.precc_start]
        snow = out[:,self.precsc_start]
        # energy_residual = (torch.sum((t_phy*self.cpd - qc_phy*self.alv - qi_phy*self.als) * dP / self.grav, dim=-1) - \
        #                   rain * self.rho_h2o * self.alv - \
        #                   snow * self.rho_h2o * self.als)
        energy_residual = (torch.sum((t_phy - qc_phy*self.alv0 - qi_phy*self.als0) * dP, dim=-1) - rain*self.alv0 - snow*self.als0)
        
        return energy_residual


    def calc_uv_correction(self, dP, u_res, v_res):
        if self.vertical_weighting == 'learned':
            u_weights_relu = F.relu(self.u_weights)
            u_weights_relu = u_weights_relu / torch.sum(u_weights_relu)# * self.ncorr_lvls
            v_weights_relu = F.relu(self.v_weights)
            v_weights_relu = v_weights_relu / torch.sum(v_weights_relu)# * self.ncorr_lvls
        else:
            u_weights_relu = torch.tensor(1 / self.ncorr_lvls)
            v_weights_relu = torch.tensor(1 / self.ncorr_lvls)
        # Add dimension so that it is broadcastable over height dimension

        if not self.dP_scaled:
            dP = torch.sum(dP, dim=-1)[:,None] / self.ncorr_lvls
        # print(dP_sum/self.ncorr_lvls)
        # dP_sum = dP_sum[:,None]
                         
        # u_cor = u_res[:,None] / dP * self.grav * u_weights_relu
        # v_cor = v_res[:,None] / dP * self.grav * v_weights_relu
        u_cor = u_res[:,None] / dP * u_weights_relu
        v_cor = v_res[:,None] / dP * v_weights_relu
        
        # # dP = dP * dP_sum / torch.sum(dP, dim=-1)
        # dP_sum = torch.sum(dP, dim=-1)
        # u_cor = u_res[:,None] * self.output_scale[self.u_phy_start_c:self.u_phy_end] / dP * self.grav # * u_weights_relu
        # v_cor = v_res[:,None] * self.output_scale[self.v_phy_start_c:self.v_phy_end] / dP * self.grav # * v_weights_relu
        
        return u_cor, v_cor
    
    def calc_q_correction(self, dP, mass_res):
        if self.vertical_weighting == 'learned':
            qv_weights_relu = F.relu(self.qv_weights)
            qv_weights_relu = qv_weights_relu / torch.sum(qv_weights_relu)# * self.ncorr_lvls
            qc_weights_relu = F.relu(self.qc_weights)
            qc_weights_relu = qc_weights_relu / torch.sum(qc_weights_relu)# * self.ncorr_lvls
            qi_weights_relu = F.relu(self.qi_weights)
            qi_weights_relu = qi_weights_relu / torch.sum(qi_weights_relu)# * self.ncorr_lvls
        else:
            qv_weights_relu = torch.tensor(1 / self.ncorr_lvls)
            qc_weights_relu = torch.tensor(1 / self.ncorr_lvls)
            qi_weights_relu = torch.tensor(1 / self.ncorr_lvls)
        # Add dimension so that it is broadcastable over height dimension
        if not self.dP_scaled:
            dP = torch.sum(dP, dim=-1)[:,None] / self.ncorr_lvls
        # mass_res = mass_res[:,None] / dP * self.grav
        mass_res = mass_res[:,None] / dP
        
        tracer_weights_relu = F.relu(self.tracer_weights)
        tracer_weights_relu = tracer_weights_relu/torch.sum(tracer_weights_relu)
        
        qv_cor = mass_res * tracer_weights_relu[0] * qv_weights_relu
        qc_cor = mass_res * tracer_weights_relu[1] * qc_weights_relu
        qi_cor = mass_res * tracer_weights_relu[2] * qi_weights_relu
        
        return qv_cor, qc_cor, qi_cor
    
    def calc_temperature_correction(self, dP, energy_res):
        if self.vertical_weighting == 'learned':
            t_weights_relu = F.relu(self.t_weights)
            t_weights_relu = t_weights_relu / torch.sum(t_weights_relu)# * self.ncorr_lvls
        else:
            t_weights_relu = torch.tensor(1 / self.ncorr_lvls)

        if not self.dP_scaled:
            dP = torch.sum(dP, dim=-1)[:,None] / self.ncorr_lvls
        # dP_sum = dP_sum[:,None]
        # Add dimension so that it is broadcastable over height dimension
        # t_cor = energy_res[:,None] / self.cpd / dP * self.grav * t_weights_relu
        t_cor = energy_res[:,None] / dP * t_weights_relu
        
        return t_cor

    
    def reconstruct_pfull(self, dP, ps):
        phalf = torch.concatenate([-dP, ps[:,None]], dim=-1)
        phalf = torch.flip(torch.cumsum(torch.flip(phalf, [1]), dim=-1), [1]) # [:,::-1] not possible
        pfull = (phalf[:,1:] + phalf[:,:-1]) / 2
        
        return pfull

    
    def reconstruct_dp(self, pmid, ps):
        phalf = torch.empty((pmid.shape[0], pmid.shape[1]+1), dtype=pmid.dtype, layout=pmid.layout, device=pmid.device)
        phalf[:,-1] = ps
        for i in range(pmid.shape[1]-1,-1,-1):
            phalf[:,i] = 2*pmid[:,i] - phalf[:,i+1]
        
        dp = phalf[:,1:] - phalf[:,:-1]
        
        return dp
    
    
    def apply_temperature_rules(self, T):
        # Create an output tensor, initialized to zero
        output = torch.zeros_like(T)

        # Apply the linear transition within the range 253.16 to 273.16
        mask = (T >= 253.16) & (T <= 273.16)
        output[mask] = (T[mask] - 253.16) / 20.0  # 20.0 is the range (273.16 - 253.16)

        # Values where T > 273.16 set to 1
        output[T > 273.16] = 1

        # Values where T < 253.16 are already set to 0 by the initialization
        return output


    def forward(self, x):
        # --- add pfull --- #
        # dP = x[:,:,self.dP_start_idx] * self.dP_div + self.dP_sub
        # ps = x[:,0,self.dP_start_idx+1] * self.ps_div + self.ps_sub
        # pfull = self.reconstruct_pfull(dP, ps) # todo append pfull here?
        # pfull = (pfull - self.pfull_sub) / self.pfull_div
        # x = torch.concat([x, pfull[:,:,None]], dim=-1)
        # --- discard dp --- #
        # x = torch.concat([x[:,:,:self.dP_start_idx], x[:,:,self.dP_start_idx+1:]], dim=2)
        # if self.energy_constraint:
        #     # --- get t,qn before --- #
        #     t_before = x[:,:,self.t_start_idx].clone() * self.t_div + self.t_sub
        #     qn_before = x[:,:,self.qn_start_idx].clone() * self.qn_div + self.qn_sub
        #     qn_before = -torch.log(1 - qn_before+self.EPS) / self.lbd_qn
        #     liq_frac = self.apply_temperature_rules(t_before)
        #     qc_before = liq_frac * qn_before
        #     qi_before = (1-liq_frac) * qn_before
        #     # --- end get t,qn before --- #
        
        # phalf = x[:,:,self.p_start_idx] * self.p_div + self.p_sub
        # ps = x[:,0,self.p_start_idx+1] 
        # dP = torch.empty_like(phalf)
        # dP[:,:-1] = phalf[:,1:] - phalf[:,:-1]
        # dP[:,-1] = ps - phalf[:,-1]
        
        # x[:,:,:self.n_column_features] = 0
        
        pmid = x[:,:,self.p_start_idx] * self.p_div + self.p_sub
        ps = x[:,0,self.p_start_idx+1] * self.ps_div + self.ps_sub
        dP = self.reconstruct_dp(pmid, ps) / self.p0
        
        # dP = x[:,:,self.p_start_idx] * self.p_div + self.p_sub
        
        # x = x[:,:,:self.p_start_idx+1] # get rid of ps
        x = x[:,:,:self.p_start_idx] # get rid of p and ps
        # x = torch.cat([x[:,:,:self.p_start_idx],
        #                x[:,:,self.p_start_idx+2:]], dim=2) # get rid of p and ps
        # x = torch.cat([x[:,:,:self.p_start_idx],
        #                x[:,:,self.p_start_idx+2:self.p_tm1_start_idx],
        #                x[:,:,self.p_tm1_start_idx+1:]], dim=2) # get rid of p and ps and ps (t-1)
        
        output = self.network(x)
        output[...,self.output_size-1] = F.relu(output[...,self.output_size-1])
        output[...,self.output_size-2] = F.relu(output[...,self.output_size-2])
        
        # out_unnormed = output[:,:self.output_size] / self.output_scale
        out_phy_normed = output[:,:self.output_size] / self.output_scale * self.phys_output_scale
        
        dP_corr = dP[:,self.nlev-self.ncorr_lvls:]
        # dp_corr = dp_corr / torch.sum(dp_corr, dim=-1)[:,none] * dp_sum[:,none]
        
        if self.energy_constraint or self.mass_constraint or self.uv_constraint:
            # output = self.network(x)
            u_res, v_res = self.calc_uv_residual(dP_corr, out_phy_normed)
            mass_res = self.calc_mass_residual(dP_corr, out_phy_normed)
            # energy_res = self.calc_energy_residual(dP, out_phy_normed)
            # dP_sum = torch.sum(dP[:,self.nlev-self.ncorr_lvls:], dim=-1)
            u_cor, v_cor = self.calc_uv_correction(dP_corr, u_res, v_res)
            qv_cor, qc_cor, qi_cor = self.calc_q_correction(dP_corr, mass_res)
            
            u_cor = u_cor * self.output_scale[self.u_phy_start_c:self.u_phy_end] / \
                self.phys_output_scale[self.u_phy_start_c:self.u_phy_end]
            v_cor = v_cor * self.output_scale[self.v_phy_start_c:self.v_phy_end] / \
                self.phys_output_scale[self.v_phy_start_c:self.v_phy_end]
            qv_cor = qv_cor * self.output_scale[self.qv_phy_start_c:self.qv_phy_end] / \
                self.phys_output_scale[self.qv_phy_start_c:self.qv_phy_end]
            qc_cor = qc_cor * self.output_scale[self.qc_phy_start_c:self.qc_phy_end] / \
                self.phys_output_scale[self.qc_phy_start_c:self.qc_phy_end]
            qi_cor = qi_cor * self.output_scale[self.qi_phy_start_c:self.qi_phy_end] / \
                self.phys_output_scale[self.qi_phy_start_c:self.qi_phy_end]
            # t_cor = self.calc_temperature_correction(dP_corr, energy_res)
            tot_res = torch.zeros_like(output[...,0])
        else:
            u_res, v_res = self.calc_uv_residual(dP_corr, out_phy_normed)
            mass_res = self.calc_mass_residual(dP_corr, out_phy_normed)
            energy_res = self.calc_energy_residual(dP_corr, out_phy_normed)
            # --- dummy for tracing --- #
            t_cor = torch.tensor(0.)
            qv_cor = torch.tensor(0.)
            qc_cor = torch.tensor(0.)
            qi_cor = torch.tensor(0.)
            u_cor = torch.tensor(0.)
            v_cor = torch.tensor(0.)
            # --- --- #
            # u_res = torch.abs(torch.nan_to_num(u_res))
            # v_res = torch.abs(torch.nan_to_num(v_res))
            # mass_res = torch.abs(torch.nan_to_num(mass_res))
            # energy_res = torch.abs(torch.nan_to_num(energy_res))
            
            # tot_res = (u_res+v_res)*self.wind_scale + \
            #               mass_res*self.alv + \
            #               energy_res
            tot_res = (u_res+v_res) + \
                          mass_res + \
                          energy_res
        
        if self.uv_constraint:
            tot_res = tot_res + u_res+v_res

            output[:,self.u_phy_start_c:self.u_phy_end] = output[:,self.u_phy_start_c:self.u_phy_end] - u_cor
            output[:,self.v_phy_start_c:self.v_phy_end] = output[:,self.v_phy_start_c:self.v_phy_end] - v_cor

        if self.mass_constraint:
            tot_res = tot_res + mass_res

            output[:,self.qv_phy_start_c:self.qv_phy_end] = output[:,self.qv_phy_start_c:self.qv_phy_end] - qv_cor
            output[:,self.qc_phy_start_c:self.qc_phy_end] = output[:,self.qc_phy_start_c:self.qc_phy_end] - qc_cor
            output[:,self.qi_phy_start_c:self.qi_phy_end] = output[:,self.qi_phy_start_c:self.qi_phy_end] - qi_cor

        if self.energy_constraint:
            out_phy_normed = output[:,:self.output_size] / self.output_scale * self.phys_output_scale
            # # --- get t,qn after --- #
            # t_new = t_before + out_phy_normed[:,self.t_phy_start:self.t_phy_end] * self.timestep
            # qn_new = qn_before + out_phy_normed[:,self.qn_phy_start:self.qn_phy_end] * self.timestep
            # liq_frac = self.apply_temperature_rules(t_new)
            # qc_new = liq_frac * qn_new
            # qi_new = (1-liq_frac) * qn_new
            # qc_phy = (qc_new - qc_before) / self.timestep
            # qi_phy = (qi_new - qi_before) / self.timestep
            # # --- end get t,qn after --- #
            # # Perform energy correction after mass changes have been applied (latent heating changes)
            energy_res = self.calc_energy_residual(dP_corr, out_phy_normed)
            t_cor = self.calc_temperature_correction(dP_corr, energy_res)
            
            t_cor = t_cor * self.output_scale[self.t_phy_start_c:self.t_phy_end] / \
                self.phys_output_scale[self.t_phy_start_c:self.t_phy_end]

            output[:,self.t_phy_start_c:self.t_phy_end] = output[:,self.t_phy_start_c:self.t_phy_end] - t_cor
            
            tot_res = tot_res + energy_res

        if self.use_confidence:
            output[:,self.output_size:] = F.relu(output[:,self.output_size:])
        
        # out_unnormed = output[:,:self.output_size] / self.output_scale
        # u_res, v_res = self.calc_uv_residual(dP, out_unnormed)
        # mass_res = self.calc_mass_residual(dP, out_unnormed)
        # energy_res = self.calc_energy_residual(dP, out_unnormed, qc_phy, qi_phy)

        tot_res *= self.e0
        output = torch.concat([output, tot_res[:,None]], dim=-1)        
        return output#, u_res, v_res, mass_res, energy_res

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        if self.noise_scheduler is not None:
            x = add_additive_noise(x, self.noise_scheduler.noise_std)
            # x = add_multiplicative_noise(x, self.noise_scheduler.noise_std)
            self.log('noise_std', self.noise_scheduler.noise_std, prog_bar=prog_bar, on_epoch=True, on_step=False, logger=True)
        # print(x.size(), y.size())
        # y = y.view(y.size(1)*y.size(0), y.size(2))
        # x = x.view(x.size(1)*x.size(0), x.size(2), x.size(3))
        y_pred = self(x)
        # if self.pinn:
        residual = torch.abs(y_pred[...,-1:]) * 385
        y_pred = y_pred[...,:-1] # discard total residual output
        # loss = self.criterion(y_pred[:, JOINT_MASK==1], y[:, JOINT_MASK==1])
#             predloss = self.criterion(yhat[:, self.zeroout_mask==0], y[:, self.zeroout_mask==0])
#             predloss = torch.mean(predloss, dim=0)
#             confloss = self.criterion(yloss[:, self.zeroout_mask==0], predloss)
#             loss = torch.mean(predloss+confloss)
        if self.use_confidence:
            # confidence loss
            yhat = y_pred[:,:self.n_column_targets*self.nlev+self.n_scalar_targets]
            ylhat = y_pred[:,self.n_column_targets*self.nlev+self.n_scalar_targets:]
            
            yhat = yhat[:,~self.zeroout_mask]
            ylhat = ylhat[:,~self.zeroout_mask]
            y = y[:,~self.zeroout_mask]
            
            yhatloss = self.criterion(yhat, y)
            # confloss computation
            # yhatloss = torch.mean(yhatloss, dim=0)
            # 1. predict actual huberloss
            confloss = self.criterion(ylhat, yhatloss)
            # 2. predict MAE
            # yhatloss_mae = self.mae_loss(yhat, y)
            # confloss = self.criterion(ylhat, yhatloss_mae)
            # end confloss computation
            loss = torch.mean(yhatloss+confloss)
            
            # From 1st place:
            # def loss_fn(x, preds):
            #     confidence = preds[:, col_num_y:]
            #     preds = preds[:, :col_num_y]
            #     loss = tf.math.abs(x-preds)
            #     loss = loss*loss_mask
            #     loss_2 = tf.math.abs(loss-confidence)
            #     loss_2 = loss_2*loss_mask
            #     return tf.reduce_mean(loss+loss_2)
            self.log('train_loss_yhat', torch.mean(yhatloss), prog_bar=prog_bar, on_epoch=True, on_step=False, logger=True)
        else: # normal loss
            yhat = yhat[:,~self.zeroout_mask]
            y = y[:,~self.zeroout_mask]
            loss = self.criterion(yhat, y)
            
        # diff loss
        y3d = y[:,:self.nlev*self.n_column_targets].view(-1,self.n_column_targets,self.nlev)
        yhat3d = yhat[:,:self.nlev*self.n_column_targets].view(-1,self.n_column_targets,self.nlev)
        y3d_diff = torch.diff(y3d, dim=2)
        yhat3d_diff = torch.diff(yhat3d, dim=2)
        ydiffloss = torch.mean(self.criterion(yhat3d_diff, y3d_diff))
        loss = loss + ydiffloss
        
        mean_residual = torch.mean(residual)
        self.log('lterm1_pred_conf', loss, prog_bar=prog_bar, on_epoch=True, on_step=False, logger=True)
        self.log('lterm2_residual', mean_residual, prog_bar=prog_bar, on_epoch=True, on_step=False, logger=True)
        if self.pinn:
            loss = (1-self.pinn_weight)*loss + self.pinn_weight * mean_residual
        self.log('train_loss', loss, prog_bar=prog_bar, on_epoch=True, on_step=False, logger=True)

        # self.lrs.append(self.lr_schedulers().get_last_lr())
        # with open('/work/bd1179/b309215/ClimSimKaggle/leap-climsim-kaggle-5th/lr_step.txt', 'a') as f:
        #     f.write(f'{self.lr_schedulers().get_last_lr()}\n')
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        # print(x.size(), y.size())
        # y = y.view(y.size(1)*y.size(0), y.size(2))
        # x = x.view(x.size(1)*x.size(0), x.size(2), x.size(3))
        
        # if torch.cuda.is_available(): # GPU time (if using GPU)
        #     torch.cuda.synchronize()
        #     start_gpu_time = time.time()
        # else: # CPU time
        #     start_cpu_time = time.time()
            
        y_pred = self(x)
        y_pred = y_pred[...,:-1] # discard total residual output
        
        # if torch.cuda.is_available(): # Measure GPU time (if using GPU)
        #     torch.cuda.synchronize()
        #     gpu_inference_time = time.time() - start_gpu_time
        #     self.gpu_times.append(gpu_inference_time)
        # else: # Measure CPU time
        #     cpu_inference_time = time.time() - start_cpu_time
        #     self.cpu_times.append(cpu_inference_time)
            
        if self.use_confidence:
            yhat = y_pred[:,:self.n_column_targets*self.nlev+self.n_scalar_targets]
            ylhat = y_pred[:,self.n_column_targets*self.nlev+self.n_scalar_targets:]
        
        y_std = Y_STD.to(y.device)
        y_mean = Y_MEAN.to(y.device)
        
        y = (y * y_std) + y_mean
        yhat[:, y_std < (1.1 * ERR)] = 0
        yhat = (yhat * y_std) + y_mean

        val_score = r2_score(yhat, y)
        self.log('val_score', val_score, on_epoch=True, on_step=False, sync_dist=True, add_dataloader_idx=True)#, logger=True, prog_bar=prog_bar)

        yhatloss = self.criterion(yhat, y)
        confloss = self.criterion(ylhat, yhatloss)
        loss = torch.mean(yhatloss+confloss)
        self.log('val_loss_yhat', torch.mean(yhatloss), on_epoch=True, on_step=False, sync_dist=True, add_dataloader_idx=True)
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True, add_dataloader_idx=True)

        # yhat[:, ADJUSTMENT_MASK==0] = y[:, ADJUSTMENT_MASK==0]

        # yhat[:, MASK==0] = 0
        # y[:, MASK==0] = 0
        yhat[:, self.zeroout_mask] = 0
        y[:, self.zeroout_mask] = 0

        
        masked_val_score = r2_score(yhat, y)
        # self.val_score.append(val_score)
        # self.masked_val_score.append(masked_val_score)
        self.log('masked_val_score', masked_val_score, on_epoch=True, on_step=False, sync_dist=True, add_dataloader_idx=True)#, logger=True, prog_bar=prog_bar)
        # return {"val_score": val_score, "masked_val_score": masked_val_score}
        # self.log('n_model_params', self.n_model_params, on_epoch=True, on_step=False, add_dataloader_idx=True)#, logger=True, prog_bar=prog_bar)
        return val_score

    def on_validation_end(self):
        if self.noise_scheduler is not None:
            current_val_score = self.trainer.callback_metrics.get("val_score")
            new_noise_std = self.noise_scheduler.step(current_val_score)

    def configure_optimizers(self):

#         ###################################
#         LEARNING_RATE = 6.5e-4

#         optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=2e-3)

        # # milestones = [3, 6, 8, 11, 14, 17]
        # milestones = [5, 10, 15, 20, 25, 30]
        # gamma = 0.65
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
        # optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=2e-3)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)

        # return [optimizer], [scheduler]
        
#         optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0002)
#         optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0002) # for 1.6M model
#         # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) # for 5.1M model
#         # optimizer = optim.AdamW(model.parameters(), lr=0.00065, weight_decay=0.01) # for 11.4M model
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
        
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler" : {
#                 "scheduler": scheduler,
#                 "monitor": "val_score"
#             },
#         }

        if argv1 == 0:
            optimizer = optim.AdamW(self.parameters(), lr=0.005, weight_decay=0.0002)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=STEPS_PER_EPOCH*2, T_mult=2)
    
            return {
                "optimizer": optimizer,
                "lr_scheduler" : {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }
        
        # if argv1 == 2:
        #     optimizer = optim.AdamW(self.parameters(), lr=0.00065, weight_decay=0.01)
        # optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        
        if argv1 == 1 or argv1 == 4:
            return optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        elif argv1 == 5 or argv1 == 6 or argv1 == 7:
            return optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.0002)
        elif argv1 == 2:
            optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        elif argv1 == 3 or argv1 == 8:
            optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.0002)
            
        #     # if argv1 == 1:
        #     #     return optimizer
        #     # optimizer = SOAP(self.parameters(), lr=0.001, weight_decay=0.0002)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=2)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=10**(-0.2), patience=6)
        return {
            "optimizer": optimizer,
            "lr_scheduler" : {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_score"
            },
        }
        
#         optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)

#         return optimizer
    # def on_before_zero_grad(self, *args, **kwargs):
    #     if self.global_step > 50 and self.global_step % 8 == 0:
    #         self.ema.update(self.network.parameters())
import torch
from torch import nn

NLEV_ORIG = 60

class WrappedModelQn(nn.Module):
    def __init__(self, original_model, input_sub, input_div, out_scale, lbd_qn, feature_lengths, nlev):
        super(WrappedModelQn, self).__init__()
        self.original_model = original_model
        self.input_sub = nn.Parameter(torch.tensor(input_sub, dtype=torch.float32), requires_grad=False)
        self.input_div = nn.Parameter(torch.tensor(input_div, dtype=torch.float32), requires_grad=False)
        self.out_scale = nn.Parameter(torch.tensor(out_scale, dtype=torch.float32), requires_grad=False)
        self.lbd_qn = nn.Parameter(torch.tensor(lbd_qn, dtype=torch.float32), requires_grad=False)
        self.n_column_features, self.n_scalar_features = \
            feature_lengths
        self.timestep = 900. # s
        # self.out_shape = out_shape
    
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

    def preprocessing(self, x):
        
        # # convert v4 input array to v5 input array:
        # xout = x
        # xout_new = torch.zeros((xout.shape[0], 1405), dtype=xout.dtype)
        # xout_new[:,0:120] = xout[:,0:120]
        # xout_new[:,120:180] = xout[:,120:180] + xout[:,180:240]
        # xout_new[:,180:240] = self.apply_temperature_rules(xout[:,0:NLEV])
        # xout_new[:,240:840] = xout[:,240:840] #NLEV*14
        # xout_new[:,840:900] = xout[:,840:900]+ xout[:,900:960] #dqc+dqi
        # xout_new[:,900:1080] = xout[:,960:1140]
        # xout_new[:,1080:1140] = xout[:,1140:1200]+ xout[:,1200:1260]
        # xout_new[:,1140:1405] = xout[:,1260:1525]
        # x = xout_new
        
        #do input normalization
        x[...,120:180] = 1 - torch.exp(-x[...,120:180] * self.lbd_qn)
        x = (x - self.input_sub) / self.input_div
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
        
        #prune top 15 levels in qn input
        x[...,120:120+15] = 0
        #clip rh input
        x[..., 60:120] = torch.clamp(x[..., 60:120], 0, 1.2)
        # print(torch.nonzero(np.abs(torch.mean(x, axis=0) - 1)<1e-6))

        bdim = x.size()[0]
        x = torch.cat(
            [
                x[:,:self.n_column_features*NLEV].reshape(bdim, self.n_column_features, NLEV).transpose(2, 1),
                x[:,self.n_column_features*NLEV:].unsqueeze(2).repeat(1, NLEV, 1).view(bdim, NLEV, self.n_scalar_features),
            ],
            -1,
        )
        return x

    def postprocessing(self, x):
        yhat = x[...,:self.out_scale.shape[0]] # for throwing away confidence predictions
        error = x[...,self.out_scale.shape[0]:2*self.out_scale.shape[0]]
        residual = x[...,self.out_scale.shape[0]*2]
        # residual = torch.zeros_like(error[...,0])
        
        mean_error = torch.mean(error, dim=1)
        
        yhat[...,60:75] = 0
        yhat[...,120:135] = 0
        yhat[...,180:195] = 0
        yhat[...,240:255] = 0
        yhat = yhat/self.out_scale
        
        return torch.concatenate([yhat, mean_error[:,None], residual[:,None]], dim=1)

    def forward(self, x):
        t_before = x[...,0:60].clone()
        # qc_before = x[...,120:180].clone()
        # qi_before = x[...,180:240].clone()
        qn_before = x[...,120:180].clone()
        liq_frac = self.apply_temperature_rules(t_before)
        qc_before = liq_frac * qn_before
        qi_before = (1-liq_frac) * qn_before
        # print("qc_before2: ", torch.mean(qc_before))
        # print("qi_before2: ", torch.mean(qi_before))
        
        x = self.preprocessing(x)
        # print("qn_before2 (normed): ", torch.mean(x[:,:,2]))
        with torch.no_grad():#, torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True): # autocast for 16 bit precision
            x = self.original_model(x)
            # x,ur,vr,mr,er = self.original_model(x)
        x = self.postprocessing(x)
        
        t_new = t_before + x[...,0:60]*self.timestep
        # print("t_after2: ", torch.mean(t_new))
        qn_new = qn_before + x[...,120:180]*self.timestep
        liq_frac = self.apply_temperature_rules(t_new)
        # print(qn_new)
        # print()
        # print(liq_frac)
        qc_new = liq_frac*qn_new
        qi_new = (1-liq_frac)*qn_new
        # print("qc_after2: ", torch.mean(qc_new))
        # print("qi_after2: ", torch.mean(qi_new))
        # print(qc_new)
        # print(qi_new)
        xout = torch.zeros((x.shape[0],self.out_scale.shape[0]+NLEV+2))
        xout[...,0:120] = x[...,0:120]
        xout[...,120:180] = (qc_new - qc_before)/self.timestep
        xout[...,180:240] = (qi_new - qi_before)/self.timestep
        # print("qc_phy2: ", torch.mean(xout[...,120:180]))
        # print("qi_phy2: ", torch.mean(xout[...,180:240]))
        xout[...,240:] = x[...,180:]
        # print(x)
    
        return xout#,ur,vr,mr,er#, pred_error_mean

        
class WrappedModelQnSimple(nn.Module):
    def __init__(self, original_model, input_sub, input_div, out_scale, lbd_qn, feature_lengths, nlev):
        super(WrappedModelQn, self).__init__()
        self.original_model = original_model
        self.input_sub = nn.Parameter(torch.tensor(input_sub, dtype=torch.float32), requires_grad=False)
        self.input_div = nn.Parameter(torch.tensor(input_div, dtype=torch.float32), requires_grad=False)
        self.out_scale = nn.Parameter(torch.tensor(out_scale, dtype=torch.float32), requires_grad=False)
        self.lbd_qn = nn.Parameter(torch.tensor(lbd_qn, dtype=torch.float32), requires_grad=False)
        self.n_column_features, self.n_scalar_features = \
            feature_lengths
        self.timestep = 900. # s
        # self.out_shape = out_shape
    
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

    def preprocessing(self, x):
        
        # # convert v4 input array to v5 input array:
        # xout = x
        # xout_new = torch.zeros((xout.shape[0], 1405), dtype=xout.dtype)
        # xout_new[:,0:120] = xout[:,0:120]
        # xout_new[:,120:180] = xout[:,120:180] + xout[:,180:240]
        # xout_new[:,180:240] = self.apply_temperature_rules(xout[:,0:NLEV])
        # xout_new[:,240:840] = xout[:,240:840] #NLEV*14
        # xout_new[:,840:900] = xout[:,840:900]+ xout[:,900:960] #dqc+dqi
        # xout_new[:,900:1080] = xout[:,960:1140]
        # xout_new[:,1080:1140] = xout[:,1140:1200]+ xout[:,1200:1260]
        # xout_new[:,1140:1405] = xout[:,1260:1525]
        # x = xout_new
        
        #do input normalization
        x[...,120:180] = 1 - torch.exp(-x[...,120:180] * self.lbd_qn)
        x = (x - self.input_sub) / self.input_div
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
        
        #prune top 15 levels in qn input
        x[...,120:120+15] = 0
        #clip rh input
        x[..., 60:120] = torch.clamp(x[..., 60:120], 0, 1.2)
        # print(torch.nonzero(np.abs(torch.mean(x, axis=0) - 1)<1e-6))

        bdim = x.size()[0]
        x = torch.cat(
            [
                x[:,:self.n_column_features*NLEV].reshape(bdim, self.n_column_features, NLEV).transpose(2, 1),
                x[:,self.n_column_features*NLEV:].unsqueeze(2).repeat(1, NLEV, 1).view(bdim, NLEV, self.n_scalar_features),
            ],
            -1,
        )
        return x

    def postprocessing(self, x):
        x[...,60:75] = 0
        x[...,120:135] = 0
        x[...,180:195] = 0
        x[...,240:255] = 0
        x = x/self.out_scale
        
        return x

    def forward(self, x):
        t_before = x[...,0:60].clone()
        qn_before = x[...,120:180].clone()
        liq_frac = self.apply_temperature_rules(t_before)
        qc_before = liq_frac * qn_before
        qi_before = (1-liq_frac) * qn_before
        
        x = self.preprocessing(x)
        with torch.no_grad():#, torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True): # autocast for 16 bit precision
            x = self.original_model(x)
            # x,ur,vr,mr,er = self.original_model(x)
        x = self.postprocessing(x)
        
        t_new = t_before + x[...,0:60]*self.timestep
        qn_new = qn_before + x[...,120:180]*self.timestep
        liq_frac = self.apply_temperature_rules(t_new)
        qc_new = liq_frac*qn_new
        qi_new = (1-liq_frac)*qn_new
        xout = torch.zeros((x.shape[0],self.out_scale.shape[0]+NLEV))
        xout[...,0:120] = x[...,0:120]
        xout[...,120:180] = (qc_new - qc_before)/self.timestep
        xout[...,180:240] = (qi_new - qi_before)/self.timestep
        xout[...,240:] = x[...,180:]
    
        return xout#,ur,vr,mr,er#, pred_error_mean

class WrappedModelQcQi(nn.Module):
    def __init__(self, original_model, input_sub, input_div, out_scale, lbd_qc, lbd_qi, feature_lengths, nlev):
        super(WrappedModelQcQi, self).__init__()
        self.original_model = original_model
        self.input_sub = nn.Parameter(torch.tensor(input_sub, dtype=torch.float32), requires_grad=False)
        self.input_div = nn.Parameter(torch.tensor(input_div, dtype=torch.float32), requires_grad=False)
        self.out_scale = nn.Parameter(torch.tensor(out_scale, dtype=torch.float32), requires_grad=False)
        self.lbd_qc = nn.Parameter(torch.tensor(lbd_qc, dtype=torch.float32), requires_grad=False)
        self.lbd_qi = nn.Parameter(torch.tensor(lbd_qi, dtype=torch.float32), requires_grad=False)
        self.n_column_features, self.n_scalar_features = \
            feature_lengths
        self.nlev = nlev
        self.prune_lvl = nlev - NLEV_ORIG + 15
        self.prune_lvl = self.prune_lvl if self.prune_lvl > 0 else 0
        # self.timestep = 900. # s
        # self.out_shape = out_shape
    
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

    def preprocessing(self, x):
        # todo change for wolat
        # x = x[...,:-2]
        
        #do input normalization
        x[...,2*self.nlev:3*self.nlev] = 1 - torch.exp(-x[...,2*self.nlev:3*self.nlev] * self.lbd_qc)
        x[...,3*self.nlev:4*self.nlev] = 1 - torch.exp(-x[...,3*self.nlev:4*self.nlev] * self.lbd_qi)
        x = (x - self.input_sub) / self.input_div
        x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
        
        #prune top 15 levels in qn input
        x[...,self.nlev*2:self.nlev*2+self.prune_lvl] = 0
        x[...,self.nlev*3:self.nlev*3+self.prune_lvl] = 0
        #clip rh input
        x[..., self.nlev:2*self.nlev] = torch.clamp(x[..., self.nlev:2*self.nlev], 0, 1.2)
        # print(torch.nonzero(np.abs(torch.mean(x, axis=0) - 1)<1e-6))

        bdim = x.size()[0]
        x = torch.cat(
            [
                x[:,:self.n_column_features*self.nlev].reshape(bdim, self.n_column_features, self.nlev).transpose(2, 1),
                x[:,self.n_column_features*self.nlev:].unsqueeze(2).repeat(1, self.nlev, 1).view(bdim, self.nlev, self.n_scalar_features),
            ],
            -1,
        )
        return x

    def postprocessing(self, x):
        yhat = x[...,:self.out_scale.shape[0]] # for throwing away confidence predictions
        error = x[...,self.out_scale.shape[0]:self.out_scale.shape[0]*2]
        residual = x[...,self.out_scale.shape[0]*2]
        
        mean_error = torch.mean(error, dim=1)
        # print(torch.mean(mean_error))
        
        yhat[...,1*self.nlev:1*self.nlev+self.prune_lvl] = 0
        yhat[...,2*self.nlev:2*self.nlev+self.prune_lvl] = 0
        yhat[...,3*self.nlev:3*self.nlev+self.prune_lvl] = 0
        yhat[...,4*self.nlev:4*self.nlev+self.prune_lvl] = 0
        yhat[...,5*self.nlev:5*self.nlev+self.prune_lvl] = 0
        yhat = yhat/self.out_scale
        
        return torch.concatenate([yhat, mean_error[:,None], residual[:,None]], dim=1)

    def forward(self, x):
        
        x = self.preprocessing(x)
        with torch.no_grad():#, torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True): # autocast for 16 bit precision
            x = self.original_model(x)
            # print(torch.mean(x[:,362:]))
        x = self.postprocessing(x)
    
        return x#, pred_error_mean
        
class WrappedModelTruth(nn.Module):
    def __init__(self, original_model, input_sub, input_div, out_scale, lbd_qn, feature_lengths, out_shape):
        super(WrappedModelTruth, self).__init__()
        self.out_scale = nn.Parameter(torch.tensor(out_scale, dtype=torch.float32), requires_grad=False)
        self.timestep = 900. # s
        self.out_shape = out_shape
    
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

    def forward(self, x, y):
        t_before = x[...,0:60].clone()
        # qc_before = x[...,120:180].clone()
        # qi_before = x[...,180:240].clone()
        qn_before = x[...,120:180].clone()
        liq_frac = self.apply_temperature_rules(t_before)
        qc_before = liq_frac * qn_before
        qi_before = (1-liq_frac) * qn_before
        
        x = y.clone()
        
        t_new = t_before + x[...,0:60]*self.timestep
        qn_new = qn_before + x[...,120:180]*self.timestep
        liq_frac = self.apply_temperature_rules(t_new)
        # print(qn_new)
        # print()
        # print(liq_frac)
        qc_new = liq_frac*qn_new
        qi_new = (1-liq_frac)*qn_new
        # print(qc_new)
        # print(qi_new)
        xout = torch.zeros((x.shape[0],self.out_shape))
        xout[...,0:120] = x[...,0:120]
        xout[...,120:180] = (qc_new - qc_before)/self.timestep
        xout[...,180:240] = (qi_new - qi_before)/self.timestep
        xout[...,240:] = x[...,180:]
        # print(x)
    
        return xout#, pred_error_mean
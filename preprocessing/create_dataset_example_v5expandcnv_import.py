from climsim_utils.data_utils import *

def read_save_data(args):
    ym, split, save_base_path = args
    print(f'Reading data for {ym}')
    grid_path = '/p/scratch/icon-a-ml/heuer1/LEAP/ClimSim_high-res/ClimSim_high-res_grid-info.nc'
    norm_path = '/p/project/icon-a-ml/heuer1/ClimSim/preprocessing/normalizations/'

    grid_info = xr.open_dataset(grid_path)
    input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v5_pervar.nc')
    input_max = xr.open_dataset(norm_path + 'inputs/input_max_v5_pervar.nc')
    input_min = xr.open_dataset(norm_path + 'inputs/input_min_v5_pervar.nc')
    output_scale = xr.open_dataset(norm_path + 'outputs/output_scale_std_lowerthred_v5.nc')

    data = data_utils(grid_info = grid_info, 
                      input_mean = input_mean, 
                      input_max = input_max, 
                      input_min = input_min, 
                      output_scale = output_scale,
                      input_abbrev = 'mlexpandcnv',
                      output_abbrev = 'mlo',
                      # ml_backend = 'pytorch',
                      normalize=False,
                      save_h5=False,
                      save_zarr=True,
                      save_npy=False
                      )
    
    data.data_path = f'/p/scratch/icon-a-ml/heuer1/LEAP/ClimSim_high-res/train/'#/{ym}/'

    # set inputs and outputs to V5 subset
    data.set_to_v5cnvqcqi_vars()

    regex = f'E3SM-MMF.mlexpandcnv.{ym}-*.nc'
    # set regular expressions for selecting training data
    # data.set_regexps(data_split = 'train',
    #                 regexps = [f'E3SM-MMF.mlexpandcnv.000[1234567]-*-*.nc',#{i:02d}-*.nc', # years 1 through 7
    #                         'E3SM-MMF.mlexpandcnv.0008-01-*-*.nc']) # first month of year 8
    data.set_regexps(data_split = split,
                     # regexps = regex_list)
                     # regexps = [regex_list[i]])
                     regexps = [regex])
    # set temporal subsampling
    data.set_stride_sample(data_split = split, stride_sample = 1)
    # create list of files to extract data from
    data.set_filelist(data_split = split, start_idx=0)
    
    save_path = os.path.join(save_base_path, ym)
    os.makedirs(save_path)#, exist_ok=True)
    data.save_as_npy(data_split = split, save_path = save_path)
    print(f'Finished processing of {save_path}')

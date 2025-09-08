import os
import glob
import xarray as xr
import numpy as np
import metpy.constants.nounit as metconstnondim
from numba import njit

def process_one_file(args):
    """
    Process a single NetCDF file by updating its dataset with information from previous files.
    
    Args:
        i: int
            The index of the current file in the full file list.
        nc_files_in: list of str
            List of the full filenames.
        lat: xarray.DataArray
            DataArray of latitude.
        lon: xarray.DataArray
            DataArray of longitude.
        input_abbrev: str
            The input file name abbreviation, the default input data should be 'mli'.
        output_abbrev: str
            The output file name abbreviation, the default output data should be 'mlo'.
        input_abbrev_new: str
            The abbreviation for the new input file name.
    
    Returns:
        None
    """
    timestep = 1200 # s
    ncols = 21600
    
    i, nc_files_in, lat, lon, iface, mask_strato, input_abbrev, output_abbrev, rad_abbrev, input_abbrev_new = args
    xr_args = dict()
    # xr_args = dict(chunks='auto')
    # xr_args = dict(chunks={'lev':1})
    new_file_path = nc_files_in[i].replace(input_abbrev, input_abbrev_new)
    if os.path.exists(new_file_path):
        print(f'{new_file_path} already exists, skipping this one')
        return None
    
    # print(f'Calculating for {new_file_path}')
    dsin = xr.open_dataset(nc_files_in[i], **xr_args)
    dsin_prev = xr.open_dataset(nc_files_in[i-1], **xr_args)
    dsin_prev2 = xr.open_dataset(nc_files_in[i-2], **xr_args)
    dsout = xr.open_dataset(nc_files_in[i].replace(input_abbrev, output_abbrev), **xr_args)
    dsout_prev = xr.open_dataset(nc_files_in[i-1].replace(input_abbrev, output_abbrev), **xr_args)
    dsout_prev2 = xr.open_dataset(nc_files_in[i-2].replace(input_abbrev, output_abbrev), **xr_args)
    dsrad = xr.open_dataset(nc_files_in[i].replace(input_abbrev, rad_abbrev), **xr_args).rename({'col': 'ncol'})
    dsrad_prev = xr.open_dataset(nc_files_in[i-1].replace(input_abbrev, rad_abbrev), **xr_args).rename({'col': 'ncol'})
    dsrad_prev2 = xr.open_dataset(nc_files_in[i-2].replace(input_abbrev, rad_abbrev), **xr_args).rename({'col': 'ncol'})

    dsin['tm_state_t'] = dsin_prev['state_t']
    dsin['tm_state_q0001'] = dsin_prev['state_q0001']
    dsin['tm_state_q0002'] = dsin_prev['state_q0002']
    dsin['tm_state_q0003'] = dsin_prev['state_q0003']
    dsin['tm_state_u'] = dsin_prev['state_u']
    dsin['tm_state_v'] = dsin_prev['state_v']
    
    dsin['state_t_phy'] = (dsout['state_t'] - dsin['state_t']) / timestep - dsrad['ptend_t']
    dsin['state_q0001_phy'] = (dsout['state_q0001'] - dsin['state_q0001']) / timestep
    dsin['state_q0002_phy'] = (dsout['state_q0002'] - dsin['state_q0002']) / timestep
    dsin['state_q0003_phy'] = (dsout['state_q0003'] - dsin['state_q0003']) / timestep
    dsin['state_u_phy'] = (dsout['state_u'] - dsin['state_u']) / timestep
    dsin['state_v_phy'] = (dsout['state_v'] - dsin['state_v']) / timestep
    
    if mask_strato:
        tropopauses = tropopause_profile_2d(dsin.state_pmid.values,
                                             dsin.state_t.values,
                                             qv_profile=dsin.state_q0001.values,
                                             pmin=0.01, pmax=450e2)
        ptp = tropopauses[:,1]
        mask_strato = dsin.state_pmid.values < ptp

        for varname in dsin:
            if varname.endswith('phy'):
                # print(f'Stratospheric masking of {varname}')
                dsin[varname].values[mask_strato] = 0

    dsin['state_t_prvphy'] = (dsout_prev['state_t'] - dsin_prev['state_t']) / timestep - dsrad_prev['ptend_t']
    dsin['state_q0001_prvphy'] = (dsout_prev['state_q0001'] - dsin_prev['state_q0001']) / timestep
    dsin['state_q0002_prvphy'] = (dsout_prev['state_q0002'] - dsin_prev['state_q0002']) / timestep
    dsin['state_q0003_prvphy'] = (dsout_prev['state_q0003'] - dsin_prev['state_q0003']) / timestep
    dsin['state_u_prvphy'] = (dsout_prev['state_u'] - dsin_prev['state_u']) / timestep

    dsin['tm_state_t_prvphy'] = (dsout_prev2['state_t'] - dsin_prev2['state_t']) / timestep - dsrad_prev2['ptend_t']
    dsin['tm_state_q0001_prvphy'] = (dsout_prev2['state_q0001'] - dsin_prev2['state_q0001']) / timestep
    dsin['tm_state_q0002_prvphy'] = (dsout_prev2['state_q0002'] - dsin_prev2['state_q0002']) / timestep
    dsin['tm_state_q0003_prvphy'] = (dsout_prev2['state_q0003'] - dsin_prev2['state_q0003']) / timestep
    dsin['tm_state_u_prvphy'] = (dsout_prev2['state_u'] - dsin_prev2['state_u']) / timestep

    dsin['state_t_dyn'] = (dsin['state_t'] - dsout_prev['state_t']) / timestep
    dsin['state_q0_dyn'] = (dsin['state_q0001'] - dsout_prev['state_q0001'] + dsin['state_q0002'] - dsout_prev['state_q0002'] + dsin['state_q0003'] - dsout_prev['state_q0003']) / timestep
    dsin['state_u_dyn'] = (dsin['state_u'] - dsout_prev['state_u']) / timestep

    dsin['tm_state_t_dyn'] = (dsin_prev['state_t'] - dsout_prev2['state_t']) / timestep
    dsin['tm_state_q0_dyn'] = (dsin_prev['state_q0001'] - dsout_prev2['state_q0001'] + dsin_prev['state_q0002'] - dsout_prev2['state_q0002'] + dsin_prev['state_q0003'] - dsout_prev2['state_q0003']) / timestep
    dsin['tm_state_u_dyn'] = (dsin_prev['state_u'] - dsout_prev2['state_u']) / timestep
    
    dsin['dP'] = get_pressure_thickness(dsin['state_ps'], iface, dsin['state_pmid'].coords)

    dsin['tm_state_ps'] = dsin_prev['state_ps']
    dsin['tm_pbuf_SOLIN'] = dsin_prev['pbuf_SOLIN']
    dsin['tm_pbuf_SHFLX'] = dsin_prev['pbuf_SHFLX']
    dsin['tm_pbuf_LHFLX'] = dsin_prev['pbuf_LHFLX']
    dsin['tm_pbuf_COSZRS'] = dsin_prev['pbuf_COSZRS']

    dsin['lat'] = lat
    dsin['lon'] = lon
    clat = lat.copy()
    slat = lat.copy()
    icol = lat.copy()
    clat[:] = np.cos(lat*np.pi/180.)
    slat[:] = np.sin(lat*np.pi/180.)
    icol[:] = np.arange(1,ncols+1)
    dsin['clat'] = clat
    dsin['slat'] = slat
    dsin['icol'] = icol

    dsin.to_netcdf(new_file_path)
    # delayed_obj = dsin.to_netcdf(new_file_path, compute=False)
    # with ProgressBar():
    #     results = delayed_obj.compute()

    return None


def get_pint(PS, iface, coords):
    hyai, hybi, P0 = iface['hyai'], iface['hybi'], iface['P0']
    PINT = P0 * hyai + PS * hybi
    PINT.attrs.update({'units': 'Pa',
                       'long_name': 'Pressure at interface levels'})
    PINT = PINT.rename({'ilev': 'lev'})
    PINT = PINT.transpose(*coords.dims)
    return PINT

def get_pressure_thickness(PS, iface, coords):
    PINT    = get_pint(PS, iface, coords)
    dP_temp = PINT.diff('lev', n=1)
    # dP_temp = dP_temp.rename({'ilev': 'lev'})
    # dP_temp = dP_temp.transpose(*coords.dims)
    dP = xr.DataArray(dP_temp.values, coords=coords, dims=coords.dims,
                      attrs={'units': 'Pa', 
                             'long_name': 'Pressure thickness of each level'})
    return dP

def compute_altitude(pressure, temperature, humidity, g=metconstnondim.g, R_d=metconstnondim.Rd):
    """
    Compute altitude from pressure, temperature, and humidity profiles.
    
    Parameters:
    - pressure: Array of pressures (Pa)
    - temperature: Array of temperatures (K)
    - humidity: Array of relative humidities (fraction, 0-1)
    - g: Gravitational acceleration (m/s^2)
    - R_d: Specific gas constant for dry air (J/(kg·K))
    
    Returns:
    - Array of altitudes (m)
    """
    
    reverse_profile = pressure[0,0] < pressure[-1,0]
    if reverse_profile:
        pressure = pressure[::-1]
        temperature = temperature[::-1]
        humidity = humidity[::-1]

    T_v = temperature * ((humidity + 0.622)
                          / (0.622 * (1 + humidity)))
    
    result = np.empty_like(pressure)
    result[0,:] = 0

    for i in range(1,pressure.shape[0]):
        # for j in range(pressure.shape[1]):
        #     result[i,j] = -R_d / g * np.trapz(T_v[:i+1,j], np.log(pressure[:i+1,j]))
        result[i,:] = -R_d / g * np.trapz(T_v[:i+1], np.log(pressure[:i+1]), axis=0)

    if reverse_profile:
        return result[::-1]
    else:
        return result

# @njit
def tropopause_profile_2d(p_profile, t_profile, z_profile=None, qv_profile=None, dtdz_crit=-2.0, tp_thickness=2000, zmin=0, pmin=None, pmax=None):
    if not z_profile:
        z_profile = compute_altitude(p_profile, t_profile, qv_profile)
    # print(z_profile)
    result = np.empty((t_profile.shape[1], 4), dtype=t_profile.dtype)
    for i in range(t_profile.shape[1]):
        result[i,:] = tropopause_profile(p_profile[:,i], t_profile[:,i], z_profile[:,i], dtdz_crit, tp_thickness, zmin, pmin, pmax)
        
    return result

@njit
def tropopause_profile(p_profile, t_profile, z_profile, dtdz_crit=-2.0, tp_thickness=2000, zmin=0, pmin=None, pmax=None):
    """
    Function calculates the tropopause based on the WMO's definition.
    It returns the tropopause temperature and pressure where the vertical
    temperature gradient dT/dz climbs above dtdz_crit [K/km] and stays above that
    for at least tp_thickness [m].
    Returns: ttp, ptp, ztp, thickness_flag
    """
    reverse_profile = p_profile[0] < p_profile[-1]
    
    if reverse_profile:
        p_profile = p_profile[::-1]
        t_profile = t_profile[::-1]
        z_profile = z_profile[::-1]
    
    if zmin != 0:
        height_cond = np.where(z_profile>=zmin)
        t_profile = t_profile[height_cond]
        p_profile = p_profile[height_cond]
        z_profile = z_profile[height_cond]
    
    # pMin = 85.0*units.mbar
    # pMax = 450.0*units.mbar
    # if pmin is not None:
    #     pres_cond = np.where(p_profile>=pmin)
    #     t_profile = t_profile[pres_cond]
    #     p_profile = p_profile[pres_cond]
    #     z_profile = z_profile[pres_cond]
    if pmax is not None:
        pres_cond = np.where(p_profile<=pmax)
        t_profile = t_profile[pres_cond]
        p_profile = p_profile[pres_cond]
        z_profile = z_profile[pres_cond]
    # print(z_profile)

    # calculate gamma = dT/dz for each layer
    dz = z_profile[1:] - z_profile[:-1]
    # # units [K/km] but the sign isn't switched, so: dtdz = - lapse rate 
    dtdz =  (t_profile[1:] - t_profile[:-1]) / (dz/1000)
    # dtdz = np.gradient(t_profile, z_profile)*1000
    
    # look for tropopause
    # find layer(s) where gamma crosses gamma = dtdz_crit
    ks_raw = zero_crossings(dtdz-dtdz_crit)[0]
    # print(ks_raw)
    helper = (dtdz-dtdz_crit)[ks_raw]
    ks = ks_raw[np.where(helper > 0)]
    
    # calculate preliminary tropopause levels for each drop below dtdz_crit
    prel_tp = [lin_interp(z_profile[k-1:k+1], dtdz[k-1:k+1], dtdz_crit) for k in ks]
    # print(prel_tp)
    
    # Set a flag how often the thickness criterion was
    # checked and not fulfilled in the Process
    thickness_flag = 0
    
    # Check if preliminary tropopause fulfills thickness criterion
    # print(prel_tp)
    for i,ztp in enumerate(prel_tp):
        k = ks[i]
        dz_preltp_up = z_profile[k] - ztp
        # print(dz_preltp_up)
        # Calculate number of levels inside the tp_thickness (usually the 2km range imposed by the WMO's criterion)
        dz_array = np.append(dz_preltp_up, dz[k:])
        nlevs = levels_in_tplayer(dz_array, tp_thickness)
        if not nlevs:
            return np.nan, np.nan, np.nan, np.nan
        
        # Return tropopause pressure and temperature if any preliminary tropopause fulfills
        # the thickness criterion, cut dz_array at tp_thickness km
        dz_end = tp_thickness - np.sum(dz_array[:nlevs])
        dz_array = np.append(dz_array[:nlevs], dz_end)
        dtdz_array = np.append(dtdz_crit, dtdz[k:k+nlevs])
        
        # dtdz array flipped as thickness criterion beginns with lowermost layer going up
        if thickness_criterion(dtdz_array, dz_array, dtdz_crit):
            # temperature at tropopause
            ttp = lin_interp(t_profile[ks[i]-1:ks[i]+1], dtdz[ks[i]-1:ks[i]+1], dtdz_crit)
            # pressure at tropopause
            ptp = lin_interp(p_profile[k-1:k+1], z_profile[k-1:k+1], ztp)
            # height calculated at tropopause
            if ptp < pmin:
                continue
            return ttp, ptp, ztp, thickness_flag
        else:
            thickness_flag += 1
    return np.nan, np.nan, np.nan, thickness_flag
    
@njit
def thickness_criterion(dtdz, dz, dtdz_crit):
    """
    Check for the WMO's thickness criterion for tropopauses. dtdz has to stay below dtdz_crit K/km on average between 
    the preliminary tropopause and each of the layers above in tp_thickness [m].
    """

    # print(dtdz)
    for i in np.arange(2, dtdz.size, 1):
        # print(np.average(dtdz[:i]))#, weights=dz[:i]))
        if np.average(dtdz[:i], weights=dz[:i]) < dtdz_crit:
        # if np.average(dtdz[:i]) < dtdz_crit:
            return False
    return True
    
@njit
def levels_in_tplayer(dz_array, tp_thickness):
    i = 0
    while sum(dz_array[:i]) < tp_thickness:
        i += 1
        if i == dz_array.size:
            return False
    return i-1

@njit
def lin_interp(y, x, x_int):
    return y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (x_int - x[0])

@njit
def zero_crossings(array):
    """
    Finds the array indices after the zero crossings.
    takes: array/numpy array
    returns: numpy array
    """
    
    # Vector of boolean values (True, False) indicating between which
    # indices the sign of the array values changes
    # sign_switches[0] = True would mean between there is a sign change
    # between the first and the second entry in the array
    # with np.errstate(invalid='ignore'):
    sign_switches = ((array[:-1] * array[1:]) <= 0)

    # Set up a vector that indicates the positions of the array after
    # each sing change by appending one False at the beginning of sign_switches
    after_switch = np.append(False, sign_switches)
    
    # get the indices in the array that lie after each sign_switch
    indices_after = np.where(after_switch == True)

    return(indices_after)
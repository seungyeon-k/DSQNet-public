import yaml

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, 'w') as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def complete_config(cfg):
    info_types = cfg['common_components']['info_types']
    n_types = len(info_types)
    n_primitives = cfg['common_components']['n_primitives']
    full_params = cfg['common_components']['full_params']
    full_num_params = {}
    for key, value in full_params.items():
        full_num_params[key] = len(value)
    cfg['common_components']['full_num_params'] = full_num_params
    num_pointclouds = cfg['common_components']['num_pointclouds']
    num_pointcloudsinput = cfg['common_components']['num_pointcloudsinput']
    num_gt_pointclouds = cfg['common_components']['num_gt_pointclouds']
    input_normalization = cfg['common_components']['input_normalization']

    len_physical_params = 0
    if 'box' in info_types:
        len_physical_params += 3
    if 'cone' in info_types:
        len_physical_params += 3
    if 'cylinder' in info_types:
        len_physical_params += 2
    if 'sphere' in info_types:
        len_physical_params += 2
    if 'torus' in info_types:
        len_physical_params += 3
    if 'rectangle_ring' in info_types:
        len_physical_params += 6
    if 'cylinder_ring' in info_types:
        len_physical_params += 4
    if 'semi_sphere_shell' in info_types:
        len_physical_params += 2
    if 'superquadric' in info_types:
        len_physical_params += 5
    # if 'supertoroid' in info_types:
    #     len_physical_params += 6

    list_num_each_param = [n_types, 12, len_physical_params]

    for key in ['model', 'logger']:
        cfg[key]['n_types'] = n_types
        cfg[key]['info_types'] = info_types
        cfg[key]['n_primitives'] = n_primitives
        cfg[key]['list_num_each_param'] = list_num_each_param
        cfg[key]['full_params'] = full_params
        cfg[key]['full_num_params'] = full_num_params
        cfg[key]['len_physical_params'] = len_physical_params

    for key in cfg['model']['head'].keys():
        cfg['model']['head'][key]['len_physical_params'] = len_physical_params

    for key in cfg['data'].keys():
        cfg['data'][key]['n_types'] = n_types
        cfg['data'][key]['info_types'] = info_types
        cfg['data'][key]['n_params'] = len_physical_params
        cfg['data'][key]['n_primitives'] = n_primitives
        cfg['data'][key]['full_params'] = full_params
        cfg['data'][key]['full_num_params'] = full_num_params
        cfg['data'][key]['num_pointclouds'] = num_pointclouds
        cfg['data'][key]['num_pointcloudsinput'] = num_pointcloudsinput
        cfg['data'][key]['num_gt_pointclouds'] = num_gt_pointclouds
        cfg['data'][key]['input_normalization'] = input_normalization

    cfg['training']['loss']['list_num_each_param'] = list_num_each_param
    cfg['training']['loss']['full_num_params'] = full_num_params
    cfg['training']['loss']['info_types'] = info_types

    return cfg
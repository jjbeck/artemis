import yaml

def check_config_file(config_path):
    with open (config_path) as file:
        config_param = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    return config_param['Boot Round']

import yaml

class metrics():

    def __init__(self):
        print("metrics")

    def calculate_config(self, config_path):
        with open(config_path) as file:
            config_param = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
        self.boot_round = config_param["Boot Round"]

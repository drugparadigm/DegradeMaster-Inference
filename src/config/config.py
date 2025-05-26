import argparse
import yaml

def get_args(mode='name'):
    parser = argparse.ArgumentParser()
    if mode == 'name':
        parser.add_argument('--config', type=str, default='config.yml')
    elif mode == 'case':
        parser.add_argument('--config', type=str, default='src/config/config_c.yml')
        parser.add_argument('api:app')
        parser.add_argument('-b',default='0.0.0.0:5000')
        parser.add_argument('--workers', default=1)
        parser.add_argument('--threads', default=5)
        parser.add_argument('--access-logfile', default='-')
        
        # parser.add_argument('json_data', type=str)

    args = parser.parse_args()

    config = yaml.load(
        open(args.config),
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**config)

    return args
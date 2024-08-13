import os.path
from argparse import ArgumentParser
import json
from hashlib import md5
from time import time
import pandas as pd

import git
from datetime import datetime

import os
import json
import pandas as pd

from tracker.utils import flatten_dict


def read_experiments(output_path):
    """
    Read experiments from given location.

    :param output_path: p
    :return: Dataframe containing all experiments found in {output_path}
    """
    data = []
    path, directories, _ = next(os.walk(output_path))
    for directory in directories:
        meta_path = os.path.join(path, directory, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

            if not 'files' in meta:
                _, _, files = next(os.walk(os.path.join(path, directory)))
                files.remove('meta.json')
                meta['files'] = files
            meta['path'] = path
            data.append(flatten_dict(meta))

    return pd.DataFrame(data).set_index('hash')


def read_dataframes(df_experiments):
    """
    Read dataframes for each experiment.
    :param df_experiments:
    :return:
    """
    df_dict = {}
    for idx, row in df_experiments.iterrows():
        files = row['files']
        path = row['path']

        df_dict[idx] = dict()
        for file in files:
            df_experiments = pd.read_csv(os.path.join(path, idx, file))
            df_dict[idx][file] = df_experiments
    return df_dict


class Experiment:
    """
    A class to setup and track iterative experiments.
    """
    def __init__(self, name, description=None):
        self._name = name
        self._parameters = {}
        self._files = {}
        self._description = description

    def _parse(self):
        parser = ArgumentParser(description=self._description)
        for name, values in self._parameters.items():
            # kwargs for argparse .add_argument()
            kwargs = dict(type=values['type'], help=values['help'], choices=values['choices'],
                                default=values['default'], action=values['action'])
            # drop keys with value None
            kwargs = {k : v for k,v in kwargs.items() if v is not None}
            parser.add_argument(f"--{name}", **kwargs)
        return vars(parser.parse_args())

    def _compute_hash(self, args):
        # create a hash of the parameter configuration
        hash_params = {k: v for k, v in sorted(args.items(), key=lambda i: i[0]) if
                       v is not None and self._parameters[k]['hash']}
        return md5(json.dumps(hash_params, sort_keys=True).encode()).hexdigest()

    def _git_status(self, args):
        try:
            repo = git.Repo(search_parent_directories=True, path=args.get('code_path'))
        except git.InvalidGitRepositoryError:
            return None
        return dict(
            sha=repo.head.object.hexsha,
            branch=repo.active_branch.name,
            modified=[a.a_path for a in repo.index.diff(None)] + [a.a_path for a in repo.index.diff('Head')],
            untracked=repo.untracked_files
        )

    def run(self, fct):
        args = self._parse()

        metadata = dict()
        metadata['timestamp'] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        metadata['hash'] = self._compute_hash(args)

        # add git information
        git_status = self._git_status(args)
        if git_status is None:
            print("git repository not found")
        else:
            metadata['git'] = git_status

        metadata['parameters'] = args
        if self._description:
            metadata['description'] = self._description

        # run function, time it
        t0 = time()
        extra_metadata, df_dict = fct(**args)
        metadata['seconds'] = f"{time() - t0:.2f}"

        # additional meta data
        if extra_metadata is not None:
            metadata.update(extra_metadata)

        path = os.path.join(args['output_path'], metadata['hash'])

        if args['output_path']:
            if not os.path.exists(path):
                os.makedirs(path)

            # save dataframes
            if df_dict is not None:
                for file, df in df_dict.items():
                    df.to_csv(os.path.join(path, file))

                metadata['files'] = [*df_dict.keys()]

            # save meta data
            with open(os.path.join(path, 'meta.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Experiment completed in {metadata['seconds']} seconds. Saved to {path}.")

        return metadata, df_dict

    def add_parameter(self, name, type=None, default=None, required=True, description="", hash=True, help=None,
                      choices=None, action=None):
        """
        Add input parameter. Follows the argparse syntax.
        """
        self._parameters[name] = dict(type=type,
                                      default=default,
                                      required=required,
                                      description=description,
                                      hash=hash,
                                      help=help,
                                      choices=choices,
                                      action=action)

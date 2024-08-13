import argparse
import os
import json
import hashlib
import time
from assembly_gym.utils.rendering import plot_assembly_env, plot_cra_assembly
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot as plt

from assembly_gym.utils.structures import trapezoid_bridge, hexagon, tower, hexagon_bridge_3, hexagon_bridge_5, levitating_block, horizontal_bridge
from assembly_gym.utils.stability import is_stable_pybullet, is_stable_rbe, is_stable_cra, is_stable_rbe_penalty, is_stable_cra_penalty


def compute_hash(**kwargs):
    return hashlib.md5(json.dumps(dict(**kwargs), sort_keys=True).encode('utf-8')).hexdigest()


STRUCTURES = [
    (hexagon_bridge_3, dict(freeze_last=True)),
    (hexagon_bridge_5, dict(freeze_last=True)),
    (trapezoid_bridge, dict(freeze_last=True)),
    (trapezoid_bridge, dict(freeze_last=False)),
    (horizontal_bridge, dict(freeze_last=False)),
    (horizontal_bridge, dict(freeze_last=True)),
    (hexagon, dict()),
    (tower, dict(num_blocks=10)),
    (levitating_block, dict()),
    (levitating_block, dict(freeze_last=True)),
]

METHODS = [
    ('pybullet', is_stable_pybullet, dict(return_states=False)),
    ('rbe', is_stable_rbe, dict()),
    ('rbe_penalty', is_stable_rbe_penalty, dict(tol=1e-3)),
    ('cra', is_stable_cra, dict()),
    ('cra_penalty', is_stable_cra_penalty, dict(tol=1e-3)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--recompute_existing', action='store_true')
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=0.8)

    args = parser.parse_args()

    output_path = args.output_path
    mu = args.mu
    density = args.density

    # # suppress warnings
    # import logging
    # logging.disable(logging.WARNING)

    for structure, structure_kwargs in STRUCTURES:
        env, actions = structure(mu=mu, density=density, **structure_kwargs)

        structure_id = compute_hash(__name__=structure.__name__, **structure_kwargs)
        structure_path = os.path.join(output_path, structure_id)

        if not os.path.exists(structure_path):
            os.makedirs(structure_path)

        json_path = os.path.join(structure_path, 'structure.json')
        
        if os.path.exists(json_path):
            data = json.load(open(json_path))
        else:
            data = dict()
            data['structure'] = dict(name=structure.__name__, kwargs=structure_kwargs, plots_env=dict(), plots_cra=dict())
            data['methods'] = dict()
            data['tests'] = dict()


        for method_name, method, meth_kwargs in METHODS:
            method_id = compute_hash(name=method_name, **meth_kwargs)
            data['methods'][method_id] = dict(name=method_name, kwargs=meth_kwargs)
        
        for step, (action, is_stable) in enumerate(actions):
            env.step(action)
            
            # plotting
            path = os.path.join(structure_path, f'structure_{step}.png')
            if not os.path.exists(path) or args.recompute_existing:
                fig, ax = plt.subplots(1,1, figsize=(5,5))
                plot_assembly_env(env, fig=fig, ax=ax)
                plt.tight_layout()
                plt.savefig(path)
                plt.close('all')
                data['structure']['plots_env'][step] = os.path.abspath(path)

            path = os.path.join(structure_path, f'structure_cra_{step}.png')
            if not os.path.exists(path) or args.recompute_existing:
                fig, ax = plt.subplots(1,1, figsize=(5,5))
                plot_cra_assembly(env.assembly_env.cra_assembly, plot_forces=False, fig=fig, ax=ax, bounds=env.assembly_env.bounds)
                plt.tight_layout()
                plt.savefig(path)
                plt.close('all')
                data['structure']['plots_cra'][step] = os.path.abspath(path)

            # setup test data dict
            test_id = compute_hash(mu=mu, density=density, step=step)
            if test_id in data['tests']:
                test_data = data['tests'][test_id]
            else:
                test_data = dict(step=step, is_stable=is_stable, mu=mu, density=density)
                data['tests'][test_id] = test_data


            for method_name, method, meth_kwargs in METHODS:
                method_id = compute_hash(name=method_name, **meth_kwargs)

                if test_id in test_data and not args.recompute_existing:
                    continue
                
                t = time.time()
                res, extra = method(env.assembly_env, **meth_kwargs)
                test_data[method_id] = dict(is_stable=res, extra=extra, time=time.time()-t)

        # save json
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
    

if __name__ == '__main__':
    main()
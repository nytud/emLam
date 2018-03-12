"""Remote execution for preprocess_corpus.py."""

try:
    import itertools.izip_longest as zip_longest
    import itertools.izip as zip
except:
    from itertools import zip_longest

from io import StringIO
import os
import time

from fabric.api import cd, env, execute, put, run, task
from fabric.context_managers import settings
from fabric.contrib import files

from emLam.utils import source_target_file_list
from emLam.utils.config import get_config_file


commands = ['setup', 'do_it', 'cleanup']


class FabricError(Exception):
    pass


@task
def parallellism(cores=8):
    """
    The number of processes to be run parallelly: one per <cores> cores.
    Required, as some preprocessors (hunlp-GATE, CoreNLP) take up lots of
    resources, and it doesn't make sense to run too many of them.
    """
    result = run("cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1")
    if result.failed:
        return 1
    else:
        return (int(result) + 1) // 8


@task
def setup_infrastructure(config, local_args):
    run('mkdir -p {}'.format(config['remote_dir']))
    # Should I do this? Work dir most likely exists anyway
    run('mkdir -p {}'.format(config['work_dir']))
    with cd(config['remote_dir']):
        put(get_config_file(local_args.configuration), 'rconfig.conf')


@task
def setup_source(config, remote_dir):
    """Clones the git repository and updates it."""
    with cd(remote_dir):
        if files.exists('emLam'):
            if config['always_update']:
                with cd('emLam'):
                    run('git pull')
        else:
            run('git clone {}'.format(config['repository']))
        with cd('emLam'):
            run('git checkout {}'.format(config['object']))


@task
def setup_environment(config, remote_dir, source_dir):
    # Create the tmux session
    tid = config['tmux']
    run('tmux new-session -A -d -s {}'.format(tid))

    sentinel = '{}/pip_done'.format(remote_dir)
    if config['virtualenv']:
        if run('which virtualenv', warn_only=True).failed:
            if config['allow_user_packages']:
                run('pip install --user virtualenv')
            else:
                raise FabricError('virtualenv is not available and user '
                                  'packages are not allowed.')
        venv_dir = os.path.join(remote_dir, config['virtualenv'])
        if not files.exists(venv_dir):
            run('virtualenv -p python3 {}'.format(venv_dir))
        # It is an error to not be able to enter the environment
        source_venv = 'source {}'.format(os.path.join(venv_dir, 'bin/activate'))
        run(source_venv)
        run("tmux send -t {} '{}' ENTER".format(tid, source_venv))
        run("tmux send -t {} 'pip install -U pip setuptools' ENTER".format(
            tid))
        run("tmux send -t {} 'pip install -e {}' ENTER".format(
            tid, source_dir))
        run("tmux send -t {} 'touch {}' ENTER".format(tid, sentinel))
    elif config['allow_user_packages']:
        # Not in tmux, so that we can see if an error happens
        run('pip install --user -U pip setuptools')
        run('pip install --user -e {}'.format(source_dir))
        run('touch {}'.format(sentinel))
    else:
        raise FabricError('User packages are not allowed -- cannot run '
                          'without virtualenv.')

    for _ in range(30):
        if files.exists(sentinel):
            run('ls {}'.format(remote_dir))
            run('rm {}'.format(sentinel))
            break
        time.sleep(1)
    else:
        raise FabricError('Could not install the emLam package.')


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks. The itertools recipe."
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def input_chunks(local_args, processes):
    """Splits the input into equal sized chunks."""
    sfs = list(zip(
        *source_target_file_list(local_args.source, local_args.target_dir)))[0]
    t_chunks = list(grouper(sfs, sum(processes.values())))
    t_chunks[-1] = [e for e in t_chunks[-1] if e]
    chunks = list(zip(*t_chunks))
    ret = {}
    i = 0
    for k, v in processes.items():
        ret[k] = [f for chunk in chunks[i:i + v] for f in chunk]
        i += v
    return ret


@task
def upload_input(inputs, remote_dir):
    """Uploads the input."""
    put(StringIO('\n'.join(inputs[env.host_string])),
        '{}/input.lst'.format(remote_dir))


def setup(remote_config, local_args):
    """
    Sets up the processing. Divides the input files among the hosts specified,
    creates the tmux sessions and virtualenvs on the remote machines, updating
    the code as necessary.
    """
    # Sets up the environment
    hosts = remote_config['Resources']['hosts']
    if not isinstance(hosts, list):
        hosts = [hosts]
    execute(setup_infrastructure, remote_config['Infrastructure'],
            local_args, hosts=hosts)
    remote_dir = remote_config['Infrastructure']['remote_dir']
    execute(setup_source, remote_config['Source'], remote_dir, hosts=hosts)
    source_dir = os.path.join(remote_dir, 'emLam')
    execute(setup_environment, remote_config['Environment'], remote_dir,
            source_dir, hosts=hosts)

    # The input files
    processes = execute(parallellism, remote_config['Resources']['num_cores'],
                        hosts=hosts)
    inputs = input_chunks(local_args, processes)
    execute(upload_input, inputs, remote_dir, hosts=hosts)


def do_it(remote_config, local_args):
    """Runs the jobs parallelly."""
    hosts = remote_config['Resources']['hosts']
    infconf = remote_config['Infrastructure']
    tid = remote_config['Environment']['tmux']
    execute(run, "tmux send -t {} 'cd {}' ENTER".format(
        tid, os.path.abspath(infconf['work_dir'])), hosts=hosts)
    execute(
        run, "tmux send -t {} 'python {} {} > {} 2>&1' ENTER".format(
            tid, os.path.join(infconf['remote_dir'], 'emLam',
                              'scripts', 'preprocess_corpus.py'),
            remote_config['cmd_line'],
            os.path.join(infconf['remote_dir'], 'emLam.log')),
        hosts=hosts
    )


def cleanup(remote_config, local_args):
    """
    Cleans up: removes the remote directory, uninstalls the Python packages
    and kills the tmux session.
    """
    hosts = remote_config['Resources']['hosts']
    infconf = remote_config['Infrastructure']
    envconf = remote_config['Environment']

    if infconf['delete_remote_dir']:
        execute(run, 'rm -rf {}'.format(infconf['remote_dir']), hosts=hosts)

    if not envconf['virtualenv'] and envconf['allow_user_packages']:
        execute(run, 'pip uninstall emLam virtualenv',
                warn_only=True, hosts=hosts)

    execute(run, 'tmux kill-session -t {}'.format(envconf['tmux']), hosts=hosts)
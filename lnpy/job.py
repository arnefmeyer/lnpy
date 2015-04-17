#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Cluster computing classes

    TODO: backend classes, e.g., GridEngineBackend and SlurmBackend

"""

import os
from os import makedirs
from os.path import join, exists, split, splitext
import subprocess
from shutil import copyfile
import itertools
import copy
from time import gmtime, strftime, time
import numpy as np
import multiprocessing as mp


def _detect_backend(order=['sbatch', 'qsub']):
    """automatically detect cluster backend"""

    infos = {'sbatch': 'SLURM', 'qsub': 'GridEngine'}
    for cmd in order:
        res = subprocess.Popen("command -v %s" % cmd, shell=True,
                               stdout=subprocess.PIPE).stdout.read()

        if cmd in res:
            return infos[cmd]

    return None


def _submit_job_parallel(job):

    job.submit()


class ClusterJob():
    """A simple job class that wraps GridEngine or SLURM command line arguments

    Parameters
    ----------
    script : str
        Path to the (python) script to be submitted

    arguments : list
        A list of arguments that will be written to the job file

    queue : str
        Name of the queue/partition

    name : str
        Name of the job

    tempdir : str
        Directory for temporary files

    copy : boolean
        Copy script?

    mem_request : int
        Memory request in MB

    time_request : int
        Time request in seconds

    compute_local : boolean
        Perform computations on local computer instead of submitting them
        to the cluster?

    n_workers : int
        Use n_workers when computing on local computer

    backend : str
        Enforce a specific backend

    env_vars : dict
        Dictionary with environment variables to be exported before calling
        the excecutive
    """

    def __init__(self, script, arguments='', queue='', name='', email=None,
                 tempdir='', copy=True, mem_request=None, time_request=None,
                 stdout='', stderr='', verbose=False, shell='python',
                 account=None, n_workers=1, compute_local=False, backend=None,
                 env_vars=None):

        self.script = script
        self.arguments = arguments
        self.queue = queue
        self.name = name
        self.email = email
        self.tempdir = tempdir
        self.copy = copy
        self.mem_request = mem_request
        self.time_request = time_request
        self.stdout = stdout
        self.stderr = stderr
        self.verbose = verbose
        self.n_workers = n_workers
        self.compute_local = compute_local
        self.backend = backend
        self.env_vars = env_vars
        self.account = account
        self.shell = shell

    def submit(self):
        """submit job using qsub or slurm backend"""

        sdir, sfile = split(self.script)
        if self.tempdir == '':
            self.tempdir = sdir
        elif not exists(self.tempdir):
            makedirs(self.tempdir)

        script_file = self.script
        if self.copy:
            name, ext = splitext(sfile)
            while True:
                tt = strftime('%Y_%m_%d_%H_%M_%S', gmtime())
                script_file = join(self.tempdir, '%s_copy_%s%s' %
                                   (name, tt, ext))
                if not exists(script_file):
                    break
            copyfile(self.script, script_file)

        backend = self.backend

        if backend is None:
            backend = _detect_backend()

        if self.compute_local or backend is None:
            print "Could not detect HPC backend." \
                  "Computing on local computer instead!"

            cmd = ''
            env_vars = self.env_vars
            if env_vars is not None:
                for val, key in enumerate(env_vars):
                    if isinstance(val, str):
                        cmd += "export %s=%s; " % (key, val)
                    elif isinstance(val, int):
                        cmd += "export %s=%d; " % (key, val)
                    elif isinstance(val, float):
                        cmd += "export %s=%f; " % (key, val)

            cmd += '{0} {1} {2}'.format(self.shell, self.script,
                                        self.arguments)
            os.system(cmd)

        else:

            if backend == 'GRIDENGINE':
                batch_file = join(self.tempdir, self.name + '.sge')
                self._to_sge_file(batch_file, script_file)
                cmd = 'qsub %s' % batch_file

            elif backend == 'SLURM':
                batch_file = join(self.tempdir, self.name + '.slurm')
                self._to_slurm_file(batch_file, script_file)
                cmd = 'sbatch %s' % batch_file

            if self.verbose:
                print 'Submitting job'
                print '  backend:', backend
                print '  temporary directory: %s' % self.tempdir
                print '  script file: %s' % script_file
                print '  arguments:', self.arguments

            os.system(cmd)

    def _to_sge_file(self, sge_file, script_file):
        """create sge file with job parameters"""

        with open(sge_file, "w") as f:
            f.write("#$ -S /bin/bash\n")
            if self.name != '':
                f.write("#$ -N %s\n" % self.name)
            if self.queue != '':
                f.write("#$ -q %s\n" % self.queue)
            if self.mem_request is not None:
                f.write("#$ -l h_vmem=%gM\n" % self.mem_request)
                if self.mem_request > 22*1024:
                    f.write("#$ -l bignode=true\n")
            if self.time_request is not None:
                f.write("#$ -l h_rt=%02d:%02d:%02d\n" %
                        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                               [(self.time_request,), 60, 60]))
            if self.stdout != "":
                f.write("#$ -o %s\n" % self.stdout)
            if self.stderr != "":
                f.write("#$ -e %s\n" % self.stderr)

            if self.n_workers is None or self.n_workers <= 0:
                f.write("#$ -pe smp %s\n" % 1)
            else:
                f.write("#$ -pe smp %s\n" % self.n_workers)

            f.write("#$ -wd %s\n" % os.getcwd())

            # Command to be executed
            f.write("export TERM=xterm;\n")

            env_vars = self.env_vars
            if env_vars is not None:
                for val, key in enumerate(env_vars):
                    if isinstance(val, str):
                        f.write("export %s=%s\n" % (key, val))
                    elif isinstance(val, int):
                        f.write("export %s=%d\n" % (key, val))
                    elif isinstance(val, float):
                        f.write("export %s=%f\n" % (key, val))

            f.write("%s %s %s\n" % (self.shell, script_file, self.arguments))
            f.write("exit\n")

    def _toqsubstring(self):
        """convert job parameters to qsub string"""

        opts = ''
        if self.name != '':
            opts += ' -N %s' % self.name
        if self.queue != '':
            opts += ' -q %s' % self.queue
        if self.mem_request is not None:
            opts += ' -l vf=%dM' % self.mem_request
        if self.time_request is not None:
            opts += ' -l h_rt=%02d:%02d:%02d' % \
                reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                       [(self.time_request,), 60, 60])
        if self.stdout != "":
            opts += ' -o %s' % self.stdout
        if self.stderr != "":
            opts += ' -e %s' % self.stderr
        return opts

    def _to_slurm_file(self, slurm_file, script_file):

        with open(slurm_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("#$ -S /bin/bash -l\n")

            if self.name != '':
                f.write("#SBATCH -J %s\n" % self.name)

            partition = self.queue
            if partition == '':
                partition = 'compute'
            f.write("#SBATCH -p %s\n" % partition)

            f.write("#SBATCH -N %d\n" % self.n_workers)

            if self.account is not None:
                f.write('#SBATCH -A %s' % self.account)

            if self.mem_request is not None:
                f.write("#SBTACH --mem-per-cpu=%d\n" % self.mem_request)

            if self.time_request is not None:
                f.write("#SBATCH --time=%02d:%02d:%02d\n" %
                        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                               [(self.time_request,), 60, 60]))
            if self.stdout != '':
                f.write("#SBATCH -o %s\n" % self.stdout)
            if self.stderr != '':
                f.write("#SBATCH -e %s\n" % self.stderr)

            if self.email is not None:
                f.write("#SBATCH --mail-user %s\n" % self.email)
                f.write("#SBATCH --mail-type=ALL\n")

            # Command to be executed
            f.write("export TERM=xterm;\n")

            env_vars = self.env_vars
            if env_vars is not None:
                for val, key in enumerate(env_vars):
                    if isinstance(val, str):
                        f.write("export %s=%s\n" % (key, val))
                    elif isinstance(val, int):
                        f.write("export %s=%d\n" % (key, val))
                    elif isinstance(val, float):
                        f.write("export %s=%f\n" % (key, val))

            f.write("%s %s %s\n" % (self.shell, script_file, self.arguments))
            f.write("exit\n")


class ClusterBatch():
    """Create a batch of cluster jobs"""

    def __init__(self, script, parameters, tempdir, template_job=None,
                 verbose=True, compute_local=False, n_parallel=-1,
                 **kwargs):

        self.script = script
        self.parameters = parameters
        self.tempdir = tempdir
        self.template_job = template_job
        self.verbose = verbose
        self.compute_local = compute_local
        self.n_parallel = n_parallel
        self.job_args = kwargs

    def submit(self):
        """dispatch all jobs for this batch"""

        # Create unique directory
        while True:
            tmpdir = join(self.tempdir,
                          strftime('%Y_%m_%d_%H_%M_%S', gmtime()))
            if not exists(tmpdir):
                makedirs(tmpdir)
                logdir = join(tmpdir, 'log')
                makedirs(logdir)
                break

        if self.verbose:
            print "temp. dir: %s" % tmpdir
            print "log dir: %s" % logdir

        # Create combinations of all parameters
        paramkeys = self.parameters.keys()
        values = []
        for k in paramkeys:
            values.append(self.parameters[k])
        params = list(itertools.product(*values))

        template_job = self.template_job
        if template_job is None:
            template_job = ClusterJob(self.script, **self.job_args)

        # Create one job for each parameter combination
        jobs = []
        n_jobs = len(params)
        if self.verbose:
            print "creating %d jobs ..." % n_jobs,
            t0 = time()

        for i in range(n_jobs):

            job_name = '%s_%d' % (template_job.name, i+1)

            # Copy script to temporary folder
            script_dir, script_file = split(self.script)
            script_name, script_ext = splitext(script_file)
            copy_name = '%s_%d' % (script_name, i+1)
            script_copy = join(tmpdir, copy_name + script_ext)
            copyfile(self.script, script_copy)

            # Save parameters with keys as numpy file
            npd = dict()
            for k, key in enumerate(paramkeys):
                npd.update({key: params[i][k]})

            npz_file = join(tmpdir, copy_name + '.npz')
            np.savez(npz_file, **npd)

            # Standard output and error log
            stdout = join(logdir, copy_name + '.o')
            stderr = join(logdir, copy_name + '.e')

            # Use template job as base
            job = copy.deepcopy(template_job)
            job.name = job_name
            job.arguments = npz_file
            job.tempdir = tmpdir
            job.copy = False
            job.script = script_copy
            job.stdout = stdout
            job.stderr = stderr
            job.compute_local = self.compute_local

            jobs.append(job)

        if self.verbose:
            print "done in %0.2f seconds" % (time() - t0)
            print "submitting jobs using qsub ...",
            t0 = time()

        if self.compute_local and self.n_parallel != 1:
            if self.n_parallel < 1:
                n_parallel = mp.cpu_count()
            else:
                n_parallel = self.n_parallel

            pool = mp.Pool(processes=n_parallel)
            pool.map(_submit_job_parallel, jobs)

        else:
            for job in jobs:
                job.submit()

        if self.verbose:
            print "done in %0.2f seconds" % (time() - t0)


if __name__ == '__main__':

    import sys
    from os.path import expanduser

    if len(sys.argv) > 1:
        data = np.load(sys.argv[1])
        a = data['a'].item()
        print "starting job", a

        X = np.random.randn(1000, 1000)
        whatever = np.linalg.svd(X)
        print "finished job", a

    else:
        user_dir = expanduser('~')
        test_dir = join(user_dir, 'Desktop', 'job_test')

        if not exists(test_dir):
            makedirs(test_dir)

        script_file = __file__
        params = {'a': range(10)}
        batch = ClusterBatch(script_file, params, test_dir, verbose=True,
                             compute_local=False, n_parallel=-1, queue='test',
                             mem_request=1000)
        batch.submit()

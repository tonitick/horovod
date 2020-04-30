import os
import sys
import json

from os.path import isfile

from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser(description='top on a set of hosts')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--start', dest='start', action='store_true', help='start top on the machines')
    group.add_argument('--stop', dest='stop', action='store_true', help='stop top on the machines')
    parser.add_argument('-o', '--output', dest='output', action='store', default='top', help='the path for top to log to on the machines. used when start')
    parser.add_argument('-m', '--machine_file', dest='workers', action='store', metavar='workers', default='workers', help='json file storing the hostnames')
    parser.add_argument('-i', '--interval', dest='interval', action='store', metavar='second(s)', default=1, type=float, help='top interval')
    return parser

def start_on_worker(worker_host, worker_path, is_stop, interval = 1):
    check_existing_top_cmd = 'ssh %s "cat top.pid 2>/dev/null"' % (worker_host)
    existing_pids = os.popen(check_existing_top_cmd).read().strip()

    if is_stop:
        if existing_pids == '':
            sys.stdout.write("top is not running on %s\n" % worker_host)
            return

        # Stop top
        existing_pids = existing_pids.split('\n')
        for pid in existing_pids:
            os.system('ssh %s "kill %s"' % (worker_host, pid))

        os.system('ssh %s "rm top.pid"' % (worker_host))
        sys.stdout.write("top on %s stopped.\n" % worker_host)
    else:
        # Check whether there's already a running Worker
        if existing_pids != '':
            sys.stdout.write("top on %s already started\n" % (worker_host))
            return

        # Create the output directory if it's not there
        os.system('ssh %s "mkdir -p %s"' % (worker_host, worker_path))

        # Start top
        pids = os.popen('ssh %s "ps -u $USER | awk \'\$4==\\"python\\" {print \$1}\'"'%worker_host).read().strip().split('\n')
        for pid in pids:
            if pid != str(os.getpid()):
                # os.system('ssh {0} "top -b -d {1} -p {2} | awk -v OFS=\'\t\' \'\$1+0>0 {{print systime(),\$9,\$10; fflush()}}\' > {3}/pid_{2} & echo \$! >> top.pid && wait \$!" &'.format(
                #     worker_host,
                #     interval,
                #     pid,
                #     worker_path
                # ))
                os.system('ssh {0} "collectl -sZ -i {1}:{1} --procfilt p{2} -f {3}/pid_{2} & echo \$! >> top.pid && wait \$!" &'.format(
                    worker_host,
                    interval,
                    pid,
                    worker_path
                ))

        sys.stdout.write("top started on %s.\n" % worker_host)

if __name__ == '__main__':
    args = get_parser().parse_args()

    # Read worker hosts
    if not isfile(args.workers):
        print("{0} file does not exit! Please give a worker file.".format(args.workers))
        exit(0)
    f = open(args.workers)
    lines = f.readlines()
    f.close()
    worker_hosts = set([worker_host.strip() for worker_host in lines])

    # Start
    if args.start:
        for worker_host in worker_hosts:
            start_on_worker(worker_host, args.output, False, args.interval)
    else:
        # Stop
        if args.stop:
            for worker_host in worker_hosts:
                start_on_worker(worker_host, args.output, True)

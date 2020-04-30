import json
import os.path
from pandas import concat
from pandas import read_csv
from pandas import merge_asof

from numpy import arange
from os import system as os_sys
from sys import argv
from sys import stderr
from sys import stdout

from datetime import datetime


# Arguments
from argparse import ArgumentParser
def parse_cmd(argv=None):
    parser = ArgumentParser(
        description='Plot CPU & network utilization from collectl and top logs.')
    parser.add_argument('--input-collectl',type=str, nargs='+', dest='input_c', help='The input collectl data directory or file.')
    parser.add_argument('--input-top', type=str, nargs='+', dest='input_t', help='The input top data directory or file.')
    parser.add_argument('-o','--output', type=str,dest='output', help='The output directory.')
    parser.add_argument('-r', '--raw', action='store_true', dest='raw_input',
                        help='Read raw collectl logs. Write preprocessed data to the dat dir under input dir.')
    parser.add_argument('--skip_header', dest='skip_header', type=int,metavar='n',
                        help='The number of lines to skip in the beginning. This overwrites values in conf.')
    parser.add_argument('-i', '--interval', dest='interval', action='store', metavar='second(s)', default=1, type=float, help='top interval')
    return parser.parse_args(argv)


class Conf:
    """ configurations for plotting """

    # default values
    def __init__(self, args = None):
        self.linewidth = 1
        self.max_net_rx_kb = 120832.0
        self.max_net_tx_kb = 120832.0
        self.max_disk_r_kb = 200000.0
        self.max_disk_w_kb = 200000.0
        self.scale_cpu = 1
        self.plot_average = 1

        # Default plot templates
        self.template = {
            "net": {
                # "figsize": (8, 6),
                "columns": ['[NET]RxKBTot', '[NET]TxKBTot'],
                "xlabel": 'Time(s)',
                "ylabel": 'KB'
            },
            "cpu": {
                # "figsize": (8, 6),
                "columns": ['PCT'],
                "xlabel": 'Time(s)',
                "ylabel": 'Utilization %'
            }
        }

        if args is not None:
            self.get_conf(args)

    def get_conf(self, args):
        self.interval = args.interval
        self.output = None
        self.xlim = None
        self.ylim = None

        self.input_c = args.input_c
        self.input_t = args.input_t
        self.output = args.output if args.output is not None else self.output
        if self.output is None:
            self.output = os.path.dirname(os.path.abspath(args.input_c[0])) + "/figs"
        # print("Outputs are put in {0}".format(self.output))
        assert(self.xlim is None or len(self.xlim) is 2)
        assert(self.ylim is None or len(self.ylim) is 2)

        # Retrieve collectl input files
        self.in_files_c = []
        for input in self.input_c:
            if os.path.isdir(input):
                self.in_files_c.extend([
                    os.path.join(input, f) for f in os.listdir(input)
                    if os.path.isfile(os.path.join(input, f))
                ])
            elif os.path.isfile(input):
                self.in_files_c.append(input)
            else:
                stderr.write("Input path %s does not exist\n" % input)
        if len(self.in_files_c) is 0:
            exit(0)

        # Retrieve top input files
        self.in_files_t = []
        for input in self.input_t:
            if os.path.isdir(input):
                self.in_files_t.extend([
                    os.path.join(input, f) for f in os.listdir(input)
                    if os.path.isfile(os.path.join(input, f))
                ])
            elif os.path.isfile(input):
                self.in_files_t.append(input)
            else:
                stderr.write("Input path %s does not exist\n" % input)
        if len(self.in_files_t) is 0:
            exit(0)


def preprocess(in_files, raw_input):
    preprocessed = []
    cmd = ""
    for in_file in in_files:
        out_file = os.path.basename(in_file).split('.')[0]
        cmd = "{3} echo 'Save preprocessed file {2} to {4}' && collectl -p {0} -P --sep 9 > {1}/{2} &".format(in_file, raw_input, out_file, cmd, os.path.relpath(raw_input))
        preprocessed.append(os.path.join(raw_input, out_file))

    os_sys("mkdir -p {0}".format(raw_input))
    cmd = "{0} wait".format(cmd)
    os_sys(cmd)
    return preprocessed

# No X11
from matplotlib import use as pltUse
pltUse('Agg')
from matplotlib import pyplot as plt

class Painter:
    def __init__(self, conf):
        self.conf = conf
        self.columns = ['']
        self.plot_conf = self.conf.template

    def plot_all(self):
        if len(self.conf.in_files_c or self.conf.in_files_t) is 0:
            return
        with open(self.conf.in_files_c[0]) as cur_f:
            columns = cur_f.readline().strip()[1:].split('\t')
        for in_file_c in self.conf.in_files_c:
            # load data
            data_c = read_csv(in_file_c, sep="\t", comment='#', names=columns)

            # convert data
            data_c['UTC'] = range(len(data_c))
            data_c = data_c.astype({'UTC': 'int64'})
            for i in range(len(data_c)):
                time_str = str(data_c.at[i, 'Date']) + ' ' + data_c.at[i, 'Time']
                time = datetime.strptime(time_str, "%Y%m%d %H:%M:%S")
                data_c.at[i, 'UTC'] = time.strftime("%s")
            # print(data_c['UTC'])
            data_c = data_c.astype({'UTC': 'float64'})
            lpos = 0
            while lpos < len(data_c):
                rpos = lpos + 1
                while rpos < len(data_c) and data_c.at[lpos, 'UTC'] == data_c.at[rpos, 'UTC']:
                    rpos = rpos + 1
                for i in range(rpos - lpos):
                    data_c.at[lpos + i, 'UTC'] = data_c.at[lpos + i, 'UTC'] + i * 1.0 / (rpos - lpos)
                
                lpos = rpos
            
            # print(data_c)
            
            with open(self.conf.in_files_t[0]) as cur_f:
                columns = cur_f.readline().strip()[1:].split('\t')
            for in_file_t in self.conf.in_files_t:
                # load data
                data_t = read_csv(in_file_t, sep="\t", comment='#', names=columns)

                # convert data
                data_t['UTC'] = range(len(data_t))
                data_t = data_t.astype({'UTC': 'int64'})
                for i in range(len(data_t)):
                    time_str = str(data_t.at[i, 'Date']) + ' ' + data_t.at[i, 'Time']
                    time = datetime.strptime(time_str, "%Y%m%d %H:%M:%S")
                    data_t.at[i, 'UTC'] = time.strftime("%s")
                # print(data_t['UTC'])
                data_t = data_t.astype({'UTC': 'float64'})
                lpos = 0
                while lpos < len(data_t):
                    rpos = lpos + 1
                    while rpos < len(data_t) and data_t.at[lpos, 'UTC'] == data_t.at[rpos, 'UTC']:
                        rpos = rpos + 1
                    for i in range(rpos - lpos):
                        data_t.at[lpos + i, 'UTC'] = data_t.at[lpos + i, 'UTC'] + i * 1.0 / (rpos - lpos)
                    
                    lpos = rpos
                
                data = merge_asof(data_t, data_c, on='UTC')
                print(data[['UTC'] + self.plot_conf['net']['columns'] + self.plot_conf['cpu']['columns']])
                
                # lpos = -1
                # for i in range(len(data_t)):
                #     if data_t.at[i, 'UTC'] != data_t.at[0, 'UTC']:
                #         lpos = i
                #         break
                # rpos = -1
                # for i in range(len(data_c)):
                #     if data_c.at[i, 'UTC'] == data_t.at[lpos, 'UTC']:
                #         rpos = i
                #         break
                # # data = data_c[(rpos - lpos):(rpos - lpos+len(data_t))][self.plot_conf['net']['columns']].reset_index(drop=True)
                # data = data_c[(rpos - lpos):(rpos - lpos + len(data_t))].reset_index(drop=True)
                # data['CPU%'] = data_t['CPU%']
                # data['MEM'] = data_t['MEM']
                # print(data)

                # plot
                output_file = os.path.join(self.conf.output, os.path.splitext(os.path.basename(in_file_c))[0] + os.path.splitext(os.path.basename(in_file_t))[0] + '.png')
                # self.plot_mix(data, output_file)

    def plot_mix(self, df, output_file):
        # print(df)
        fig, ax1 = plt.subplots(figsize=(32, 24))
        # plt.tick_params(labelsize=16)
            
        template = self.plot_conf['net']
        ax1.set_xlabel(template["xlabel"], fontsize=18)
        ax1.set_ylabel(template["ylabel"], fontsize=18)
        plt.tick_params(labelsize=16)
        if self.conf.xlim != None:
            ax1.set_xlim(self.conf.xlim)
        if self.conf.ylim != None:
            ax1.set_ylim(self.conf.ylim)
        df_to_plot = df[1000:1500][template['columns']]
        print("output len:")
        print(len(df_to_plot))
        print(len(arange(0, round(len(df_to_plot) * self.conf.interval, 2), self.conf.interval)))
        df_to_plot['Time(s)'] = arange(0, round(len(df_to_plot) * self.conf.interval, 2), self.conf.interval)
        df_to_plot.plot(x='Time(s)', linewidth=self.conf.linewidth, ax=ax1,
                color=['#00FFFF', '#008B8B'],
                # color='brg',
                #style=['-','--','-.']
                )
        
        ax2 = ax1.twinx()
        template = self.plot_conf['cpu']
        ax2.set_xlabel(template["xlabel"], fontsize=18)
        ax2.set_ylabel(template["ylabel"], fontsize=18)
        plt.tick_params(labelsize=16)
        if self.conf.xlim != None:
            ax2.set_xlim(self.conf.xlim)
        if self.conf.ylim != None:
            ax2.set_ylim(self.conf.ylim)
        df_to_plot = df[1000:1500][template['columns']]
        df_to_plot['Time(s)'] = arange(0, round(len(df_to_plot) * self.conf.interval, 2), self.conf.interval)
        df_to_plot.plot(x='Time(s)', linewidth=self.conf.linewidth, ax=ax2,
                color=['#800000'],
                # color='brg',
                #style=['-','--','-.']
                )        
        
        
        
        lgd = plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.1, prop={'size': 18})
        fig.tight_layout()
        plt.savefig(output_file, bbox_extra_artists=(lgd, ), bbox_inches='tight')
        print("Saved fig {0}".format(output_file))
        plt.close()


def main(args):
    print("args:")
    print(args)
    conf = Conf(args)
    print("conf:")
    print(conf.in_files_c)
    print(conf.in_files_t)

    if args.raw_input:
        conf.in_files_c = preprocess(conf.in_files_c, os.path.dirname(os.path.abspath(args.input_c[0])) + "/dat")
        conf.in_files_t = preprocess(conf.in_files_t, os.path.dirname(os.path.abspath(args.input_t[0])) + "/dat")

    # Create output dir in case it does not exist
    os_sys('mkdir -p %s' % conf.output)

    Painter(conf).plot_all()


if __name__ == "__main__":
    args = parse_cmd(argv[1:])
    main(args)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

BAR_WIDTH = 0.35       # the width of the bars


def plot_cpu(df_docker_cpu, df_vms_cpu, nf_names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(11)  # the x locations for the groups
    docker_plot = ax.bar(ind, df_docker_cpu['cpu_mean'], BAR_WIDTH, color='royalblue', yerr=df_docker_cpu['cpu_err'], capsize=3)
    vm_plot = ax.bar(ind+BAR_WIDTH, df_vms_cpu['cpu_mean'], BAR_WIDTH, color='seagreen', yerr=df_vms_cpu['cpu_err'], capsize=3)
    ax.set_ylabel('CPU [%]')
    ax.set_xlabel('Funkcje sieciowe')
    ax.set_title('Zużycie procesora przez funkcje sieciowe w stanie bezczynności')
    ax.set_xticks(ind + BAR_WIDTH / 2)
    ax.set_xticklabels(nf_names)
    ax.legend((docker_plot[0], vm_plot[0]), ('docker', 'kvm'))
    plt.savefig('idle_cpu_usage.png')
    plt.show()


def plot_mem(df_docker_mem, df_vms_mem, nf_names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(11)  # the x locations for the groups
    docker_plot = ax.bar(ind, df_docker_mem['mem_mean'], BAR_WIDTH, color='royalblue', yerr=df_docker_mem['mem_err'], capsize=3)
    vm_plot = ax.bar(ind+BAR_WIDTH, df_vms_mem['mem_mean'], BAR_WIDTH, color='seagreen', yerr=df_vms_mem['mem_err'], capsize=3)
    ax.set_ylabel('RSS [MiB]')
    ax.set_xlabel('Funkcje sieciowe')
    ax.set_title('Zużycie pamięci przez funkcje sieciowe w stanie bezczynności')
    ax.set_xticks(ind + BAR_WIDTH / 2)
    ax.set_xticklabels(nf_names)
    ax.legend((docker_plot[0], vm_plot[0]), ('docker', 'kvm'))
    plt.savefig('idle_mem_usage.png')
    plt.show()


df_vms = pd.read_csv('mean_idle_vms.csv')
df_vms_mem = pd.read_csv('mem_vms_idle.csv')
df_docker = pd.read_csv('mean_idle_docker.csv')
# in order to combine both df we need to do some processing to match
nf_names = ['amf', 'ausf', 'bsf', 'db',
            'nrf', 'nssf', 'pcf', 'smf',
            'udm', 'udr', 'upf']
df_docker['nf_name'] = nf_names
df_docker_cpu = df_docker[['nf_name', 'cpu_mean', 'cpu_ci_lower', 'cpu_ci_upper', 'cpu_err']]
df_vms_cpu = df_vms[['Domain name_', '%CPU_mean', '%CPU_ci_lower', '%CPU_ci_upper', '%CPU_err']]
df_vms_cpu.rename(columns={
    'Domain name_': 'nf_name',
    '%CPU_mean': 'cpu_mean',
    '%CPU_ci_lower': 'cpu_ci_lower',
    '%CPU_ci_upper': 'cpu_ci_upper',
    '%CPU_err': 'cpu_err'
}, inplace=True)
df_vms_cpu['nf_name'] = nf_names


df_docker_mem = df_docker[['nf_name', 'mem_mean', 'mem_ci_lower', 'mem_ci_upper', 'mem_err']]
plot_mem(df_docker_mem, df_vms_mem, nf_names)
plot_cpu(df_docker_cpu, df_vms_cpu, nf_names)

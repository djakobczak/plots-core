from turtle import color
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

import utils

# static data
Gi2G = 1.07374

# base img + builder (MiB)
vm_img = [541, 1710]
vm_img = list(map(lambda el: el*Gi2G, vm_img))
# vm_img += [sum(vm_img)]

# base img + docker_open5gs (MB)
docker_img = [72.78, 484.6]
# docker_img += [sum(docker_img)]
index = ['Ubuntu20', '5gs']

base_img_size = [72.78, 541*Gi2G]
builder_img_size = [484.6, 1710*Gi2G]


df = pd.DataFrame(
    {
        'docker': docker_img,
        'kvm': vm_img,
    },
    index=index
)

# index = ['kvm', 'docker']
# df = pd.DataFrame(
#     {
#         'Ubuntu20': base_img_size,
#         '5gs': builder_img_size,
#     },
#     index=index
# )
matplotlib.rcParams.update({'font.size': 24})
# ax = df.plot.bar(rot=0, color=utils.COLORS)
idx = [0, 0.7]
fig, ax = plt.subplots(figsize=(12,10))
ax.bar(idx, base_img_size, width=0.45, color='orangered')
ax.bar(idx, builder_img_size, width=0.45, bottom=base_img_size, color='skyblue')
# ax.set_title('Porównanie rozmiaru obrazu funkcji sieciowej')
ax.set_xlabel('Platforma wirtualizacji')
ax.set_ylabel('Rozmiar [MB]')
ax.set_xticks(idx)
ax.set_xticklabels(['docker', 'kvm'])
ax.set_xlim([-0.4, 1.1])
ax.legend(['warstwa bazowa', 'pozostałe warstwy'])
plt.savefig('image_size.png')
plt.show()

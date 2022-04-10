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

base_img_size = [541*Gi2G, 72.78]
builder_img_size = [1710*Gi2G, 484.6]


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

ax = df.plot.bar(rot=0, color=utils.COLORS)
ax.set_title('Porównanie rozmiaru obrazu funkcji sieciowej')
ax.set_xlabel('Warstwa')
ax.set_ylabel('Rozmiar [MB]')
plt.savefig('image_size.png')
plt.show()

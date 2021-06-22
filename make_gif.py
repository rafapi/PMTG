import imageio
import os


gif_name = "infinity"

print('creating gif\n')
all_files = sorted(os.listdir('images'),
                   key=lambda x: int(x.partition('_')[2].partition('.')[0]))
last = all_files[-1]
with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
    for filename in all_files:
        image = imageio.imread(f'images/{filename}')
        print(f'Processing: {filename}')
        if filename == last:
            for _ in range(10):
                writer.append_data(image)
        writer.append_data(image)
print('gif complete\n')

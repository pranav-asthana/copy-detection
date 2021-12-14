import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import os
import random
import augly.image as imaugs

def select_images():
    with open('./list_files/references') as r:
        with open('./list_files/subset_1_references') as s:
            total = r.readlines()
            sub = s.readlines()

            diffs = [item for item in total if item not in sub]

            with open('./list_files/augmented_references', 'w') as d:
                d.writelines(diffs[:10001])
            
def load_images():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    with open('./list_files/augmented_references') as ar:            
            refs = ar.readlines()
            for line in refs:
                line = line.strip()
                print(line)
                file_name = '.'.join((line, 'jpg'))
                print(file_name)
                bucket = 'drivendata-competition-fb-isc-data'
                resource = f'all/reference_images/{file_name}' 
                target = f'./images/aug_ref/{file_name}'
                print(resource)
                print(target)
                s3.download_file(bucket, resource, target)

def augly_image(input_image, output_image):
    aug_image = input_image
    if random.randint(0,1):
        aug_image = imaugs.overlay_emoji(aug_image, opacity=random.random(), emoji_size=random.random())
    if random.randint(0,1):
        aug_image = imaugs.pad_square(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.overlay_onto_screenshot(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.change_aspect_ratio(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.convert_color(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.hflip(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.crop(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.color_jitter(aug_image, brightness_factor=random.random(), contrast_factor=random.random(), saturation_factor=random.random())
    if random.randint(0,1):
        aug_image = imaugs.vflip(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.saturation(aug_image, factor=random.random())
    if random.randint(0,1):
        aug_image = imaugs.random_noise(aug_image)
    if random.randint(0,1):
        aug_image = imaugs.overlay_emoji(aug_image, opacity=random.random(), emoji_size=random.random())
    if random.randint(0,1):
        aug_image = imaugs.perspective_transform(aug_image)
    
    aug_image = aug_image.save(output_image)

def augment_queryset():
    with open('./list_files/augmented_ground_truth.csv', 'w') as agt:
        with open('./list_files/augmented_references') as ar:
            agt.write('query_id,reference_id')
            agt.write('\n')
            refs = ar.readlines()
            i = 25000
            for line in refs:
                line = line.strip()
                full_name = 'Q' + str(i)
                i += 1

                infile_name = '.'.join((line, 'jpg'))
                outfile_name = '.'.join((full_name, 'jpg'))

                input_image = f'./images/aug_ref/{infile_name}'
                output_image = f'./images/aug_queries/{outfile_name}'

                augly_image(input_image, output_image)
                agt.write(f'{full_name},{line}')
                agt.write('\n')
                print(output_image)


if __name__ == '__main__':
    augment_queryset()

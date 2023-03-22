# Download CompCars
import os
import time


def download_compcars():
    # Download CompCarsWeb
    compcarsweb_filename = [
        'data.zip',
        'data.z01', 'data.z02', 'data.z03', 'data.z04', 'data.z05',
        'data.z06', 'data.z07', 'data.z08', 'data.z09', 'data.z10',
        'data.z11', 'data.z12', 'data.z13', 'data.z14', 'data.z15',
        'data.z16', 'data.z17', 'data.z18', 'data.z19', 'data.z20',
        'data.z21', 'data.z22'
    ]
    compcarsweb_urls = [
        'https://drive.google.com/file/d/1T9K-7-K2bHFepntxIz4guYaqPfRWYW8p/view?usp=share_link',
        'https://drive.google.com/file/d/1jRgYO7KaX08BjPWpO5m4yjnli9KT40H2/view?usp=share_link',
        'https://drive.google.com/file/d/1wMoVAd2UniiC1k94EVac2MtwJRAlaqEg/view?usp=share_link',
        'https://drive.google.com/file/d/1x8rfnd48g6YeIB2sYZlKR9yUWq0EVTXN/view?usp=share_link',
        'https://drive.google.com/file/d/1J1UKu30fg5DoNq7PIGJDPTsNRccpeAO0/view?usp=share_link',
        'https://drive.google.com/file/d/1pHyr3RmgNSqgS1FJyyYI8Eko6kgzA8yP/view?usp=share_link',
        'https://drive.google.com/file/d/15cQhDjUILTP5JOijWInzN9mtuLSZOTCv/view?usp=share_link',
        'https://drive.google.com/file/d/1Qy9FrgJT-N33HXKrwPAhDe7cg2Uwd95z/view?usp=share_link',
        'https://drive.google.com/file/d/1nd3J1WZ81eaJuNw8LJ_-Azouo7uO8221/view?usp=share_link',
        'https://drive.google.com/file/d/1Td81PVNWAH_LEgjPX3bZiH3-OzwVAfSK/view?usp=share_link',
        'https://drive.google.com/file/d/1i6szbo8UYNmuoA3eLwUd5RyGV6l5C-qt/view?usp=share_link',
        'https://drive.google.com/file/d/1QHGbUiU5CwgSAppJFLpVrTlLm8vI9dK9/view?usp=share_link',
        'https://drive.google.com/file/d/1HfW88qqiuKBQWRnZJE9yWMU7_K3Rnr0_/view?usp=share_link',
        'https://drive.google.com/file/d/1la7Jt8ti2yxRtUG_4YC4zWPId37ja5jF/view?usp=share_link',
        'https://drive.google.com/file/d/1Fo1xpfuRWcLk4Yt-bdFMgmW99e3X1SBM/view?usp=share_link',
        'https://drive.google.com/file/d/1WTajdHaJ0sYo2gBOvCAtWNoM9S9W7aYv/view?usp=share_link',
        'https://drive.google.com/file/d/1BBM7fT0rs_-S15llKPwV2uv7K8lVpG5X/view?usp=share_link',
        'https://drive.google.com/file/d/1HMxnsz5-u4OM9UgLM30PLQYGIl4bwC_L/view?usp=share_link',
        'https://drive.google.com/file/d/1WfENLN1QsmHNTBZjqNkRp9g7hN1tv1wq/view?usp=share_link',
        'https://drive.google.com/file/d/1bweCNot6B8wk6KhcnDmdaguxRD7kxEIp/view?usp=share_link',
        'https://drive.google.com/file/d/1ewPYhpmJY9Ug-GzKZJ_B6eSEsnnoaXeK/view?usp=share_link',
        'https://drive.google.com/file/d/1i9nsf6g_jV46vMaz8ZqshG6vJOJFvDDL/view?usp=share_link',
        'https://drive.google.com/file/d/1Hq53fIctob6Dh3TiyDvNkWI10Z-pDruV/view?usp=share_link',
    ]
    
    # Create /data dir to store images here
    os.system('sudo mkdir /data')
    os.system('sudo chown jovyan /data')
    os.system('sudo chgrp jovyan /data')
    
    # Start downloading
    # for filename, url in zip(compcarsweb_filename, compcarsweb_urls):
    #     fileid = url.split('/')[5]
    #     download_command = f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fileid}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={fileid}\" -O /data/{filename} && rm -rf /tmp/cookies.txt"
    #     os.system(download_command)
    
    # Waiting for all download to complete
    # while not os.path.exists('/data/data.z22'):
    #     time.sleep(5)
    # os.system('zip -F /data/data.zip --out /data/combined.zip')
    # while not os.path.exists('/data/combined.zip'):
    #     time.sleep(5)
    # os.system('unzip -P d89551fd190e38 /data/combined.zip')
    # time.sleep(2)

    # Download class mapping file
    # class_mapping_filename = [
    #     'class_mapping_make_webnature.npy',
    #     'class_mapping_model_webnature.npy',
    # ]
    # class_mapping_url = [
    #     'https://drive.google.com/file/d/11rk-ZyuwswfTcGudr331XsNnFnHWlsdV/view?usp=share_link',
    #     'https://drive.google.com/file/d/1IRLoDZXbBcoK864uKIDxuWa5TIaxCOfX/view?usp=share_link',
    # ]
    # for filename, url in zip(class_mapping_filename, class_mapping_url):
    #     fileid = url.split('/')[5]
    #     download_command = f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fileid}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={fileid}\" -O /data/{filename} && rm -rf /tmp/cookies.txt"
    #     os.system(download_command)
    #     time.sleep(2)
    
    # Download CompCarsSV
    compcarssv_filename = [
        'sv_data.zip',
        'sv_data.z01', 'sv_data.z02', 'sv_data.z03', 
    ]
    compcarssv_urls = [
        'https://drive.google.com/file/d/1fXAIAusPmSdnaO6IQgwzIp6BNPgTo0kv/view?usp=share_link',
        'https://drive.google.com/file/d/1KK5802tW2fXJN-q9rj2yrJ8hZASItXe_/view?usp=share_link',
        'https://drive.google.com/file/d/13s3RUeD4xsC3N2My8YxTkqymEAo_mmdt/view?usp=share_link',
        'https://drive.google.com/file/d/1pTaaCSQ7KQxlFyoqwP7DUizbQFuopW_y/view?usp=share_link',
    ]
    for filename, url in zip(compcarssv_filename, compcarssv_urls):
        fileid = url.split('/')[5]
        download_command = f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fileid}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={fileid}\" -O /data/{filename} && rm -rf /tmp/cookies.txt"
        # os.system(download_command)
        print(download_command)
    
    # Waiting for all download to complete
    # time.sleep(10)
    # os.system('zip -F /data/sv_data.zip --out /data/sv_combined.zip')
    # time.sleep(2)
    # os.system('unzip -P d89551fd190e38 /data/sv_combined.zip')
    # time.sleep(2)
    
    print('Download for CompCars completed')
    return None


def download_stanford_car():
    os.system('wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz -P /data/stanford-car-classification')
    os.system('wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat -P /data/stanford-car-classification')
    os.system('wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -P /data/stanford-car-classification')
    
    os.system('tar -C stanford-car-classification/data -xzvf /data/stanford-car-classification/car_ims.tgz')
    os.system('tar -C stanford-car-classification/data -xzvf /data/stanford-car-classification/car_devkit.tgz')
    
    return None


def download_mohsin_vmmr():
    filename = 'mohsin_vmmr.zip'
    url = 'https://drive.google.com/file/d/1x7WBOOzUcAicVDbexHtK0prx9GWi0zoK/view?usp=sharing'
    fileid = url.split('/')[5]
    download_command = f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fileid}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={fileid}\" -O /data/{filename} && rm -rf /tmp/cookies.txt"
    os.system(download_command)
    os.system('unzip /data/mohsin_vmmr.zip')
    print('Download for mohsin-vmmr completed')
    return None
    
download_compcars()
# download_stanford_car()
# download_mohsin_vmmr()

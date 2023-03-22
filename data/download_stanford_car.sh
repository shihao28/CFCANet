sudo mkdir /data
sudo chown jovyan /data
sudo chgrp jovyan /data
wget https://ai.stanford.edu/~jkrause/car196/cars_train.tgz -P /data/stanford-car
wget https://ai.stanford.edu/~jkrause/car196/cars_test.tgz -P /data/stanford-car
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -P /data/stanford-car
tar -C /data/stanford-car -xzvf /data/stanford-car/cars_train.tgz
tar -C /data/stanford-car -xzvf /data/stanford-car/cars_test.tgz
tar -C /data/stanford-car -xzvf /data/stanford-car/car_devkit.tgz
wget https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat -P /data/stanford-car/devkit
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FE4Y2jIGJRnNiPTnYJ8im1H7vW1o28Cp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*//p')&id=1FE4Y2jIGJRnNiPTnYJ8im1H7vW1o28Cp" -O /data/stanford-car/my_stanfordcar_annotation.zip && rm -rf /tmp/cookies.txt
unzip /data/stanford-car/my_stanfordcar_annotation.zip -d /data/stanford-car
cp /data/stanford-car/my_stanfordcar_annotation/vmmr_to_vtr.json /data/stanford-car/devkit
cp /data/stanford-car/my_stanfordcar_annotation/*.npy /data/stanford-car
cp /data/stanford-car/my_stanfordcar_annotation/car_poor_images_train.txt /data/stanford-car

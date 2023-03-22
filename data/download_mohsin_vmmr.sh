sudo mkdir /data
sudo chown jovyan /data
sudo chgrp jovyan /data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1x7WBOOzUcAicVDbexHtK0prx9GWi0zoK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1x7WBOOzUcAicVDbexHtK0prx9GWi0zoK" -O /data/mohsin_vmmr.zip && rm -rf /tmp/cookies.txt
unzip /data/mohsin_vmmr.zip -d /data

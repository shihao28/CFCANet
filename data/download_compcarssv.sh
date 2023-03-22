sudo mkdir /data
sudo chown jovyan /data
sudo chgrp jovyan /data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fXAIAusPmSdnaO6IQgwzIp6BNPgTo0kv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*//p')&id=1fXAIAusPmSdnaO6IQgwzIp6BNPgTo0kv" -O /data/sv_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KK5802tW2fXJN-q9rj2yrJ8hZASItXe_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*//p')&id=1KK5802tW2fXJN-q9rj2yrJ8hZASItXe_" -O /data/sv_data.z01 && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13s3RUeD4xsC3N2My8YxTkqymEAo_mmdt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*//p')&id=13s3RUeD4xsC3N2My8YxTkqymEAo_mmdt" -O /data/sv_data.z02 && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pTaaCSQ7KQxlFyoqwP7DUizbQFuopW_y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*//p')&id=1pTaaCSQ7KQxlFyoqwP7DUizbQFuopW_y" -O /data/sv_data.z03 && rm -rf /tmp/cookies.txt
zip -F /data/sv_data.zip --out /data/combined_sv.zip
unzip -P d89551fd190e38 /data/combined_sv.zip -d /data/CompCars
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xVU2vN7CnQszCpJvKMgEB3wkCvzZylbU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*//p')&id=1xVU2vN7CnQszCpJvKMgEB3wkCvzZylbU" -O /data/CompCars/sv_data/my_compcarssv_annotation.zip && rm -rf /tmp/cookies.txt
unzip /data/CompCars/sv_data/my_compcarssv_annotation.zip -d /data/CompCars/sv_data
cp /data/CompCars/sv_data/my_compcarssv_annotation/class_mapping_make_sv.npy /data/CompCars/sv_data/class_mapping_make_sv.npy
cp /data/CompCars/sv_data/my_compcarssv_annotation/class_mapping_model_sv.npy /data/CompCars/sv_data/class_mapping_model_sv.npy

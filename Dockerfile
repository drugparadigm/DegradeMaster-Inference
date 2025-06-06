FROM continuumio/miniconda3

WORKDIR /usr/src/app

COPY . .

RUN apt-get update &&\
        apt-get install -y tzdata && \
        ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime &&  \
        echo "Asia/Kolkata" > /etc/timezone
 
ENV TZ=Asia/Kolkata

# The conda environment should be from the dgx (!!NOT from your PC)

RUN conda env create -f environment.yaml

SHELL ["/bin/bash", "-c"]

RUN conda init bash

RUN chmod +x additional-softwares.sh

# Replace <env-name> with the name of your conda environment
RUN /bin/bash -c "source activate model-host-degrademaster && ./additional-softwares.sh"

ENV FLASK_APP=api.py

ENV FLASK_ENV=production

EXPOSE 5000

# Replace <env-name> with the name of your conda environment
# CMD [ "bash", "-lc", "source activate model-host-degrademaster &&  flask run --host=0.0.0.0 --port=5000" ]
CMD [ "bash", "-lc", "source activate model-host-degrademaster &&  exec gunicorn api:app -b 0.0.0.0:5000 --workers=1 --threads=5  --access-logfile -" ]
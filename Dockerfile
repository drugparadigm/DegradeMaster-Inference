FROM continuumio/miniconda3

WORKDIR /usr/src/app

COPY . .

# The conda environment should be from the dgx (!!NOT from your PC)

RUN conda env create -f environment.yaml

SHELL ["/bin/bash", "-c"]

RUN conda init bash

RUN chmod +x additional-softwares.sh

# Replace <env-name> with the name of your conda environment
RUN /bin/bash -c "source activate model-host-degrademaster && ./additional-softwares.sh"

EXPOSE 5000

# Replace <env-name> with the name of your conda environment
CMD ["bash", "-c", "source activate model-host-degrademaster && python api.py"]

.PHONY : all
all : 
	jupyter execute main.ipynb
    

.PHONY : env
env :
	mamba env create -f environment.yml -p ~/envs/hw7_env
	bash -ic 'conda activate hw7_env;python -m ipykernel install --user --name hw7_env --display-name "IPython - hw7_env"'
    
    
.PHONY : hw7_tools
hw7_tools :
	cd hw7_tools; pip install .


.PHONY : clean
clean :
	rm -f output/*.png
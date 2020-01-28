conda create --name drlnd python=3.6  
	source activate drlnd  
	git clone https://github.com/openai/gym.git  
	cd gym  
	pip install -e .  
	cd ..  
	git clone https://github.com/udacity/deep-reinforcement-learning.git  
	cd deep-reinforcement-learning/python  
	pip install .  
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
# wordle-solver

Check out blog posts about this here: https://andrewkho.github.io/wordle-solver/

# Dev instructions

Env setup
```
python3 -m venv server-env
source server-env/bin/activate
pip install -r requirements.txt
```

To update requirements
```
pip install -r server-requires.txt
pip freeze > requirements.txt
# The first line of requirements.txt needs to be 
# -f https://download.pytorch.org/whl/torch_stable.html
# and then the torch==1.8.1+cpu in order to keep the
# heroku slug-size below 500mb
```

Local dev
```
# Start server
gunicorn --pythonpath deep_rl app:app

# Start react dev server
npm run start-local
```

For pre-deploy local testing
```
npm run build-local # Build static site
heroku local # Site should be available at localhost:5000
```
Local testing looks for the pre-trained model at `data/checkpoints/a2c_deployed.ckpt` so create a symlink there or something.

To deploy
```
git push heroku master # Deploys site
```

To deploy new Lightning model/checkpoint
```
./deploy_checkpoint.sh <path_to_checkpoint>
```

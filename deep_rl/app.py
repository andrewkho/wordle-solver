import os

import flask

import a2c.play

AGENT = None
ENV = None

S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', '')
CHECKPOINT_PATH = 'checkpoints/a2c_deployed.ckpt'


app = flask.Flask(__name__, static_folder='../build/', static_url_path='/')
app.debug = 'DEBUG' in os.environ

if not S3_BUCKET_NAME:
    # We are probably running locally so enable cors
    from flask_cors import CORS
    CORS(app)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    return app.send_static_file('index.html')


@app.errorhandler(404)
def not_found(e):
  return app.send_static_file('index.html')


@app.route('/api/hello', methods=['GET'])
def hello():
    return {'msg': 'Hello world!'}


def _word_is_valid(word: str) -> bool:
    if len(word) != 5 or not word.isalpha():
        return False
    return True


def _validate_mask(mask: str) -> bool:
    if len(mask) != 5 or not all(i in '012' for i in mask):
        return False
    return True


@app.route('/api/wordle-goal/<goal_word>', methods=['GET'])
def wordle_goal(goal_word: str):
    if not _word_is_valid(goal_word):
        return {"msg": "word is invalid!"}, 400

    try:
        if AGENT is None or ENV is None:
            return "Trouble loading model, maybe try again later?", 503

        win, outcomes = a2c.play.goal(AGENT, ENV, goal_word)
    except Exception as e:
        return str(e), 403
    return {
        "win": win,
        "guesses": [guess for guess, _ in outcomes],
        "rewards": [reward for _, reward in outcomes],
    }


@app.route('/api/wordle-suggest', methods=['GET'])
def suggest():
    words = flask.request.args['words'].split(',')
    masks = flask.request.args['masks'].split(',')
    if len(words) == 1 and len(words[0].strip()) == 0:
        words = []
    if len(masks) == 1 and len(masks[0].strip()) == 0:
        masks = []

    if len(words) > 6 or any(not _word_is_valid(w) for w in words):
        return {"msg": "words are invalid!"}, 400
    if len(masks) != len(words) or any(not _validate_mask(m) for m in masks):
        return {"msg": "masks are invalid!"}, 400

    seq = [
        (word, [int(i) for i in mask])
        for word, mask in zip(words, masks)
    ]

    try:
        if AGENT is None or ENV is None:
            return {"msg": "Trouble loading model, maybe try again later?"}, 503

        suggestion = a2c.play.suggest(AGENT, ENV, sequence=seq)
    except Exception as e:
        print("Caught exception", str(e))
        return str(e), 403

    return {
        "suggestion": suggestion,
    }


def _startup():
    global AGENT, ENV

    if not S3_BUCKET_NAME:
        # Assume we're local
        url = f'data/{CHECKPOINT_PATH}'
    else:
        url = f's3://{S3_BUCKET_NAME}/{CHECKPOINT_PATH}'
    print(f"Startup: Loading checkpoint from {url}...")
    _, AGENT, ENV = a2c.play.load_from_checkpoint(url)
    print("done!")
    print("Mask Based State Updates:", ENV.mask_based_state_updates)


_startup()

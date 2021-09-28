# Import Libraries
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, flash, send_file
from tensorflow.keras.models import load_model
import os, re
import shutil, tempfile, weakref


# Do not show unnecessary warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# USE FULL POWER OF GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)


### Load Names Data

with open("data/us_names_utf_8", "rb") as f:
    names = [name.decode("utf-8") for name in f.readlines()]


### Get info from Names Data

# create names set
names_set = set([name.strip() for name in names])

# add start(<) and end(>) tokens
for i in range(len(names)):
    names[i] = "<" + names[i].strip() + ">"

## create dicts char to index and vice versa

# get all chars
all_chars_set = set("".join(names))
all_chars = sorted(all_chars_set)

vocab_size = len(all_chars) + 1 # +1 for empty character (padding)

# lookup dicts
char_to_index = dict([(ch, ix+1) for ix, ch in enumerate(all_chars)])
index_to_char = dict([(ix, ch) for ch, ix in char_to_index.items()])

# max sequence len
max_len = max([len(name) for name in names])

## Seq to Name and Name to Seq
def name_to_seq(name):
    return [char_to_index[ch] for ch in name]

def seq_to_name(seq):
    return "".join([index_to_char[i] for i in seq])

### Load the inference model

# Model Settings
embedding_dim = 64
# Load the Model
inference_model = load_model("model/us_name_generator_inference_model.h5")


### Generate Names

## Helper Functions

def generate_names(seed="", num_names=1):
    """Generate chars given a seed (it can be an empty string)
    
    Predicts next chars according to previous chars in a loop.
    Next chars are chosen using probabilities to create different names each time.
    
    Parameters:
        seed (string) = "":
            initial characters of the wanted name
        num_names (integer) = 1:
            number of names to generate
    
    Returns:
        (list of strings):
            generated names
    """
    
    # add start character (<) to seeds
    seeds = ["<" + seed for _ in range(num_names)]
    outputs = seeds.copy()
    
    # create initial states
    h_state = tf.zeros(shape=(num_names, embedding_dim))
    c_state = tf.zeros(shape=(num_names, embedding_dim))
    states = [h_state, c_state]
    
    # generation loop
    for _ in range(17 - len(seed)):
        # convert text seeds to model input
        seqs = np.array([name_to_seq(seed) for seed in seeds])
        
        # predict next char
        probs, h_state, c_state = inference_model([seqs] + states)
        states = [h_state, c_state]
        probs = np.asarray(probs)[:, -1, :]
        
        # choose next chars using predicted probabilities
        for i in range(num_names):
            index = np.random.choice(list(range(vocab_size)), p=probs[i].ravel())
        
            if index == 0:
                index = char_to_index["<"]
            
            pred_char = index_to_char[index]
            seeds[i] = pred_char
            outputs[i] += pred_char
        
        # if all predicted chars are end char (>), end the generation
        if all([output[-1] == ">" for output in outputs]):
            break
            
    # process outputs
    for i, output in enumerate(outputs):
        name = re.findall("<[a-z]+>", output)
        if name:
            # get rid of start(<) and end(>) chars
            outputs[i] = name[0].strip("<>")
        else:
            outputs[i] = ""
            
    return outputs

def is_real_name(name):
    """Checks whether created name is in names dataset or not"""
    return name.strip("<> ") in names_set

def generate_artificial_names(seed="", num_names=1):
    """Generates artificial English names.
    Generated names will not be real names.
    
    Parameters:
        seed (string) = "":
            initial characters of the wanted name
        num_names (integer) = 1:
            number of names to generate
    
    Returns:
        (list of strings):
            generated names
    """
    generated_names = []
    
    stop = False
    while not stop:
        # generate names more than needed as some names may exist in real life
        num_needed_names = (num_names - len(generated_names)) * 3 // 2
        names = generate_names(seed=seed, num_names=num_needed_names)
        
        # check whether names are in dataset or not
        for name in names:
            if not is_real_name(name):
                generated_names.append(name)
                if len(generated_names) == num_names:
                    stop = True
                    break
    
    return generated_names

def is_seed_valid(seed):
    """Checks whether a given seed is valid or not"""
    for ch in seed:
        if not ch in all_chars_set:
            return False
    return True

# Removing temporary file after the download
class FileRemover(object):
    def __init__(self):
        self.weak_references = dict()  # weak_ref -> filepath to remove

    def cleanup_once_done(self, response, filepath):
        wr = weakref.ref(response, self._do_cleanup)
        self.weak_references[wr] = filepath

    def _do_cleanup(self, wr):
        filepath = self.weak_references[wr]
        print('Deleting %s' % filepath)
        shutil.rmtree(filepath, ignore_errors=True)

file_remover = FileRemover()

# dummy prediction to initialize tensorflow
# model's first prediction will be so slow without dummy prediction
generate_names("aaaaaaaaaaaa")

# Flask Functions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    ## preprocess given features
    features = list(request.form.values())
    are_inputs_valid = True

    # process inital characters
    seed = features[0].strip().lower()

    if not is_seed_valid(seed):
        flash("Please type alphabetical character(s) as initial characters of names.")
        are_inputs_valid = False

    # process number of names
    num_names = features[1].strip()

    if num_names == "":
        num_names = 1
    else:
        try:
            num_names = max(int(num_names), 1)
        except:
            flash("Please type an integer number as number of names.")
            are_inputs_valid = False

    # return if inputs are invalid
    if not are_inputs_valid:
        return render_template("index.html")

    # generate names
    generated_names = generate_artificial_names(seed=seed, num_names=num_names)
    generated_names = [name.capitalize() for name in generated_names]
    app.config["generated_names"] = generated_names.copy()

    return render_template('index.html', generated_names=generated_names)

@app.route('/download')
def download_names():
    """Downloads generated names in txt format"""
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    filename = "temp.txt"
    with open(os.path.join("tmp", filename), "wb") as f:
        f.write("\n".join(app.config["generated_names"]).encode("utf-8"))
    f = open(os.path.join("tmp", filename), "rb")
    resp = send_file(f, as_attachment=True, attachment_filename="generated_names.txt")
    file_remover.cleanup_once_done(resp, "tmp")
    return resp


if __name__ == "__main__":
    app.run(debug=True)

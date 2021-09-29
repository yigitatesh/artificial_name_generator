print("Loading Packages...")

import os, re

# Do not show unnecessary warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# USE FULL POWER OF GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import numpy as np


### Load Data
print("Loading Data...")

with open("data/us_names.txt") as f:
    names = [name.lower() for name in f.readlines()]


### Prepare Data

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

print("Loading the Generator...")
inference_model = tf.keras.models.load_model("model/us_name_generator_inference_model.h5")


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
    for _ in range(17):
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

def is_real_name(name):
    """Checks whether created name is in names dataset or not"""
    return name.strip("<> ") in names_set

def is_seed_valid(seed):
    """Checks whether a given seed is valid or not"""
    if seed == "0":
        return True

    for ch in seed:
        if not ch in all_chars_set:
            return False
    return True


## user input functions
def get_seed_from_user():
    """Gets initial characters (seed) from user"""
    valid = False
    seed = "" #default value

    # user info
    print("\nType initial characters of name(s) to generate.")
    print("(You can directly enter to not indicate any initial character)")
    print("(Type \"0\" without quotes to exit)")

    while not valid:
        seed = input("Your input: ")
        seed = seed.lower().strip()

        if is_seed_valid(seed):
            valid = True
        else:
            print("\nPlease type alphabetical character(s).\n")

    return seed

def get_num_names_from_user():
    """Gets number of names to be generated from user"""
    valid = False
    num_names = 1 #default value

    # user info
    print("\nType number of names to generate.")
    print("(You can directly Enter to generate 1 name)")

    while not valid:
        num_names = input("Your input: ")

        if num_names.strip() == "":
            num_names = 1
            break

        try:
            num_names = max(int(num_names), 1)
            valid = True
        except:
            print("\nPlease type an integer number.\n")

    return num_names


## main function
def main():
    # dummy generation for initialization
    generate_artificial_names()

    print("\nWelcome to the English Name Generator!")
    print("These created names will NOT be REAL NAMES!")
    print("They are being created by an Artificial Intelligence.")

    run = True
    while run:
        ## generate names
        # get initial characters
        seed = get_seed_from_user()

        # exit the generator
        if seed == "0":
            print("\nSee you again")
            break

        # get number of names
        num_names = get_num_names_from_user()

        # generate and print names
        print("\nYour English Names:\n")

        generated_names = generate_artificial_names(seed=seed, num_names=num_names)
        for i, name in enumerate(generated_names):
            print("{}: {}".format(i+1, name))


# START THE GENERATOR APP
if __name__ == "__main__":
    main()

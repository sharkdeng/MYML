## Artificial Intelligence is not a big deal
# keeping a good habit is important
# when there is a good machine
# time for you to shine



## 1 Config
# in jupyter, you can use class Config
# in python (publication code), you can use argparse.ArgumentParser


# example
class Config:
    data = '../data/penn/'
    emsize = 850 # size of word embeddings
    nhid =  850 # number of hidden units per layer
    nhidlast = 850 # number of hidden units for the last rnn layer
    lr = 20 # initial learning rate
    clip = 0.25 # gradient clipping
    epochs = 8000 # upper epoch limit
    batch_size = 64 # batch size
    bptt = 35 # sequence length
    dropout = 0.75 # dropout applied to layers (0 = no dropout)
    dropouth = 0.25 # dropout for hidden nodes in rnn layers (0 = no dropout)
    dropoutx = 0.75 # dropout for input nodes rnn layers (0 = no dropout
    dropouti = 0.2 # dropout for input embedding layers (0 = no dropout)
    dropout3 = 0.1 # dropout to remove words from embedding layer (0 = no dropout)
    seed = 1267 # random seed
    nonmono = 5 # random seed
    cuda = False # use CUDA
    log_interval = 200 # report interval
    save = 'EXP' # path to save the final model
    alpha = 0 # alpha L2 regularization on RNN activation (alpha = 0 means no regularization)
    beta = 1e-3  # beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)
    wdecay = 8e-7 # weight decay applied to all weights
    resume_train = False # continue train from a checkpoint
    small_batch_size = -1 # the batch size for computation. batch_size should be divisible by small_batch_size.\
                            # In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                            # until batch_size is reached. An update step is then performed.
    max_seq_len_detal = 20 # max sequence length
    arch = 'DARTS' # which architecture to u se






## 2 Experiment Logging

# 2-1 create log dir
if not Config.resume_train:
    Config.save = 'E-{}'.format(time.strftime("%Y-%m-%d-%H%M%S"))
    create_exp_dir(Config.save, scripts_to_save=glob.glob('../rnn/*.py'))


# 2-2 creat logger
logging_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=logging_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(Config.save, 'log.txt')
fh.setFormatter(logging.Formatter(logging_format))
logging.getLogger().addHandler(fh)

logging.info('What you want to say?')





## dataset


## model


## 3 count model parameter size (important to report in paper)
total_params = sum(x.data.nelement() for x in model.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model total parameters: {}'.format(total_params))
logging.info('Genotype: {}'.format(genotype))




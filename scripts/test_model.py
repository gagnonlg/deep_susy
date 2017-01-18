import numpy as np
import model

definition = model.ModelDefinition(
    n_hidden_layers=1,
    n_hidden_units=100,
    learning_rate=0.01,
    momentum=0.5,
    l2_reg=1e-6,
    min_epochs=10,
    max_epochs=100,
    patience=1,
    reweight=False,
    normalize=True
)

# ####################################

data_X = np.random.normal([0,0], [1,1], size=[10000,2])
data_Y = np.logical_xor(
    data_X[:,0] > 0,
    data_X[:,1] > 0
).astype(np.float32)

test_X = np.random.normal([0,0], [1,1], size=[1000,2])
test_Y = np.logical_xor(
    data_X[:,0] > 0,
    data_X[:,1] > 0
).astype(np.float32)

# #####################################

definition.train(data_X, data_Y)

###

# def main():

#     # parse the args
#     args = argparse.ArgumentParser()
#     args.add_argument('--data', required=True)
#     args.add_argument('--def', required=True)
#     args = args.parse_args()

#     # load the definition
#     defn = {}
#     execfile(args.def, defn)
#     if 'definition' not in defn:
#         raise RuntimeError('definition not found in file ' + def_path)

#     # load the data
#     data = h5.File(args.data, 'r')
#     trainX = np.array(data['training/inputs'])
#     trainY = np.array(data['training/labels'])
#     testX = np.array(data['validation/inputs'])
#     testY = np.array(data['validation/labels'])

#     # train and eval
#     defn['definition'].train(trainX, trainY).evaluate(testX, testY).save()

########################################################################

# def job_scripts(datapath, defpath):
#     return """
# mkdir ${PBS_JOBID}_${PBS_JOBNAME}
# cd ${PBS_JOBID}_${PBS_JOBNAME}

# git clone ~/dev/deep_susy/git .
# . scripts/setup.sh

# python2 scripts/train_and_evaluate.py --data %s --def %s""" % (datapath, defpath)


# def launch(data, defn):

#     script = job_script(data, defn)

#     cmd = [
#         'qsub'
#         '-d', '/lcg/storage15/atlas/gagnon/work',
#         '-N', 'optimization',
#         '-joe',
#         '-koe',
#     }
    
#     qsub = subprocess.Popen(cmd, stdin=subprocess.PIPE)
#     stdout, stderr = qsub.communicate(job_script(data, defn))

#     return stdout

# def optimize(ntries, data, defn):

#     jobs = []
#     for _ in range(ntries):
#         jobs.append(launch(data, defn))
#         time.sleep(0.5s)

#     launch_wrangler(jobs)
    
        
    

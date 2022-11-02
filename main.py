import os

import numpy as np
import pandas as pd

import sidekit
from data_init import data_inint_main
from extract_features import extract_features_main
from i_vector import i_vector_main

if __name__ == "__main__":
    data_inint_main()
    extract_features_main()
    i_vector_main()

    if(True):
        # Load UBM model
        model_name = "ubm_{}.h5".format(64)
        ubm = sidekit.Mixture()
        ubm.read(os.path.join('./i_vec_out', "ubm", model_name))

        # Load TV matrix
        filename = "tv_matrix_{}".format(64)
        outputPath = os.path.join('./i_vec_out', "ivector", filename)
        fa = sidekit.FactorAnalyser(outputPath+".h5")

        # Extract i-vectors from enrollment data
        print('extracting training i-vectors')
        filename = 'enroll_stat_{}.h5'.format(64)
        enroll_stat = sidekit.StatServer.read(os.path.join('./i_vec_out', 'stat', filename))
        i_vectors_train = fa.extract_ivectors_single( ubm=ubm, stat_server=enroll_stat, uncertainty=False)

        print('saving training i-vectors')
        i_id_train = list(np.array([''.join(['/'+x for x in a.split('/')[1:]])[1:] for a in i_vectors_train.segset]))
        i_label_train = list(np.array([int(label.split('/')[0][2:]) for label in i_vectors_train.modelset], dtype=int))
        i_vec_train = list(np.array(i_vectors_train.stat1))

        d = {'id':i_id_train, 'label':i_label_train, 'xvector':i_vec_train}
        i_vector = pd.DataFrame(data=d)
        i_vector.to_csv('i_vectors/i_vector_train_v2.csv')
        print('saved')

        # Extract i-vectors from testing data
        print('extracting testing i-vectors')
        filename = 'test_stat_{}.h5'.format(64)
        test_stat = sidekit.StatServer.read(os.path.join('./i_vec_out', 'stat', filename))
        i_vectors_train = fa.extract_ivectors_single( ubm=ubm, stat_server=test_stat, uncertainty=False)

        print('saving testing i-vectors')
        i_id_train = list(np.array([''.join(['/'+x for x in a.split('/')[1:]])[1:] for a in i_vectors_train.segset]))
        i_label_train = list(np.array([int(label.split('/')[0][2:]) for label in i_vectors_train.modelset], dtype=int))
        i_vec_train = list(np.array(i_vectors_train.stat1))

        d = {'id':i_id_train, 'label':i_label_train, 'xvector':i_vec_train}
        i_vector = pd.DataFrame(data=d)
        i_vector.to_csv('i_vectors/i_vector_test_v2.csv')
        print('saved')

        # IMPORTANT!
        # After this point the PLDA classifier is used as implemented in the x-vector system
        # Use the PLDA in the x-vector system by simply giving the path to the i-vector .csv file
        # instead of the x-vector .csv file when training the PLDA.

    print("ALL DONE!!")
